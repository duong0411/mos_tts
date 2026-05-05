#!/usr/bin/env python3
"""
Print per-layer hidden stats in the same format as C++ moss_gpt2.c (MOSS_DEBUG_LAYER_STATS_VERBOSE=1):

  [moss_layer] stack=global|local begin call=N ...
  [moss_layer] stack=... call=N layer=L dim=D stage=input|ln1_out|attn_out_pre_proj|...

Uses forward hooks on MossTTSNanoGPT2Block / ln_f (matches moss_gpt2.c row=last token).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, List, Tuple

import torch


def print_moss_cpp_line(stack: str, call: int, layer: int, dim: int, stage: str, t: torch.Tensor) -> None:
    row = t[0, -1].detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    d = int(row.numel())
    if dim > 0 and dim != d:
        dim = d
    else:
        dim = d
    sm = float(row.sum().item())
    l2 = float(torch.linalg.vector_norm(row).item())
    mn = float(row.min().item())
    mx = float(row.max().item())
    print(
        f"[moss_layer] stack={stack} call={call} layer={layer} dim={dim} stage={stage} "
        f"sum={sm:.9g} l2={l2:.9g} min={mn:.9g} max={mx:.9g}",
        flush=True,
    )


class MossCppLayerLogCtx:
    """Shared call counter (same as single static int in moss_gpt2.c)."""

    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls
        self.call_idx = 0
        self.handles: List[Any] = []
        self.forward_restores: List[Tuple[Any, Callable[..., Any]]] = []

    def should_log(self) -> bool:
        return self.call_idx <= self.max_calls

    def wrap_gpt2_forward(self, gpt2: torch.nn.Module, stack: str) -> None:
        orig = gpt2.forward

        def fwd(*a: Any, **kw: Any) -> Any:
            ie = kw.get("inputs_embeds")
            if ie is None:
                raise ValueError("log_python_layer_trace: expected inputs_embeds in GPT2 forward kwargs")
            s_len, d = ie.shape[1], ie.shape[2]
            inner = int(gpt2.config.n_inner or 4 * d)
            self.call_idx += 1
            c = self.call_idx
            if c <= self.max_calls:
                n_layer = len(gpt2.h)
                print(
                    f"[moss_layer] stack={stack} begin call={c} n_layer={n_layer} S={s_len} D={d} I={inner} last_row={s_len - 1}",
                    flush=True,
                )
            return orig(*a, **kw)

        gpt2.forward = fwd  # type: ignore[method-assign]
        self.forward_restores.append((gpt2, orig))

    def install_hooks(self, gpt2: torch.nn.Module, stack: str) -> None:
        d_model = int(gpt2.config.hidden_size)
        n_inner = int(gpt2.config.n_inner or 4 * d_model)
        n_layers = len(gpt2.h)

        def ok() -> bool:
            return self.should_log()

        for li, block in enumerate(gpt2.h):
            if li == 0:

                def ln1_pre_input(m: Any, inp: Tuple[torch.Tensor, ...], stack=stack, ctx=self, D=d_model) -> None:
                    if not ok() or not inp or inp[0] is None:
                        return
                    print_moss_cpp_line(stack, ctx.call_idx, -1, D, "input", inp[0])

                self.handles.append(block.ln_1.register_forward_pre_hook(ln1_pre_input))

            def ln1_out_h(m: Any, inp: Any, out: torch.Tensor, stack=stack, ctx=self, li=li, D=d_model) -> None:
                if ok():
                    print_moss_cpp_line(stack, ctx.call_idx, li, D, "ln1_out", out)

            self.handles.append(block.ln_1.register_forward_hook(ln1_out_h))

            def cproj_pre(
                m: Any, inp: Tuple[torch.Tensor, ...], stack=stack, ctx=self, li=li, D=d_model
            ) -> None:
                if not ok() or not inp or inp[0] is None:
                    return
                print_moss_cpp_line(stack, ctx.call_idx, li, D, "attn_out_pre_proj", inp[0])

            self.handles.append(block.attn.c_proj.register_forward_pre_hook(cproj_pre))

            def ln2_pre_post_attn(
                m: Any, inp: Tuple[torch.Tensor, ...], stack=stack, ctx=self, li=li, D=d_model
            ) -> None:
                if not ok() or not inp or inp[0] is None:
                    return
                print_moss_cpp_line(stack, ctx.call_idx, li, D, "post_attn_res", inp[0])

            self.handles.append(block.ln_2.register_forward_pre_hook(ln2_pre_post_attn))

            def ln2_out_h(m: Any, inp: Any, out: torch.Tensor, stack=stack, ctx=self, li=li, D=d_model) -> None:
                if ok():
                    print_moss_cpp_line(stack, ctx.call_idx, li, D, "ln2_out", out)

            self.handles.append(block.ln_2.register_forward_hook(ln2_out_h))

            def gelu_out_h(m: Any, inp: Any, out: torch.Tensor, stack=stack, ctx=self, li=li, I=n_inner) -> None:
                if ok():
                    print_moss_cpp_line(stack, ctx.call_idx, li, I, "mlp_fc_in_gelu", out)

            self.handles.append(block.mlp.act.register_forward_hook(gelu_out_h))

            def fc_out_h(m: Any, inp: Any, out: torch.Tensor, stack=stack, ctx=self, li=li, D=d_model) -> None:
                if ok():
                    print_moss_cpp_line(stack, ctx.call_idx, li, D, "mlp_fc_out_pre_add", out)

            self.handles.append(block.mlp.fc_out.register_forward_hook(fc_out_h))

            def block_out_h(m: Any, inp: Any, out: Any, stack=stack, ctx=self, li=li, D=d_model) -> None:
                if not ok():
                    return
                h = out[0] if isinstance(out, tuple) else out
                print_moss_cpp_line(stack, ctx.call_idx, li, D, "post_mlp_res", h)

            self.handles.append(block.register_forward_hook(block_out_h))

        def ln_f_h(m: Any, inp: Any, out: torch.Tensor, stack=stack, ctx=self, nL=n_layers, D=d_model) -> None:
            if ok():
                print_moss_cpp_line(stack, ctx.call_idx, nL, D, "output_ln_f", out)

        self.handles.append(gpt2.ln_f.register_forward_hook(ln_f_h))

    def cleanup(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()
        for mod, orig in self.forward_restores:
            mod.forward = orig  # type: ignore[method-assign]
        self.forward_restores.clear()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Log Python GPT2 forwards in [moss_layer] format (same as C++ MOSS_DEBUG_LAYER_STATS_VERBOSE)."
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint dir or HF id (default: infer.py default)",
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--prompt-audio-path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-frames", type=int, default=96)
    parser.add_argument("--do-sample", type=int, default=0, choices=[0, 1])
    parser.add_argument("--text-temperature", type=float, default=1.0)
    parser.add_argument("--audio-temperature", type=float, default=0.8)
    parser.add_argument("--audio-top-p", type=float, default=0.95)
    parser.add_argument("--audio-top-k", type=int, default=25)
    parser.add_argument("--audio-repetition-penalty", type=float, default=1.2)
    parser.add_argument("--out-wav", required=True)
    parser.add_argument(
        "--moss-layer-log-max-calls",
        type=int,
        default=3,
        help="Same idea as MOSS_DEBUG_LAYER_CALLS: only log first N moss_gpt2-style forwards (global+local share one counter).",
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import scipy  # noqa: F401
    except ImportError:
        print(
            "error: scipy required (infer strips user-site). e.g. conda install scipy",
            file=sys.stderr,
        )
        return 2
    import infer as infer_mod  # pylint: disable=import-error

    infer_mod.set_logging()
    infer_argv = [
        "--device",
        "cpu",
        "--seed",
        str(args.seed),
        "--max-new-frames",
        str(args.max_new_frames),
        "--do-sample",
        str(args.do_sample),
        "--text-temperature",
        str(args.text_temperature),
        "--audio-temperature",
        str(args.audio_temperature),
        "--audio-top-p",
        str(args.audio_top_p),
        "--audio-top-k",
        str(args.audio_top_k),
        "--audio-repetition-penalty",
        str(args.audio_repetition_penalty),
        "--prompt-audio-path",
        args.prompt_audio_path,
        "--text",
        args.text,
        "--output-audio-path",
        args.out_wav,
    ]
    if args.checkpoint:
        infer_argv = ["--checkpoint", str(Path(args.checkpoint).expanduser().resolve())] + infer_argv
    infer_args = infer_mod.parse_args(infer_argv)

    device = infer_mod.resolve_device(infer_args.device)
    dtype = infer_mod.resolve_dtype(infer_args.dtype, device)
    if infer_args.seed is not None:
        torch.manual_seed(infer_args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(infer_args.seed)

    ckpt = Path(infer_args.checkpoint).expanduser().resolve()
    shim_dir = None
    if ckpt.is_dir() and not (ckpt / "model.safetensors").is_file() and (ckpt / "pytorch_model.safetensors").is_file():
        tmp = Path(tempfile.mkdtemp(prefix="moss_ckpt_shim_"))
        try:
            for child in ckpt.iterdir():
                if child.name in ("pytorch_model.safetensors", "model.safetensors"):
                    continue
                os.symlink(child.resolve(), tmp / child.name, target_is_directory=False)
            os.symlink((ckpt / "pytorch_model.safetensors").resolve(), tmp / "model.safetensors", target_is_directory=False)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        shim_dir = str(tmp)
        model = infer_mod.load_model(shim_dir, device=device, dtype=dtype)
    else:
        model = infer_mod.load_model(infer_args.checkpoint, device=device, dtype=dtype)
    sampling_kwargs = infer_mod.resolve_sampling_kwargs(infer_args)

    raw_text = infer_mod.resolve_text(infer_args)
    raw_prompt_text = infer_mod.resolve_prompt_text(infer_args) or ""
    enable_wetext_processing = bool(infer_args.enable_wetext_processing) and not bool(infer_args.disable_wetext_processing)
    enable_normalize_tts_text = bool(infer_args.enable_normalize_tts_text) and not bool(
        infer_args.disable_normalize_tts_text
    )
    text_normalizer_manager = None
    if enable_wetext_processing:
        text_normalizer_manager = infer_mod.WeTextProcessingManager()
        snapshot = text_normalizer_manager.ensure_ready()
        if not snapshot.ready:
            raise RuntimeError(snapshot.error or snapshot.message)
    prepared = infer_mod.prepare_tts_request_texts(
        text=raw_text,
        prompt_text=raw_prompt_text,
        voice="",
        enable_wetext=enable_wetext_processing,
        enable_normalize_tts_text=enable_normalize_tts_text,
        text_normalizer_manager=text_normalizer_manager,
    )
    text = str(prepared["text"])
    prompt_text = str(prepared["prompt_text"]).strip() or None

    ctx = MossCppLayerLogCtx(max_calls=int(args.moss_layer_log_max_calls))
    ctx.install_hooks(model.transformer, "global")
    ctx.install_hooks(model.local_transformer, "local")
    ctx.wrap_gpt2_forward(model.transformer, "global")
    ctx.wrap_gpt2_forward(model.local_transformer, "local")

    try:
        result = model.inference(
            text=text,
            output_audio_path=infer_args.output_audio_path,
            mode=infer_args.mode,
            prompt_text=prompt_text,
            prompt_audio_path=infer_args.prompt_audio_path,
            reference_audio_path=infer_args.reference_audio_path,
            text_tokenizer_path=infer_args.text_tokenizer_path,
            audio_tokenizer_type=infer_mod.MOSS_AUDIO_TOKENIZER_TYPE,
            audio_tokenizer_pretrained_name_or_path=infer_args.audio_tokenizer_pretrained_name_or_path,
            device=device,
            nq=infer_args.nq,
            max_new_frames=infer_args.max_new_frames,
            voice_clone_max_text_tokens=infer_args.voice_clone_max_text_tokens,
            voice_clone_max_memory_per_sample_gb=infer_args.voice_clone_max_memory_per_sample_gb,
            do_sample=bool(infer_args.do_sample),
            use_kv_cache=True,
            **sampling_kwargs,
        )
    finally:
        ctx.cleanup()

    print(f"python_frames={int(result['audio_token_ids'].shape[0])}")
    del model
    if shim_dir:
        shutil.rmtree(shim_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
