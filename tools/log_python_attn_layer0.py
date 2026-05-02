"""Log layer-0 attention pre-softmax stats (eager path) on first global transformer forward.

Matches C++ env MOSS_DEBUG_ATTN_LAYER0=1 diagnostics for call_idx==1:
  [moss_attn] pos_id[last_row=S-1]=...
  [moss_attn] ... pre_softmax nvalid=... sum=... l2=... min=... max=... argmax_j=...

Other layers stay on their default attn implementation (e.g. SDPA); only block0 uses eager.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="Log Python global layer0 attention pre-softmax (eager).")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--prompt-audio-path", required=True)
    parser.add_argument("--out-wav", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-frames", type=int, default=96)
    parser.add_argument("--do-sample", type=int, default=0, choices=[0, 1])
    parser.add_argument("--text-temperature", type=float, default=1.0)
    parser.add_argument("--audio-temperature", type=float, default=0.8)
    parser.add_argument("--audio-top-p", type=float, default=0.95)
    parser.add_argument("--audio-top-k", type=int, default=25)
    parser.add_argument("--audio-repetition-penalty", type=float, default=1.2)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="HF folder with config + weights (default: <repo_root>/weight/checkpoint).",
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    workspace = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint or str(workspace / "weight" / "checkpoint")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import infer as infer_mod  # pylint: disable=import-error

    infer_mod.set_logging()
    parsed = infer_mod.parse_args(
        [
            "--checkpoint",
            checkpoint,
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
    )

    device = infer_mod.resolve_device(parsed.device)
    dtype = infer_mod.resolve_dtype(parsed.dtype, device)
    if parsed.seed is not None:
        torch.manual_seed(parsed.seed)
    model = infer_mod.load_model(parsed.checkpoint, device=device, dtype=dtype)
    sampling_kwargs = infer_mod.resolve_sampling_kwargs(parsed)

    raw_text = infer_mod.resolve_text(parsed)
    raw_prompt_text = infer_mod.resolve_prompt_text(parsed) or ""
    enable_wetext = bool(parsed.enable_wetext_processing) and not bool(parsed.disable_wetext_processing)
    enable_normalize = bool(parsed.enable_normalize_tts_text) and not bool(parsed.disable_normalize_tts_text)
    text_normalizer_manager = None
    if enable_wetext:
        text_normalizer_manager = infer_mod.WeTextProcessingManager()
        snapshot = text_normalizer_manager.ensure_ready()
        if not snapshot.ready:
            raise RuntimeError(snapshot.error or snapshot.message)

    prepared = infer_mod.prepare_tts_request_texts(
        text=raw_text,
        prompt_text=raw_prompt_text,
        voice="",
        enable_wetext=enable_wetext,
        enable_normalize_tts_text=enable_normalize,
        text_normalizer_manager=text_normalizer_manager,
    )
    text = str(prepared["text"])
    prompt_text = str(prepared["prompt_text"]).strip() or None

    block0 = model.transformer.h[0]
    attn0 = block0.attn
    attn0.attn_implementation = "eager"

    g_call = {"n": 0}
    logged = {"pos": False, "scores": False}

    orig_global_forward = model.transformer.forward
    orig_block_forward = block0.forward
    attn_cls = attn0.__class__
    orig_eager = attn_cls._eager_attention

    def wrapped_global_forward(*a, **kw):
        g_call["n"] += 1
        return orig_global_forward(*a, **kw)

    def wrapped_block0_forward(hs, attention_mask=None, position_ids=None, **kw):
        if g_call["n"] == 1 and position_ids is not None and not logged["pos"]:
            logged["pos"] = True
            lr = position_ids.shape[1] - 1
            pid = int(position_ids[0, lr].item())
            print(f"[py_attn] call=1 pos_id[last_row={lr}]={pid}", flush=True)
        out, present = orig_block_forward(hs, attention_mask=attention_mask, position_ids=position_ids, **kw)
        return out, present

    def logged_eager_attention(self, query, key, value, attention_mask):
        query_r = query.transpose(1, 2)
        key_r = key.transpose(1, 2)
        value_r = value.transpose(1, 2)

        scale = 1.0
        if self.scale_attn_weights:
            scale /= float(self.head_dim) ** 0.5
        if self.scale_attn_by_inverse_layer_idx:
            scale /= float(self.layer_idx + 1)

        scores = torch.matmul(query_r, key_r.transpose(-1, -2)) * scale
        causal_mask = self._causal_attention_mask(
            attention_mask=attention_mask,
            query_length=query_r.shape[-2],
            key_length=key_r.shape[-2],
            device=query_r.device,
        )
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)

        if self.layer_idx == 0 and g_call["n"] == 1 and not logged["scores"]:
            logged["scores"] = True
            row = scores[0, 0, -1, :].detach().float().cpu()
            vm = causal_mask[0, 0, -1, :].detach().cpu()
            vals = row[vm]
            nvalid = int(vals.numel())
            sum_v = float(vals.sum())
            l2 = float(torch.linalg.vector_norm(vals).item())
            mn = float(vals.min()) if nvalid else 0.0
            mx = float(vals.max()) if nvalid else 0.0
            idxs = vm.nonzero(as_tuple=False).flatten()
            ai = int(torch.argmax(vals).item()) if nvalid else -1
            argmax_j = int(idxs[ai].item()) if nvalid else -1
            q_row = int(query.shape[1]) - 1
            print(
                f"[py_attn] call=1 layer=0 head=0 q_row={q_row} pre_softmax "
                f"nvalid={nvalid} sum={sum_v:.9g} l2={l2:.9g} min={mn:.9g} max={mx:.9g} argmax_j={argmax_j}",
                flush=True,
            )

        probs = torch.softmax(scores, dim=-1)
        if self.training and self.attn_dropout > 0:
            probs = torch.dropout(probs, self.attn_dropout, train=True)
        output = torch.matmul(probs, value_r)
        return output.transpose(1, 2).contiguous()

    setattr(attn_cls, "_eager_attention", logged_eager_attention)
    block0.forward = wrapped_block0_forward
    model.transformer.forward = wrapped_global_forward

    try:
        result = model.inference(
            text=text,
            output_audio_path=parsed.output_audio_path,
            mode=parsed.mode,
            prompt_text=prompt_text,
            prompt_audio_path=parsed.prompt_audio_path,
            reference_audio_path=parsed.reference_audio_path,
            text_tokenizer_path=parsed.text_tokenizer_path,
            audio_tokenizer_type=infer_mod.MOSS_AUDIO_TOKENIZER_TYPE,
            audio_tokenizer_pretrained_name_or_path=parsed.audio_tokenizer_pretrained_name_or_path,
            device=device,
            nq=parsed.nq,
            max_new_frames=parsed.max_new_frames,
            voice_clone_max_text_tokens=parsed.voice_clone_max_text_tokens,
            voice_clone_max_memory_per_sample_gb=parsed.voice_clone_max_memory_per_sample_gb,
            do_sample=bool(parsed.do_sample),
            use_kv_cache=True,
            **sampling_kwargs,
        )
        print(f"python_frames={int(result['audio_token_ids'].shape[0])}")
    finally:
        setattr(attn_cls, "_eager_attention", orig_eager)
        block0.forward = orig_block_forward
        model.transformer.forward = orig_global_forward
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
