import argparse
import sys
from pathlib import Path

import torch


def vec_stats(x: torch.Tensor):
    v = x.detach().to("cpu", dtype=torch.float32).reshape(-1)
    return float(v.sum().item()), float(torch.linalg.vector_norm(v).item()), float(v.min().item()), float(v.max().item())


def main() -> int:
    parser = argparse.ArgumentParser(description="Log Python global/local per-layer stats for first calls.")
    parser.add_argument("--repo", required=True)
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
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import infer as infer_mod  # pylint: disable=import-error

    infer_mod.set_logging()
    infer_args = infer_mod.parse_args(
        [
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

    device = infer_mod.resolve_device(infer_args.device)
    dtype = infer_mod.resolve_dtype(infer_args.dtype, device)
    if infer_args.seed is not None:
        torch.manual_seed(infer_args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(infer_args.seed)
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

    g_call = {"n": 0}
    l_call = {"n": 0}
    orig_g = model.transformer.forward
    orig_l = model.local_transformer.forward

    def dump_hidden(prefix: str, call_idx: int, hs):
        # hs: tuple len = n_layer + 1, where hs[0]=input, hs[i+1]=post layer i, hs[-1]=ln_f
        if not hs:
            return
        row = hs[0][0, -1, :]
        s = vec_stats(row)
        print(f"[py_layer] stack={prefix} call={call_idx} layer=-1 stage=input sum={s[0]:.9g} l2={s[1]:.9g} min={s[2]:.9g} max={s[3]:.9g}")
        n_layer = len(hs) - 1
        for li in range(n_layer - 1):
            row = hs[li + 1][0, -1, :]
            s = vec_stats(row)
            print(
                f"[py_layer] stack={prefix} call={call_idx} layer={li} stage=post_mlp_res "
                f"sum={s[0]:.9g} l2={s[1]:.9g} min={s[2]:.9g} max={s[3]:.9g}"
            )
        row = hs[-1][0, -1, :]
        s = vec_stats(row)
        print(f"[py_layer] stack={prefix} call={call_idx} layer={n_layer - 1} stage=output_ln_f sum={s[0]:.9g} l2={s[1]:.9g} min={s[2]:.9g} max={s[3]:.9g}")

    def wrapped_g(*w_args, **w_kwargs):
        g_call["n"] += 1
        w_kwargs["output_hidden_states"] = True
        out = orig_g(*w_args, **w_kwargs)
        if g_call["n"] == 1:
            dump_hidden("global", g_call["n"], out.hidden_states)
        return out

    def wrapped_l(*w_args, **w_kwargs):
        l_call["n"] += 1
        w_kwargs["output_hidden_states"] = True
        out = orig_l(*w_args, **w_kwargs)
        if l_call["n"] == 1:
            dump_hidden("local", l_call["n"], out.hidden_states)
        return out

    model.transformer.forward = wrapped_g
    model.local_transformer.forward = wrapped_l

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
    print(f"python_frames={int(result['audio_token_ids'].shape[0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
