from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def dump_row(tag: str, idx: int, row: torch.Tensor) -> None:
    vals = [int(v) for v in row.detach().cpu().tolist()]
    print(f"[py_input] {tag} row={idx} " + " ".join(str(v) for v in vals))


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump Python inference input pipeline rows.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--prompt-audio-path", required=True)
    parser.add_argument("--mode", default="voice_clone", choices=("voice_clone", "continuation"))
    parser.add_argument(
        "--audio-tokenizer-pretrained-name-or-path",
        default=None,
        help="Override audio tokenizer path/repo.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nq", type=int, default=None)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import infer as infer_mod  # pylint: disable=import-error

    infer_mod.set_logging()
    infer_args = infer_mod.parse_args(
        [
            "--checkpoint",
            str(Path(__file__).resolve().parents[1] / "weight" / "checkpoint"),
            "--device",
            "cpu",
            "--mode",
            args.mode,
            "--seed",
            str(args.seed),
            "--text",
            args.text,
            "--prompt-audio-path",
            args.prompt_audio_path,
            "--output-audio-path",
            str(Path("/tmp/py_input_dbg.wav")),
        ]
    )

    device = infer_mod.resolve_device(infer_args.device)
    dtype = infer_mod.resolve_dtype(infer_args.dtype, device)
    if infer_args.seed is not None:
        torch.manual_seed(infer_args.seed)
    model = infer_mod.load_model(infer_args.checkpoint, device=device, dtype=dtype)
    model.eval()

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

    text_tokenizer = model._load_text_tokenizer(text_tokenizer=None, text_tokenizer_path=infer_args.text_tokenizer_path)
    audio_tokenizer = model._load_audio_tokenizer(
        audio_tokenizer=None,
        audio_tokenizer_type=infer_mod.MOSS_AUDIO_TOKENIZER_TYPE,
        audio_tokenizer_pretrained_name_or_path=(
            args.audio_tokenizer_pretrained_name_or_path or infer_args.audio_tokenizer_pretrained_name_or_path
        ),
        device=device,
    )
    effective_nq = model._resolve_inference_nq(args.nq)
    target_sample_rate = int(getattr(model.config, "audio_tokenizer_sample_rate", 48000))
    target_channels = int(getattr(model.config, "audio_tokenizer_channels", 2))
    waveform, sample_rate = model._load_reference_audio(infer_args.prompt_audio_path, target_sample_rate, target_channels)
    encoded = model._call_audio_encode(audio_tokenizer=audio_tokenizer, waveform=waveform.to(device), sample_rate=sample_rate)
    prompt_audio_codes = model._mask_unused_audio_channels(model._normalize_audio_codes(encoded), nq=effective_nq).to(device)

    print(f"[py_input] normalized_text={text}")
    print(f"[py_input] prompt_audio_frames={int(prompt_audio_codes.shape[0])} n_vq={int(prompt_audio_codes.shape[1])}")
    tmax = min(3, int(prompt_audio_codes.shape[0]))
    for t in range(tmax):
        vals = [int(v) for v in prompt_audio_codes[t].detach().cpu().tolist()]
        print(f"[py_input] prompt_audio_codes t={t} " + " ".join(str(v) for v in vals))

    input_ids, attn_mask = model.build_inference_input_ids(
        text=text,
        text_tokenizer=text_tokenizer,
        mode=args.mode,
        prompt_text=prompt_text,
        prompt_audio_codes=prompt_audio_codes,
        device=device,
    )
    rows = input_ids[0]
    S = int(rows.shape[0])
    print(f"[py_input] initial_joint_rows={S} row_width={int(rows.shape[1])} mode={args.mode}")
    if S > 0:
        dump_row("joint", 0, rows[0])
    if S > 1:
        dump_row("joint", 1, rows[1])
    if S > 2:
        dump_row("joint", 2, rows[2])
    if S > 3:
        dump_row("joint", S - 3, rows[S - 3])
    if S > 2:
        dump_row("joint", S - 2, rows[S - 2])
    if S > 1:
        dump_row("joint", S - 1, rows[S - 1])
    print(f"[py_input] attn_mask_sum={int(attn_mask.sum().item())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
