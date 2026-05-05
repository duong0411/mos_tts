#!/usr/bin/env python3
"""
Export reference-audio RVQ codes exactly like infer.py (PyTorch audio tokenizer),
for use with moss_tts --prompt-audio-codes FILE (same line format as read_prompt_audio_codes_file).

Without this, C++ encodes WAV via moss_audio_tok_encode_wav_file; small differences vs Python
change the whole joint and generation length (e.g. 33 vs 75 frames) even when --text-file matches WeText.

Example:
  python3 cpp/tools/export_prompt_audio_codes_for_cpp.py \\
    --prompt-audio-path assets/audio/zh_3.wav \\
    --out zh_3_prompt_codes.txt

  ./moss_tts --model-dir weight --prompt-audio-codes zh_3_prompt_codes.txt \\
    --text-file normalized.txt --out out.wav ...
"""
from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch


def load_tts_model_for_export(checkpoint: str, device: torch.device, dtype: torch.dtype, infer_mod):
    """
    transformers>=4.5x resolves local dirs to model.safetensors only; many trees ship
    pytorch_model.safetensors (same weights as HF hub). Symlink shim avoids from_config quirks.
    """
    p = Path(checkpoint).expanduser().resolve()
    if not p.is_dir():
        model = infer_mod.load_model(checkpoint, device, dtype)
        setattr(model, "_moss_ckpt_shim_dir", None)
        return model
    if (p / "model.safetensors").is_file():
        model = infer_mod.load_model(str(p), device, dtype)
        setattr(model, "_moss_ckpt_shim_dir", None)
        return model
    legacy = p / "pytorch_model.safetensors"
    if not legacy.is_file():
        model = infer_mod.load_model(checkpoint, device, dtype)
        setattr(model, "_moss_ckpt_shim_dir", None)
        return model
    tmp = Path(tempfile.mkdtemp(prefix="moss_ckpt_shim_"))
    try:
        for child in p.iterdir():
            if child.name in ("pytorch_model.safetensors", "model.safetensors"):
                continue
            os.symlink(child.resolve(), tmp / child.name, target_is_directory=False)
        os.symlink(legacy.resolve(), tmp / "model.safetensors", target_is_directory=False)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    model = infer_mod.load_model(str(tmp), device, dtype)
    setattr(model, "_moss_ckpt_shim_dir", str(tmp))
    return model


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=None, help="MOSS-TTS-Nano repo root (default: infer parent)")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint dir (model.safetensors or pytorch_model.safetensors) or HF id (default: <repo>/weight/checkpoint)",
    )
    p.add_argument("--prompt-audio-path", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Text file: line1=T, then T lines of n_vq ints")
    p.add_argument("--device", default="cpu")
    p.add_argument("--nq", type=int, default=None)
    p.add_argument(
        "--audio-tokenizer-pretrained-name-or-path",
        default=None,
        help="Override audio tokenizer path (default: infer.py default)",
    )
    args = p.parse_args()

    repo = args.repo.resolve() if args.repo is not None else Path(__file__).resolve().parents[2]
    if not (repo / "infer.py").is_file():
        print(f"error: --repo does not look like MOSS-TTS-Nano root: {repo}", file=sys.stderr)
        return 2
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    # infer.py removes user-site from sys.path; scipy often lives there while transformers still imports it.
    try:
        import scipy  # noqa: F401
    except ImportError:
        print(
            "error: scipy required (e.g. `conda install scipy` in this env). infer.py strips user-site from sys.path.",
            file=sys.stderr,
        )
        return 2

    import infer as infer_mod  # noqa: E402

    ckpt = args.checkpoint
    if ckpt is None:
        ckpt = str((repo / "weight" / "checkpoint").resolve())

    infer_args = infer_mod.parse_args(
        [
            "--checkpoint",
            ckpt,
            "--device",
            args.device,
            "--text",
            ".",
            "--output-audio-path",
            str(Path("/tmp/export_codes_dummy.wav")),
            "--prompt-audio-path",
            str(args.prompt_audio_path.resolve()),
        ]
    )
    device = infer_mod.resolve_device(infer_args.device)
    dtype = infer_mod.resolve_dtype(infer_args.dtype, device)
    model = load_tts_model_for_export(ckpt, device, dtype, infer_mod)
    model.eval()

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
    wav_path = str(args.prompt_audio_path.resolve())
    waveform, sample_rate = model._load_reference_audio(wav_path, target_sample_rate, target_channels)
    encoded = model._call_audio_encode(
        audio_tokenizer=audio_tokenizer, waveform=waveform.to(device), sample_rate=sample_rate
    )
    prompt_audio_codes = model._mask_unused_audio_channels(
        model._normalize_audio_codes(encoded), nq=effective_nq
    ).to("cpu")

    t = int(prompt_audio_codes.shape[0])
    nq = int(prompt_audio_codes.shape[1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="ascii") as f:
        f.write(f"{t}\n")
        for i in range(t):
            row = [int(x) for x in prompt_audio_codes[i].tolist()]
            if len(row) != nq:
                print("internal error: row width", file=sys.stderr)
                return 3
            f.write(" ".join(str(x) for x in row) + "\n")

    print(f"wrote {t} frames x {nq} codes -> {args.out}", file=sys.stderr)
    print("use: moss_tts ... --prompt-audio-codes <that-file> --text-file normalized.txt ...", file=sys.stderr)

    shim = getattr(model, "_moss_ckpt_shim_dir", None)
    del model
    del audio_tokenizer
    gc.collect()
    if shim:
        shutil.rmtree(shim, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
