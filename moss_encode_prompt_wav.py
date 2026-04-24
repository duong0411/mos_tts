#!/usr/bin/env python3
"""Encode reference WAV to VQ frames using only MOSS-Audio-Tokenizer-Nano (no full TTS load).

Usage: moss_encode_prompt_wav.py MODEL_DIR WAV_PATH

Stdout: first line = T (number of frames), then T lines with n_vq space-separated integers.

Requires: torch, torchaudio, transformers.
Reads config from MODEL_DIR/config.json or MODEL_DIR/checkpoint/config.json.
Prefers local tokenizer at MODEL_DIR/audio_tokenizer when present.

If you see torchvision/torch mismatch errors, run the parent process with PYTHONNOUSERSITE=1 or fix ~/.local vs conda packages.
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

# Avoid transformers pulling torchvision image stack (often mismatched with torch builds).
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TORCHVISION_DISABLE_NMS_OP", "1")

import torch
import torchaudio
from transformers import AutoModel


def extract_tensor_candidate(output: Any) -> Any:
    if torch.is_tensor(output):
        return output
    for attr_name in ("audio_codes", "audio_token_ids", "codes", "tokens", "input_ids"):
        value = getattr(output, attr_name, None)
        if value is not None:
            return value
    if isinstance(output, dict):
        for key in ("audio_codes", "audio_token_ids", "codes", "tokens", "input_ids"):
            if key in output:
                return output[key]
        if len(output) == 1:
            return next(iter(output.values()))
    if isinstance(output, (list, tuple)) and output:
        if len(output) == 2 and isinstance(output[1], (int, float)):
            return output[0]
        return extract_tensor_candidate(output[0])
    raise TypeError(f"Unsupported audio tokenizer output type: {type(output)!r}")


def extract_audio_code_length(output: Any) -> Optional[int]:
    for attr_name in ("audio_codes_lengths", "audio_token_ids_lengths", "codes_lengths", "lengths"):
        candidate = getattr(output, attr_name, None)
        if candidate is not None:
            lengths = torch.as_tensor(candidate).reshape(-1)
            if lengths.numel() > 0:
                return int(lengths[0].item())
    if isinstance(output, dict):
        for key in ("audio_codes_lengths", "audio_token_ids_lengths", "codes_lengths", "lengths"):
            if key in output:
                lengths = torch.as_tensor(output[key]).reshape(-1)
                if lengths.numel() > 0:
                    return int(lengths[0].item())
    if isinstance(output, (list, tuple)) and len(output) >= 2:
        candidate = output[1]
        if torch.is_tensor(candidate) or isinstance(candidate, (list, tuple)):
            lengths = torch.as_tensor(candidate).reshape(-1)
            if lengths.numel() > 0:
                return int(lengths[0].item())
        if isinstance(candidate, (int, float)):
            return int(candidate)
    return None


def normalize_audio_codes(audio_codes: Any, n_vq: int) -> torch.LongTensor:
    code_length = extract_audio_code_length(audio_codes)
    tensor = torch.as_tensor(extract_tensor_candidate(audio_codes))
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim == 3:
        if tensor.shape[1] == 1 and tensor.shape[0] >= n_vq:
            tensor = tensor[:n_vq, 0, :].transpose(0, 1)
        elif tensor.shape[0] == 1:
            tensor = tensor[0]
        elif tensor.shape[1] == n_vq:
            tensor = tensor.transpose(1, 2)[0]
        elif tensor.shape[-1] == n_vq:
            tensor = tensor[0]
        else:
            raise ValueError(f"Unable to normalize audio codes with shape {tuple(tensor.shape)}")

    if tensor.ndim != 2:
        raise ValueError(f"Expected audio codes with 2 dims after normalization, got {tuple(tensor.shape)}")
    if tensor.shape[-1] != n_vq and tensor.shape[0] == n_vq:
        tensor = tensor.transpose(0, 1)
    elif tensor.shape[-1] != n_vq and tensor.shape[0] > n_vq:
        tensor = tensor[:n_vq].transpose(0, 1)
    elif tensor.shape[-1] > n_vq:
        tensor = tensor[:, :n_vq]
    if tensor.shape[-1] != n_vq:
        raise ValueError(f"Expected trailing dim {n_vq}, got {tuple(tensor.shape)}")
    if code_length is not None:
        tensor = tensor[:code_length]
    return tensor.to(dtype=torch.long)


def mask_unused_audio_channels(
    audio_token_ids: torch.Tensor,
    *,
    nq: int,
    n_vq: int,
    pad_id: int,
) -> torch.LongTensor:
    tensor = torch.as_tensor(audio_token_ids, dtype=torch.long)
    if tensor.shape[-1] != n_vq:
        raise ValueError(f"Expected trailing dim {n_vq}, got {tuple(tensor.shape)}")
    if nq < n_vq:
        tensor = tensor.clone()
        tensor[..., nq:] = int(pad_id)
    return tensor


def resolve_audio_tokenizer_channels(audio_tok: Any) -> int:
    for holder in (audio_tok, getattr(audio_tok, "config", None)):
        if holder is None:
            continue
        for attr_name in ("number_channels", "channels_numbers", "audio_channels", "channels", "num_channels"):
            value = getattr(holder, attr_name, None)
            if value is not None:
                return int(value)
    return 1


def load_reference_audio(path: Path, target_sample_rate: int, target_channels: int) -> tuple[torch.FloatTensor, int]:
    waveform, sample_rate = torchaudio.load(str(path))
    waveform = waveform.to(torch.float32)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    current_channels = int(waveform.shape[0])
    if current_channels == target_channels:
        return waveform, sample_rate
    if current_channels == 1 and target_channels > 1:
        return waveform.repeat(target_channels, 1), sample_rate
    if current_channels > 1 and target_channels == 1:
        return waveform.mean(dim=0, keepdim=True), sample_rate
    raise ValueError(f"Unsupported channel conversion: {current_channels} -> {target_channels}")


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: moss_encode_prompt_wav.py MODEL_DIR WAV_PATH", file=sys.stderr)
        raise SystemExit(2)
    model_dir = Path(sys.argv[1]).expanduser().resolve()
    wav_path = Path(sys.argv[2]).expanduser().resolve()
    if not model_dir.is_dir():
        print(f"model dir not found: {model_dir}", file=sys.stderr)
        raise SystemExit(1)
    if not wav_path.is_file():
        print(f"wav not found: {wav_path}", file=sys.stderr)
        raise SystemExit(1)

    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        cfg_path = model_dir / "checkpoint" / "config.json"
    if not cfg_path.is_file():
        print(f"config.json not found under {model_dir} or {model_dir / 'checkpoint'}", file=sys.stderr)
        raise SystemExit(1)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    n_vq = int(cfg.get("n_vq", 16))
    pad_id = int(cfg.get("audio_pad_token_id", 1024))
    local_tok_dir = model_dir / "audio_tokenizer"
    tok_id = str(local_tok_dir) if local_tok_dir.is_dir() else cfg.get("audio_tokenizer_pretrained_name_or_path")
    if not tok_id:
        print("config.json missing audio_tokenizer_pretrained_name_or_path", file=sys.stderr)
        raise SystemExit(1)
    target_sr = int(cfg.get("audio_tokenizer_sample_rate", 48000))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_kw: dict[str, object] = {"trust_remote_code": True}
    try:
        audio_tok = AutoModel.from_pretrained(str(tok_id), dtype=torch.float32, **load_kw)
    except TypeError:
        audio_tok = AutoModel.from_pretrained(str(tok_id), torch_dtype=torch.float32, **load_kw)
    if hasattr(audio_tok, "to"):
        audio_tok = audio_tok.to(device)
    audio_tok.eval()

    target_ch = resolve_audio_tokenizer_channels(audio_tok)
    waveform, sr = load_reference_audio(wav_path, target_sr, target_ch)
    waveform = waveform.to(device)

    batch_encode = getattr(audio_tok, "batch_encode", None)
    if batch_encode is None:
        print("audio tokenizer has no batch_encode", file=sys.stderr)
        raise SystemExit(1)

    with nullcontext():
        encoded = batch_encode([waveform], chunk_duration=None)

    codes = normalize_audio_codes(encoded, n_vq)
    codes = mask_unused_audio_channels(codes, nq=n_vq, n_vq=n_vq, pad_id=pad_id)
    codes = codes.detach().cpu()

    t = int(codes.shape[0])
    nvq_out = int(codes.shape[1])
    print(t)
    for i in range(t):
        print(" ".join(str(int(codes[i, j])) for j in range(nvq_out)))


if __name__ == "__main__":
    main()
