#!/usr/bin/env python3
"""
Normalize text the same way as MOSS-TTS-Nano infer.py before model.inference().

Typical usage (from MOSS-TTS-Nano repo root):
  python3 cpp/tools/prepare_infer_text_for_cpp.py --text "..." > normalized.txt
  ./moss_tts --model-dir weight --text-file normalized.txt --out out.wav ...

This matches prepare_tts_request_texts() + WeText defaults from infer.py.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# .../MOSS-TTS-Nano/cpp/tools/this.py -> repo root .../MOSS-TTS-Nano (where text_normalization_pipeline.py lives)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if (_REPO_ROOT / "text_normalization_pipeline.py").is_file():
    sys.path.insert(0, str(_REPO_ROOT))

from text_normalization_pipeline import (  # noqa: E402
    WeTextProcessingManager,
    prepare_tts_request_texts,
)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text", default=None, help="UTF-8 text (or use stdin).")
    p.add_argument("--prompt-text", default="", help="continuation prompt_text only; voice_clone ignores.")
    p.add_argument("--disable-wetext", action="store_true", help="Like infer.py --disable-wetext-processing.")
    p.add_argument(
        "--disable-normalize-tts-text",
        action="store_true",
        dest="disable_norm",
        help="Like infer.py --disable-normalize-tts-text.",
    )
    args = p.parse_args()
    raw = args.text if args.text is not None else sys.stdin.read()
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    enable_wetext = not args.disable_wetext
    wetext_mgr = None
    if enable_wetext:
        wetext_mgr = WeTextProcessingManager()
        snap = wetext_mgr.ensure_ready()
        if not snap.ready:
            print(snap.error or snap.message or "WeTextProcessing not ready", file=sys.stderr)
            sys.exit(1)

    prepared = prepare_tts_request_texts(
        text=raw.rstrip("\n"),
        prompt_text=args.prompt_text or "",
        voice="",
        enable_wetext=enable_wetext,
        enable_normalize_tts_text=not args.disable_norm,
        text_normalizer_manager=wetext_mgr,
    )
    out = str(prepared["text"])
    sys.stdout.buffer.write(out.encode("utf-8"))


if __name__ == "__main__":
    main()
