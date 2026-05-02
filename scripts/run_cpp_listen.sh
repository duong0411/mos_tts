#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TEXT="${1:-这是 C++ moss_tts 语音克隆试听示例。}"
OUT="${2:-zh_cpp_listen_clone.wav}"
exec ./moss_tts \
  --model-dir weight \
  --text "$TEXT" \
  --prompt-audio-path MOSS-TTS-Nano/assets/audio/zh_1.wav \
  --out "$OUT" \
  --frames 64 \
  --min-frames 16 \
  --do-sample 1 \
  --text-temperature 1 \
  --audio-temperature 0.8 \
  --audio-top-p 0.95 \
  --audio-top-k 25 \
  --audio-repetition-penalty 1.2 \
  --seed 0
