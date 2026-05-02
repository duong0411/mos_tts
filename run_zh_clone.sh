#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/TT/mos_tts
exec ./moss_tts \
  --model-dir weight \
  --prompt-audio-path /mnt/c/Users/TT/mos_tts/MOSS-TTS-Nano/assets/audio/zh_1.wav \
  --out /mnt/c/Users/TT/mos_tts/zh_ref_tts.wav \
  --frames 96 \
  --min-frames 24 \
  --text "欢迎关注模思智能、上海创智学院与复旦大学自然语言处理实验室。"
