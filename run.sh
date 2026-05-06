./moss_tts --model-dir ../weight \
  --text "欢迎关注模思智能、上海创智学院与复旦大学自然语言处理实验室。" \
  --prompt-audio-path /media/hdd1/duongpv/VibeVoice/MOSS-TTS-Nano/assets/audio/zh_4.wav \
  --max-new-frames 375 \
  --stream --stream-every 1 --threads 16 \
  --output /media/hdd1/duongpv/tmp/cpp_louder.wav