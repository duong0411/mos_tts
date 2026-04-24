#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${1:-../weight}"
TEXT="${2:-MOSS TTS C++ bench}"
OUT_DIR="${3:-/tmp/moss_cpp_bench}"
mkdir -p "${OUT_DIR}"
[[ -x ./moss_tts ]] || make
for backend in auto generic; do
  out="${OUT_DIR}/bench_${backend}.wav"
  start_ns=$(date +%s%N)
  ./moss_tts --model-dir "${MODEL_DIR}" --text "${TEXT}" --out "${out}" --backend "${backend}" --frames 120
  end_ns=$(date +%s%N)
  elapsed_ms=$(( (end_ns - start_ns)/1000000 ))
  size=$(stat -c%s "${out}")
  echo "backend=${backend} elapsed_ms=${elapsed_ms} wav_bytes=${size}"
done
