"""Print peak/RMS stats for WAV files."""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        print("usage: wav_peak.py file1.wav [file2.wav ...]", file=sys.stderr)
        sys.exit(2)
    for p in paths:
        path = Path(p).resolve()
        with wave.open(str(path), "rb") as w:
            nch = w.getnchannels()
            sr = w.getframerate()
            n = w.getnframes()
            sw = w.getsampwidth()
            raw = w.readframes(n)
        if sw != 2:
            print(path.name, "not 16-bit")
            continue
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
        if nch > 1:
            a = a.reshape(-1, nch)
        pk = float(np.abs(a).max()) if a.size else 0.0
        rms = float(np.sqrt(np.mean(a * a))) if a.size else 0.0
        print(f"{path.name}\tsr={sr}\tnch={nch}\tframes={n}\tpeak={pk:.8g}\trms={rms:.8g}")


if __name__ == "__main__":
    main()
