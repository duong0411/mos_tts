"""Quick WAV diagnostics: clipping, quiet fraction, duration (stdlib only)."""
from __future__ import annotations

import array
import sys
import wave
from pathlib import Path


def report(path: Path) -> None:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        n = w.getnframes()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw != 2:
        print(path.name, "not 16-bit pcm")
        return
    a = array.array("h")
    a.frombytes(raw)
    nf = len(a)
    if nf % nch != 0:
        nf -= nf % nch
    fac = 1.0 / 32768.0
    pk = 0.0
    sumsq = 0.0
    n_clip = 0
    n_quiet = 0
    n_frm = nf // nch
    i = 0
    while i < nf:
        mv = 0.0
        for j in range(nch):
            x = a[i + j] * fac
            ax = abs(x)
            if ax > pk:
                pk = ax
            mv = ax if ax > mv else mv
            sumsq += x * x
        if mv >= 0.997:
            n_clip += 1
        if mv < 0.001:
            n_quiet += 1
        i += nch
    rms = (sumsq / nf) ** 0.5 if nf else 0.0
    dur = n_frm / sr if sr else 0
    pct_clip = (100.0 * n_clip / n_frm) if n_frm else 0
    pct_quiet = (100.0 * n_quiet / n_frm) if n_frm else 0
    print(
        f"{path.name}\tdur_s={dur:.3f}\tsr={sr}\tnch={nch}\tpeak={pk:.4f}\trms={rms:.4f}\t"
        f"pct_frm_max_clip(>=0.997)={pct_clip:.2f}%\tpct_frm_quiet(<0.001)={pct_quiet:.2f}%"
    )


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("usage: wav_clip_report.py file.wav [file2.wav ...]", file=sys.stderr)
        sys.exit(2)
    for p in paths:
        if p.is_file():
            report(p)


if __name__ == "__main__":
    main()
