import wave
from pathlib import Path

import numpy as np


def read_wav_mono(path: Path):
    with wave.open(str(path), "rb") as w:
        ch = w.getnchannels()
        sr = w.getframerate()
        n = w.getnframes()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw != 2:
        raise RuntimeError(f"Only 16-bit PCM supported: {path}")
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def overlap_metrics(x, y):
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = np.linalg.norm(x0) * np.linalg.norm(y0)
    corr = float(np.dot(x0, y0) / denom) if denom > 0 else 0.0
    mse = float(np.mean((x - y) ** 2))
    mae = float(np.mean(np.abs(x - y)))
    rmse = float(np.sqrt(mse))
    return n, corr, rmse, mae


def best_lag_corr(x, y, sr, max_lag_s=1.0):
    ds = 8
    xd = x[::ds]
    yd = y[::ds]
    max_lag = int(sr * max_lag_s) // ds
    best_lag = 0
    best_corr = -2.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a = xd[lag:]
            b = yd[: len(a)]
        else:
            b = yd[-lag:]
            a = xd[: len(b)]
        m = min(len(a), len(b))
        if m < 2000:
            continue
        a = a[:m]
        b = b[:m]
        a0 = a - a.mean()
        b0 = b - b.mean()
        denom = np.linalg.norm(a0) * np.linalg.norm(b0)
        corr = float(np.dot(a0, b0) / denom) if denom > 0 else -2.0
        if corr > best_corr:
            best_corr = corr
            best_lag = lag * ds
    return best_lag, best_corr


base = Path("/mnt/c/Users/TT/mos_tts")
ref = base / "MOSS-TTS-Nano/generated_audio/infer_output.wav"
cands = [
    base / "zh_cpp_like_python.wav",
    base / "zh_cpp_listen_clone.wav",
]

x, sr = read_wav_mono(ref)
print(f"REF\t{ref}\tsr={sr}\tsamples={len(x)}\tduration={len(x)/sr:.4f}s")
print("CAND\tOVERLAP_S\tPEARSON\tRMSE\tMAE\tBEST_LAG_S\tBEST_LAG_CORR")
for p in cands:
    y, sr2 = read_wav_mono(p)
    if sr2 != sr:
        print(f"{p.name}\tSR_MISMATCH({sr2})")
        continue
    n, corr, rmse, mae = overlap_metrics(x, y)
    lag, lag_corr = best_lag_corr(x, y, sr)
    print(f"{p.name}\t{n/sr:.4f}\t{corr:.6f}\t{rmse:.6f}\t{mae:.6f}\t{lag/sr:.4f}\t{lag_corr:.6f}")
