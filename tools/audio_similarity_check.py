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
    if ch == 2:
        a = a.reshape(-1, 2).mean(axis=1)
    elif ch != 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def overlap_metrics(x: np.ndarray, y: np.ndarray):
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


def best_lag_corr(x: np.ndarray, y: np.ndarray, sr: int, max_lag_s: float = 1.0):
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
        if m <= 0:
            continue
        a = a[:m]
        b = b[:m]
        if len(a) < 2000:
            continue
        a0 = a - a.mean()
        b0 = b - b.mean()
        denom = np.linalg.norm(a0) * np.linalg.norm(b0)
        corr = float(np.dot(a0, b0) / denom) if denom > 0 else -2.0
        if corr > best_corr:
            best_corr = corr
            best_lag = lag * ds
    return best_lag, best_corr


def first_divergence(x: np.ndarray, y: np.ndarray, sr: int, threshold: float = 0.85):
    n = min(len(x), len(y))
    win = 2048
    hop = 512
    for i in range(0, n - win, hop):
        a = x[i : i + win]
        b = y[i : i + win]
        a0 = a - a.mean()
        b0 = b - b.mean()
        denom = np.linalg.norm(a0) * np.linalg.norm(b0)
        corr = float(np.dot(a0, b0) / denom) if denom > 0 else 0.0
        if corr < threshold:
            return i, corr
    return None, None


def main():
    base = Path("/mnt/c/Users/TT/mos_tts")
    ref = base / "zh_clone_sync_greedy.wav"
    cands = [
        base / "MOSS-TTS-Nano/generated_audio/infer_output_sync_greedy.wav",
        base / "MOSS-TTS-Nano/generated_audio/infer_output_sync.wav",
        base / "MOSS-TTS-Nano/generated_audio/infer_output.wav",
    ]

    x, sr = read_wav_mono(ref)
    print(f"REF\t{ref}\tsr={sr}\tmono_samples={len(x)}\tduration={len(x)/sr:.4f}s")
    print("NAME\tOVERLAP_S\tPEARSON\tRMSE\tMAE\tBEST_LAG_S\tBEST_LAG_CORR")

    for p in cands:
        y, sr2 = read_wav_mono(p)
        if sr2 != sr:
            print(f"{p.name}\tSR_MISMATCH({sr2})")
            continue
        n, corr, rmse, mae = overlap_metrics(x, y)
        lag, lag_corr = best_lag_corr(x, y, sr)
        print(f"{p.name}\t{n/sr:.4f}\t{corr:.6f}\t{rmse:.6f}\t{mae:.6f}\t{lag/sr:.4f}\t{lag_corr:.6f}")

    y_greedy, _ = read_wav_mono(cands[0])
    idx, c = first_divergence(x, y_greedy, sr, threshold=0.85)
    if idx is None:
        print("DIVERGENCE\tnot found below corr<0.85 in overlap")
    else:
        print(f"DIVERGENCE\tfirst_window_corr<0.85 at {idx/sr:.4f}s sample={idx} corr={c:.6f}")


if __name__ == "__main__":
    main()
