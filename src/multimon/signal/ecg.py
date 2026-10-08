"""ECG preprocessing and window features. Shared by training and live serving (no train/serve skew)."""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

NAMES = ["ecg_hr_mean", "ecg_hr_std", "ecg_hr_min", "ecg_hr_max", "ecg_rmssd_ms", "ecg_sdnn_ms",
         "ecg_irr_frac", "ecg_st_ratio", "ecg_n_beats", "ecg_quality"]


def bandpass(x, fs, lo, hi, order=3):
    return sosfiltfilt(butter(order, [lo, min(hi, 0.45 * fs)], btype="band", fs=fs, output="sos"), x)


def fill_gaps(x):
    """Linear-interpolate short NaN gaps. Returns None if more than 20% of the window is missing."""
    x = np.asarray(x, float)
    bad = ~np.isfinite(x)
    if bad.mean() > 0.2:
        return None
    if bad.any():
        x = x.copy()
        x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad])
    return x


def detect_rpeaks(x, fs):
    """Band-pass 5-20 Hz, squared derivative energy, adaptive threshold, then refine to the R apex."""
    y = bandpass(x, fs, 5, 20)
    w = max(1, int(0.08 * fs))
    e = np.convolve(np.diff(y, prepend=y[0]) ** 2, np.ones(w) / w, "same")
    idx, _ = find_peaks(e, height=0.2 * np.percentile(e, 99), distance=int(0.25 * fs))
    z, h = bandpass(x, fs, 0.5, 40), int(0.06 * fs)
    return np.unique([max(0, i - h) + int(np.argmax(np.abs(z[max(0, i - h):i + h + 1]))) for i in idx])


def ecg_features(x, fs):
    out = dict.fromkeys(NAMES, np.nan)
    if x is None:
        return out
    x = fill_gaps(x)
    if x is None or len(x) < 10 * fs or np.std(x) < 1e-9:
        return out
    r = detect_rpeaks(x, fs)
    rr = np.diff(r) / fs
    ok = (rr > 0.25) & (rr < 2.5)
    if len(r) < 8 or ok.sum() < 6:
        return out
    rrv, med = rr[ok], float(np.median(rr[ok]))
    hr = 60.0 / rrv
    z = bandpass(x, fs, 0.5, 40)
    base, sts = float(np.median(z)), []  # isoelectric level: most of each cycle is flat
    for p in r:
        c, d = p + int(0.06 * fs), p + int(0.10 * fs)
        if p >= int(0.1 * fs) and d < len(z) and abs(z[p] - base) > 1e-9:
            sts.append((np.median(z[c:d]) - base) / abs(z[p] - base))
    out.update(ecg_hr_mean=60.0 / float(np.mean(rrv)), ecg_hr_std=float(np.std(hr)), ecg_hr_min=float(np.percentile(hr, 5)),
               ecg_hr_max=float(np.percentile(hr, 95)), ecg_rmssd_ms=float(np.sqrt(np.mean(np.diff(rrv) ** 2)) * 1000),
               ecg_sdnn_ms=float(np.std(rrv) * 1000), ecg_irr_frac=float(np.mean(np.abs(rrv - med) > 0.2 * med)),
               ecg_st_ratio=float(np.median(sts)) if sts else np.nan, ecg_n_beats=float(len(r)), ecg_quality=float(ok.mean()))
    return out
