"""Pulse waveform (PPG) features and SpO2 numeric features."""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks
from .ecg import bandpass, fill_gaps

PPG = ["ppg_pulse_mean", "ppg_amp_mean", "ppg_amp_cv", "ppg_quality"]
SPO2 = ["spo2_mean", "spo2_min", "spo2_last", "spo2_slope_per_min", "spo2_valid_frac"]


def ppg_features(x, fs):
    out = dict.fromkeys(PPG, np.nan)
    if x is None:
        return out
    x = fill_gaps(x)
    if x is None or len(x) < 10 * fs or np.std(x) < 1e-9:
        return out
    y = bandpass(x, fs, 0.5, 8)
    pk, _ = find_peaks(y, distance=int(0.3 * fs), prominence=0.3 * np.std(y))
    if len(pk) < 8:
        return out
    iv = np.diff(pk) / fs
    ok = (iv > 0.3) & (iv < 2.0)
    amps = np.array([y[p] - y[max(0, p - int(0.5 * fs)):p].min() for p in pk[1:]])
    out.update(ppg_pulse_mean=float(60 / np.mean(iv[ok])) if ok.any() else np.nan, ppg_amp_mean=float(np.mean(amps)),
               ppg_amp_cv=float(np.std(amps) / (np.mean(amps) + 1e-9)), ppg_quality=float(ok.mean()))
    return out


def spo2_features(v, dt):
    """v: SpO2 numerics (%), sampled every dt seconds. Values outside 50-100 are probe artefacts."""
    out = dict.fromkeys(SPO2, np.nan)
    if v is None or len(v) == 0:
        return out
    v = np.asarray(v, float)
    good = np.isfinite(v) & (v >= 50) & (v <= 100)
    out["spo2_valid_frac"] = float(good.mean())
    if good.sum() < 5:
        return out
    t, s = np.flatnonzero(good) * dt, v[good]
    out.update(spo2_mean=float(s.mean()), spo2_min=float(s.min()), spo2_last=float(np.median(s[-5:])),
               spo2_slope_per_min=float(np.polyfit(t, s, 1)[0] * 60) if len(s) >= 10 else np.nan)
    return out
