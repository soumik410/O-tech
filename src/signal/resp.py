"""Respiration features from a breathing waveform (belt/impedance) or from a respiratory-rate numeric."""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks
from .ecg import bandpass, fill_gaps

NAMES = ["resp_rr_mean", "resp_rr_std", "resp_rr_min", "resp_rr_max", "resp_cv", "resp_max_gap_s"]


def _from_intervals(iv, gap):
    rr = 60.0 / iv
    return dict(resp_rr_mean=float(60 / np.mean(iv)), resp_rr_std=float(np.std(rr)), resp_rr_min=float(np.percentile(rr, 5)),
                resp_rr_max=float(np.percentile(rr, 95)), resp_cv=float(np.std(iv) / np.mean(iv)), resp_max_gap_s=gap)


def resp_features(wave=None, fs=None, numeric=None, min_amp=0.2):
    """Prefers the waveform; falls back to numerics (e.g. capnography rate). min_amp is the smallest breath
    amplitude (sensor units) counted as a breath, so a flat or disconnected sensor is not read as breathing."""
    out = dict.fromkeys(NAMES, np.nan)
    if wave is not None:
        x = fill_gaps(wave)
        if x is not None and len(x) >= 20 * fs:
            y = bandpass(x, fs, 0.05, 1.0)
            sd = float(np.std(y))
            pk = find_peaks(y, distance=int(0.9 * fs), prominence=max(0.35 * sd, min_amp))[0] if sd > 1e-9 else np.array([], int)
            dur = len(x) / fs
            if len(pk) < 3:  # no usable breaths: treat as a long pause, rate unknown
                out["resp_max_gap_s"] = float(dur)
                return out
            iv = np.diff(pk) / fs
            return _from_intervals(iv, float(max(iv.max(), dur - pk[-1] / fs)))
    if numeric is not None:
        v = np.asarray(numeric, float)
        v = v[np.isfinite(v) & (v >= 2) & (v <= 60)]
        if len(v) >= 5:
            out.update(resp_rr_mean=float(v.mean()), resp_rr_std=float(v.std()), resp_rr_min=float(np.percentile(v, 5)),
                       resp_rr_max=float(np.percentile(v, 95)), resp_cv=float(v.std() / v.mean()))
    return out
