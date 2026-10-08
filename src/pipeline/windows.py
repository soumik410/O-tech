"""Turns one cached recording into labelled, aligned feature windows.

Label = an adverse event STARTS within the next `horizon_min` minutes (early warning), where an event is a sustained
(>= min_duration_sec) SpO2 < 90, MAP < 55, or HR > 130 / < 40 on the monitor numerics. Windows that already sit inside
(or just after) an event are dropped, so the model must anticipate onsets and cannot just read the current state."""
from __future__ import annotations
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from .features import window_features


def sustained(mask, n):
    """True where `mask` has been True for at least n consecutive samples (marks the whole run)."""
    out = np.zeros(len(mask), bool)
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        if run == n:
            out[i - n + 1:i + 1] = True
        elif run > n:
            out[i] = True
    return out


def event_mask(case, cfg):
    e, dt = cfg["event"], case["num_dt"]
    n = int(e["min_duration_sec"] / dt)
    length = max((len(case[k]) for k in ("spo2", "map", "hr") if k in case), default=0)
    m = np.zeros(length, bool)
    for key, fn in (("spo2", lambda v: v < e["spo2_below"]), ("map", lambda v: v < e["map_below"]),
                    ("hr", lambda v: (v > e["hr_above"]) | (v < e["hr_below"]))):
        if key in case:
            v = np.asarray(case[key], float)
            m[:len(v)] |= sustained(np.isfinite(v) & fn(np.nan_to_num(v, nan=1e9)), n)
    return m


def case_windows(case, cfg):
    W, S, H, Q = (cfg["window"][k] for k in ("feature_sec", "stride_sec", "horizon_min", "quiet_before_min"))
    fs, dt = case["fs"], case["num_dt"]
    em = event_mask(case, cfg)
    if len(em) == 0:
        return pd.DataFrame()
    onsets = np.flatnonzero(em & ~np.r_[False, em[:-1]]) * dt
    dur = len(em) * dt
    rows = []
    for t_end in np.arange(W, dur - H * 60, S):
        a, b = int((t_end - Q * 60) / dt), int(t_end / dt)
        if em[max(a, 0):b].any():
            continue  # inside or right after an event: not an early-warning sample
        seg = {"fs": fs, "num_dt": dt}
        for k in ("ecg", "ppg"):
            if k in case:
                seg[k] = case[k][int((t_end - W) * fs):int(t_end * fs)]
        if "resp" in case:
            seg["resp"] = case["resp"][int((t_end - W) * fs):int(t_end * fs)]
        for k in ("spo2", "rr", "temp"):
            if k in case:
                seg[k] = case[k][int((t_end - W) / dt):int(t_end / dt)]
        f = window_features(seg, cfg)
        if all(np.isnan(v) for v in f.values()):
            continue
        fut = onsets[(onsets > t_end) & (onsets <= t_end + H * 60)]
        rows.append(dict(patient=case["patient"], t_end=float(t_end), label=int(len(fut) > 0),
                         onset_time=float(fut[0]) if len(fut) else np.nan, **f))
    return pd.DataFrame(rows)


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    c = {k: z[k] for k in z.files}
    c["fs"], c["num_dt"], c["patient"] = int(c["fs"]), float(c["num_dt"]), str(c["patient"])
    return c


def build_feature_table(cfg, cache_dir, n_jobs=-1):
    files = sorted(cache_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no cached recordings in {cache_dir}: run the data loaders first")
    parts = Parallel(n_jobs=n_jobs)(delayed(lambda p: case_windows(load_npz(p), cfg))(p) for p in files)
    return pd.concat([p for p in parts if len(p)], ignore_index=True)
