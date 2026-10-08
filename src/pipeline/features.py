"""Feature schema, window feature computation (shared by training AND serving), history features, patient split."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..signal import ecg, ppg_spo2, resp, thermal

MODALITIES = {"ecg": ("ecg_",), "oxy": ("ppg_", "spo2_"), "resp": ("resp_",), "temp": ("temp_",)}
META = ["patient", "t_end", "label", "onset_time"]
HISTORY_BASE = ["ecg_hr_mean", "ecg_rmssd_ms", "spo2_mean", "spo2_min", "resp_rr_mean", "ppg_amp_mean", "temp_mean"]


def modality_cols(columns, m):
    return [c for c in columns if c.startswith(MODALITIES[m])]


def window_features(seg, cfg):
    """seg: dict with optional arrays ecg/ppg/resp (at seg['fs']) and spo2/hr/rr/temp numerics (every seg['num_dt'] s),
    all covering the SAME window. Same function runs offline on dataset windows and online on live buffers."""
    fs, dt = seg["fs"], seg["num_dt"]
    f = {}
    f.update(ecg.ecg_features(seg.get("ecg"), fs))
    f.update(ppg_spo2.ppg_features(seg.get("ppg"), fs))
    f.update(ppg_spo2.spo2_features(seg.get("spo2"), dt))
    f.update(resp.resp_features(seg.get("resp"), fs, seg.get("rr"), cfg["train"]["resp_min_amp"]))
    f.update(thermal.temp_features(seg.get("temp"), dt))
    return f


def history_columns(columns, lags):
    return [f"{b}_d{k}" for b in HISTORY_BASE if b in columns for k in lags]


def add_history(df, lags):
    """Offline: value now minus value k windows earlier, per patient (temporal change features)."""
    df = df.sort_values(["patient", "t_end"]).reset_index(drop=True)
    g = df.groupby("patient", sort=False)
    for b in HISTORY_BASE:
        if b in df:
            for k in lags:
                df[f"{b}_d{k}"] = df[b] - g[b].shift(k)
    return df


def history_row(prev_rows, row, lags):
    """Online twin of add_history: prev_rows is the patient's previous feature dicts, oldest first."""
    out = dict(row)
    for b in HISTORY_BASE:
        for k in lags:
            old = prev_rows[-k].get(b, np.nan) if len(prev_rows) >= k else np.nan
            out[f"{b}_d{k}"] = row.get(b, np.nan) - old
    return out


def patient_split(df, seed, test_frac, val_frac):
    """Split by PATIENT (never by window) and stratify on whether the patient ever had an event."""
    from sklearn.model_selection import train_test_split
    p = df.groupby("patient")["label"].max().reset_index()
    tv, te = train_test_split(p, test_size=test_frac, random_state=seed, stratify=p["label"])
    tr, va = train_test_split(tv, test_size=val_frac / (1 - test_frac), random_state=seed, stratify=tv["label"])
    sets = {k: set(v["patient"]) for k, v in (("train", tr), ("val", va), ("test", te))}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    return {k: df[df["patient"].isin(s)].reset_index(drop=True) for k, s in sets.items()}
