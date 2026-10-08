"""Held-out evaluation. Reports window-level and EVENT-level metrics, because clinicians care whether each event was
caught early, and how many false alarms they get per patient-hour, not just AUROC."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

BASELINE_COLS = ["ecg_hr_mean", "spo2_mean", "resp_rr_mean", "temp_mean"]


def window_metrics(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    m = dict(n=int(len(y)), positives=int(y.sum()), auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
             brier=float(brier_score_loss(y, p)))
    for sp in (0.90, 0.95, 0.99):
        m[f"sensitivity_at_spec_{int(sp * 100)}"] = float(np.interp(1 - sp, fpr, tpr))
    return m


def level_metrics(df, levels, stride_sec):
    """Per level threshold k: window sensitivity, false alarms per patient-hour, event sensitivity and median lead time."""
    d = df[["patient", "t_end", "label", "onset_time"]].copy()
    d["level"] = levels
    out = {}
    neg_hours = (d["label"] == 0).sum() * stride_sec / 3600
    for k in (1, 2, 3, 4):
        flag = d["level"] >= k
        ev = d[d["label"] == 1].groupby(["patient", "onset_time"])
        caught = ev.apply(lambda g: bool((g["level"] >= k).any()), include_groups=False)
        lead = ev.apply(lambda g: (g["onset_time"].iloc[0] - g.loc[g["level"] >= k, "t_end"].min()) / 60 if (g["level"] >= k).any() else np.nan,
                        include_groups=False)
        out[f"level>={k}"] = dict(window_sensitivity=float(flag[d["label"] == 1].mean()),
                                  false_alarms_per_patient_hour=float((flag & (d["label"] == 0)).sum() / max(neg_hours, 1e-9)),
                                  event_sensitivity=float(caught.mean()) if len(caught) else None, events=int(len(caught)),
                                  median_lead_min=float(np.nanmedian(lead)) if len(lead) and np.isfinite(lead).any() else None)
    return out


def baseline_model(train, test):
    """Simple logistic regression on four current vital signs. The complex model must beat this to justify itself."""
    mu = train[BASELINE_COLS].median()
    f = lambda d: d[BASELINE_COLS].fillna(mu)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(f(train), train["label"])
    return lr.predict_proba(f(test))[:, 1]


def datacard(df):
    """Measured dataset documentation: subjects, windows, hours, class balance, missing rate per modality."""
    from .pipeline.features import MODALITIES, modality_cols
    cols = [c for c in df.columns if c not in ("patient", "t_end", "label", "onset_time")]
    return dict(patients=int(df["patient"].nunique()), windows=int(len(df)), positive_rate=float(df["label"].mean()),
                patients_with_event=int(df.groupby("patient")["label"].max().sum()),
                missing_rate={m: float(df[modality_cols(cols, m)].isna().all(axis=1).mean()) for m in MODALITIES})
