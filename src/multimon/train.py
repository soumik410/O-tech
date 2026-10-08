"""Training entry point. Stages: features -> thermal -> train.   python -m multimon.train --stage all"""
from __future__ import annotations
import argparse, json, platform
import numpy as np
import pandas as pd
import sklearn
from . import config
from .evaluate import baseline_model, datacard, level_metrics, window_metrics
from .models.fusion import FusionRiskModel
from .pipeline.features import add_history, patient_split
from .pipeline.windows import build_feature_table


def stage_features(cfg):
    df = build_feature_table(cfg, config.path(cfg, "cache"))
    out = config.path(cfg, "features")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("windows:", len(df), "patients:", df["patient"].nunique())


def stage_thermal(cfg):
    from .data.thermal_loader import load_thermal_table
    from .models.thermal_model import ThermalTempModel
    m = ThermalTempModel().fit(load_thermal_table(cfg))
    art = config.path(cfg, "artifacts")
    art.mkdir(parents=True, exist_ok=True)
    m.save(art / "thermal.joblib")
    print("thermal leave-subject-out MAE (C):", round(m.cv_mae_c_, 3))


def stage_train(cfg):
    t, lags = cfg["train"], cfg["window"]["history_lags"]
    df = add_history(pd.read_csv(config.path(cfg, "features")), lags)
    sp = patient_split(df, t["seed"], t["test_frac"], t["val_frac"])
    tr, va, te = sp["train"], sp["val"], sp["test"]
    report, models = {"datacard": datacard(df), "split_patients": {k: int(v["patient"].nunique()) for k, v in sp.items()}}, {}
    for kind in ("early", "late"):
        m = FusionRiskModel(kind, t["seed"], t["modality_dropout"], t["target_fpr"]).fit(tr, va)
        pv, pt = m.predict_proba(va), m.predict_proba(te)
        report[kind] = dict(val=window_metrics(va["label"].to_numpy(), pv), test=window_metrics(te["label"].to_numpy(), pt),
                            test_levels=level_metrics(te, m.level(pt), cfg["window"]["stride_sec"]), thresholds=[float(x) for x in m.thresholds_])
        models[kind] = m
    report["baseline_4_vitals_logreg"] = {"test": window_metrics(te["label"].to_numpy(), baseline_model(tr, te))}
    best = max(models, key=lambda k: report[k]["val"]["auprc"])  # chosen on VALIDATION patients only
    report["selected"] = best
    report["env"] = dict(python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__)
    art = config.path(cfg, "artifacts")
    art.mkdir(parents=True, exist_ok=True)
    models[best].save(art / "model.joblib")
    (art / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k]["test"] for k in ("early", "late", "baseline_4_vitals_logreg")}, indent=2))
    print("selected:", best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["features", "thermal", "train", "all"], default="all")
    a = ap.parse_args()
    cfg = config.load()
    if a.stage in ("features", "all"):
        stage_features(cfg)
    if a.stage == "thermal":
        stage_thermal(cfg)
    if a.stage in ("train", "all"):
        stage_train(cfg)


if __name__ == "__main__":
    main()
