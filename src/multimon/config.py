from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]

DEFAULTS = {
    "paths": {
        "raw": "data/raw",
        "cache": "data/cache",
        "features": "data/features.csv.gz",
        "artifacts": "artifacts",
    },
    "fs": 125,
    "window": {
        "feature_sec": 60,
        "stride_sec": 30,
        "horizon_min": 10,
        "quiet_before_min": 5,
        "history_lags": [10, 20],
    },
    "event": {
        "spo2_below": 90,
        "map_below": 55,
        "hr_above": 130,
        "hr_below": 40,
        "min_duration_sec": 60,
    },
    "vitaldb": {
        "n_cases": 300,
        "seed": 7,
        "numeric_dt_sec": 2,
        "tracks": {
            "ecg": "SNUADC/ECG_II",
            "ppg": "SNUADC/PLETH",
            "spo2": "Solar8000/PLETH_SPO2",
            "hr": "Solar8000/HR",
            "map": "Solar8000/ART_MBP",
            "rr": "Primus/RR_CO2",
            "temp": "Solar8000/BT",
        },
    },
    "bidmc": {"database": "bidmc"},
    "thermal": {"root": "data/raw/thermal"},
    "train": {
        "seed": 7,
        "test_frac": 0.2,
        "val_frac": 0.2,
        "modality_dropout": 0.25,
        "target_fpr": {"1": 0.25, "2": 0.10, "3": 0.03, "4": 0.01},
        "resp_min_amp": 0.2,
    },
}


def load(path=None):
    """Load configuration defaults and optionally override them from a JSON file."""
    cfg = json.loads(json.dumps(DEFAULTS))
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as f:
            overrides = json.load(f)
        cfg = _merge(cfg, overrides)
    return cfg


def _merge(base, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _merge(base[key], value)
        else:
            base[key] = value
    return base


def path(cfg, name):
    """Resolve a configured path relative to the project root."""
    return ROOT / cfg["paths"][name]
