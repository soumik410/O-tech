"""Thermal dataset reader (generic). Expected layout, which you create from whichever real thermal dataset you license:
  <thermal.root>/frames/<file>.npy   2-D float array of temperatures in degrees C (radiometric export)
  <thermal.root>/labels.csv          columns: file, subject, ref_temp_c   (reference core/oral temperature)
Nothing here is generated: if the folder is missing the thermal modality stays off."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from ..signal.thermal import thermal_region_features


def load_thermal_table(cfg):
    root = Path(cfg["thermal"]["root"])
    root = root if root.is_absolute() else Path(cfg["_root"]) / root
    lab = root / "labels.csv"
    if not lab.exists():
        raise FileNotFoundError(f"{lab} not found: provide a real thermal dataset (see module docstring)")
    df, rows = pd.read_csv(lab), []
    for r in df.itertuples():
        f = thermal_region_features(np.load(root / "frames" / f"{r.file}.npy"))
        rows.append(dict(subject=r.subject, ref_temp_c=r.ref_temp_c, **f))
    return pd.DataFrame(rows)
