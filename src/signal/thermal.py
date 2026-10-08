"""Thermal image region features and the temperature-trend features used by the fusion model."""
from __future__ import annotations
import numpy as np

TEMP = ["temp_mean", "temp_last", "temp_slope_per_min"]
REGION = ["th_head_p90", "th_chest_mean", "th_abdomen_mean", "th_legs_mean", "th_chest_lr_asym", "th_hotspot_frac", "th_body_frac"]


def thermal_region_features(frame):
    """frame: 2-D array in degrees C (radiometric). Segments body pixels, splits into head/chest/abdomen/legs."""
    out = dict.fromkeys(REGION, np.nan)
    g = np.asarray(frame, float)
    rows, cols = g.shape
    mask = g > np.percentile(g, 10) + 6
    if mask.mean() < 0.05:
        return out
    R = lambda a, b: slice(int(a * rows), max(int(b * rows), int(a * rows) + 1))
    vals = lambda a, b: g[R(a, b)][mask[R(a, b)]]
    head, chest, abd, legs = vals(0, .2), vals(.2, .5), vals(.5, .65), vals(.65, 1)
    cm, gm = mask[R(.2, .5)], g[R(.2, .5)]
    if len(chest) == 0 or not cm.any():
        return out
    cx = int(np.mean(np.nonzero(cm)[1]))
    left, right = gm[:, :cx][cm[:, :cx]], gm[:, cx:][cm[:, cx:]]
    out.update(th_head_p90=float(np.percentile(head, 90)) if len(head) else np.nan, th_chest_mean=float(chest.mean()),
               th_abdomen_mean=float(abd.mean()) if len(abd) else np.nan, th_legs_mean=float(legs.mean()) if len(legs) else np.nan,
               th_chest_lr_asym=float(np.percentile(left, 90) - np.percentile(right, 90)) if len(left) and len(right) else np.nan,
               th_hotspot_frac=float(np.mean(chest > np.median(chest) + 1.0)), th_body_frac=float(mask.mean()))
    return out


def temp_features(v, dt):
    """v: core temperature numerics (C) every dt s. In serving this series comes from the trained thermal model."""
    out = dict.fromkeys(TEMP, np.nan)
    if v is None:
        return out
    v = np.asarray(v, float)
    good = np.isfinite(v) & (v > 30) & (v < 43)
    if good.sum() < 3:
        return out
    t, s = np.flatnonzero(good) * dt, v[good]
    out.update(temp_mean=float(s.mean()), temp_last=float(s[-1]),
               temp_slope_per_min=float(np.polyfit(t, s, 1)[0] * 60) if len(s) >= 6 else np.nan)
    return out
