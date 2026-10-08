"""VitalDB (open surgical-patient dataset): downloads cases and caches each as one compressed .npz.

Cache schema (one file per case): fs, num_dt, patient, and any of ecg, ppg (waveforms at cfg['fs'] Hz) and
spo2, hr, map, rr, temp (monitor numerics every num_dt seconds). A track that does not exist for a case is simply
absent, and the pipeline treats that modality as missing. NOT RUN in the authoring sandbox (no network): check
track names with `python -m multimon.data.vitaldb_loader --list-tracks` and edit configs/pipeline.json if needed."""
from __future__ import annotations
import argparse
import numpy as np
from scipy.signal import decimate
from multimon import config

WAVES, NUMS = ("ecg", "ppg"), ("spo2", "hr", "map", "rr", "temp")
NATIVE_FS = 500  # SNUADC waveforms


def decimate_nan(x, q):
    """Anti-aliased decimation that keeps NaN gaps (interpolates, decimates, then re-masks)."""
    bad = ~np.isfinite(x)
    if bad.all():
        return np.full(len(x) // q, np.nan)
    x = np.interp(np.arange(len(x)), np.flatnonzero(~bad), x[~bad]) if bad.any() else x
    y = decimate(x, q, ftype="fir", zero_phase=True)
    m = bad[: len(y) * q].reshape(len(y), q).any(axis=1)
    y[m] = np.nan
    return y


def list_tracks(caseid):
    import vitaldb
    return vitaldb.vital_trks(caseid)


def select_cases(cfg):
    """Cases that have at least ECG, PPG and SpO2, sampled reproducibly."""
    import vitaldb
    t = cfg["vitaldb"]["tracks"]
    ids = vitaldb.find_cases([t["ecg"], t["ppg"], t["spo2"]])
    rng = np.random.default_rng(cfg["vitaldb"]["seed"])
    return sorted(rng.choice(ids, min(cfg["vitaldb"]["n_cases"], len(ids)), replace=False).tolist())


def fetch_case(caseid, cfg, out_dir):
    import vitaldb
    t, dt, fs = cfg["vitaldb"]["tracks"], cfg["vitaldb"]["numeric_dt_sec"], cfg["fs"]
    dst = out_dir / f"vitaldb_{caseid:05d}.npz"
    if dst.exists():
        return dst
    rec = {"fs": fs, "num_dt": dt, "patient": f"vitaldb_{caseid}"}
    wv = vitaldb.load_case(caseid, [t[k] for k in WAVES], 1 / NATIVE_FS)
    for i, k in enumerate(WAVES):
        if np.isfinite(wv[:, i]).any():
            rec[k] = decimate_nan(wv[:, i], NATIVE_FS // fs).astype(np.float32)
    nm = vitaldb.load_case(caseid, [t[k] for k in NUMS], dt)
    for i, k in enumerate(NUMS):
        if np.isfinite(nm[:, i]).any():
            rec[k] = nm[:, i].astype(np.float32)
    np.savez_compressed(dst, **rec)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-tracks", type=int, metavar="CASEID")
    a = ap.parse_args()
    cfg = config.load()
    if a.list_tracks:
        print(list_tracks(a.list_tracks))
        return
    out = config.path(cfg, "cache")
    out.mkdir(parents=True, exist_ok=True)
    ids = select_cases(cfg)
    for n, c in enumerate(ids):
        try:
            fetch_case(c, cfg, out)
        except Exception as e:  # one bad case must not stop the download
            print("skip", c, e)
        if n % 10 == 0:
            print(f"{n}/{len(ids)}")


if __name__ == "__main__":
    main()
