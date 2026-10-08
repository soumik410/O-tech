"""BIDMC PPG and Respiration dataset (PhysioNet): 53 ICU recordings of ~8 min with ECG, PPG, respiration waveforms
(125 Hz) and 1 Hz monitor numerics (HR, PULSE, RESP, SpO2). Used for EXTERNAL VALIDATION of our signal code against
the monitor's own numbers, not for training. NOT RUN in the authoring sandbox (no network)."""
from __future__ import annotations
import numpy as np
from multimon import config


def _norm(names):
    return [n.strip().strip(",").upper() for n in names]


def download(cfg):
    import wfdb
    dst = config.path(cfg, "raw") / "bidmc"
    dst.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database(cfg["bidmc"]["database"], str(dst))
    return dst


def load_record(root, n):
    """Returns dict(ecg, ppg, resp @125 Hz; hr, pulse, rr, spo2 @1 Hz) for record number n (1..53)."""
    import wfdb
    name = f"bidmc{n:02d}"
    w = wfdb.rdrecord(str(root / name))
    nm = wfdb.rdrecord(str(root / (name + "n")))
    wn, nn = _norm(w.sig_name), _norm(nm.sig_name)
    pick = lambda rec, names, k: rec.p_signal[:, names.index(k)] if k in names else None
    return dict(fs=int(w.fs), patient=name, ecg=pick(w, wn, "II"), ppg=pick(w, wn, "PLETH"), resp=pick(w, wn, "RESP"),
                hr=pick(nm, nn, "HR"), pulse=pick(nm, nn, "PULSE"), rr=pick(nm, nn, "RESP"), spo2=pick(nm, nn, "SPO2"))
