"""Live inference: rolling per-patient buffers -> the SAME window_features() used in training -> trained model.
Emits a light 'signal' message every chunk (for waveforms) and an 'assessment' every stride."""
from __future__ import annotations
from collections import deque
import numpy as np
import pandas as pd
from ..models.fusion import DISCLAIMER, LEVELS, FusionRiskModel
from ..pipeline.features import history_row, window_features

NICE = {"ecg_hr_mean": "heart rate", "ecg_hr_std": "heart-rate variability", "ecg_rmssd_ms": "beat-to-beat variability", "ecg_irr_frac": "rhythm irregularity",
        "ecg_st_ratio": "ST deviation", "spo2_mean": "SpO2", "spo2_min": "lowest SpO2", "spo2_last": "SpO2", "spo2_slope_per_min": "SpO2 trend",
        "resp_rr_mean": "respiratory rate", "resp_max_gap_s": "longest breathing pause", "temp_mean": "temperature", "temp_last": "temperature"}


class LiveEngine:
    def __init__(self, model_path, cfg, thermal_model=None):
        self.model, self.cfg, self.thermal, self.st = FusionRiskModel.load(model_path), cfg, thermal_model, {}

    def _state(self, pid):
        return self.st.setdefault(pid, dict(buf={}, seen={}, last=-1e9, prev=deque(maxlen=max(self.cfg["window"]["history_lags"]))))

    def push(self, ch):
        W, S, lags = (self.cfg["window"][k] for k in ("feature_sec", "stride_sec", "history_lags"))
        fs, dt, s = ch["fs"], ch["num_dt"], self._state(ch["patient"])
        temp = ch.get("temp")
        if temp is None and ch.get("thermal") is not None and self.thermal is not None:
            temp = [self.thermal.predict(np.asarray(ch["thermal"]))] * max(1, int(1 / dt))  # trained thermal estimate
        for k, x, keep in (("ecg", ch.get("ecg"), W * fs), ("ppg", ch.get("ppg"), W * fs), ("resp", ch.get("resp"), W * fs),
                           ("spo2", ch.get("spo2"), W / dt), ("rr", ch.get("rr"), W / dt), ("temp", temp, W / dt)):
            if x is not None:
                s["buf"][k] = np.r_[s["buf"].get(k, np.array([])), np.asarray(x, float)][-int(keep):]
                s["seen"][k] = ch["t"]
        b = s["buf"]
        for k in [k for k in b if ch["t"] - s["seen"][k] > 10]:
            del b[k]  # sensor silent for 10 s: treat as missing instead of reusing stale data
        last = lambda k, n: b[k][-n:] if k in b else np.array([])
        signal = dict(type="signal", patient=ch["patient"], t=ch["t"], fs=fs, ecg=last("ecg", fs).tolist(), ppg=last("ppg", fs).tolist(),
                      resp=last("resp", fs).tolist(), spo2=(float(last("spo2", 1)[-1]) if "spo2" in b else None))
        need = lambda k: W * fs if k in ("ecg", "ppg", "resp") else W / dt
        # infer only on FULL windows, exactly like training, so there is no train/serve skew
        if ch["t"] - s["last"] < S or not b or not all(len(b[k]) >= 0.98 * need(k) for k in b):
            return signal, None
        s["last"] = ch["t"]
        seg = {"fs": fs, "num_dt": dt, **{k: v for k, v in b.items()}}
        f = window_features(seg, self.cfg)
        row = history_row(list(s["prev"]), f, lags)
        s["prev"].append(f)
        df = pd.DataFrame([row]).reindex(columns=self.model.cols_all_)
        p = float(self.model.predict_proba(df)[0])
        lv = int(self.model.level(np.array([p]))[0])
        contrib, tops = self.model.contributions(df), self.model.top_features(df)
        why = []
        for m, c in sorted(((m, c) for m, c in contrib.items() if c is not None), key=lambda x: -x[1]):
            if c > 0.3:
                why += [f"{NICE.get(n, n)} {v:.1f} (usual {self.model.ref_['median'][n]:.1f})" for n, v, z in tops[m] if z > 1]
        return signal, dict(type="assessment", patient=ch["patient"], t=ch["t"], level=lv, level_name=LEVELS[lv], probability=p,
                            contributions=contrib, explanations=why, missing=[m for m, c in contrib.items() if c is None],
                            measurements=dict(hr=f["ecg_hr_mean"], spo2=f["spo2_last"], rr=f["resp_rr_mean"], temp=f["temp_last"]),
                            horizon_min=self.cfg["window"]["horizon_min"], disclaimer=DISCLAIMER)
