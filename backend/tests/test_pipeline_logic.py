"""Logic tests only. Signals/tables here are tiny generated arrays used to check CODE behaviour (shapes, labelling rules,
leakage guards, missing-modality handling). They are never used for training, evaluation or demos."""
import os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from multimon import config
from multimon.models.fusion import FusionRiskModel
from multimon.pipeline.features import add_history, history_row, patient_split
from multimon.pipeline.windows import case_windows, sustained
from multimon.serve.inference import LiveEngine
from multimon.signal.ecg import ecg_features
from multimon.signal.ppg_spo2 import ppg_features, spo2_features
from multimon.signal.resp import resp_features

CFG = config.load()
FS = 125


def ecg_wave(bpm, sec, seed=0):
    rng, n = np.random.default_rng(seed), int(sec * FS)
    x = np.zeros(n)
    for b in np.arange(0.4, sec, 60 / bpm):
        t = np.arange(n) / FS - b
        x += np.exp(-t ** 2 / (2 * .012 ** 2)) + .3 * np.exp(-(t - .25) ** 2 / (2 * .05 ** 2)) - .15 * np.exp(-(t + .04) ** 2 / (2 * .01 ** 2))
    return x + rng.normal(0, .02, n)


class Signals(unittest.TestCase):
    def test_ecg_hr(self):
        self.assertAlmostEqual(ecg_features(ecg_wave(75, 60), FS)["ecg_hr_mean"], 75, delta=2)

    def test_ecg_flat_or_gappy_gives_nan_not_crash(self):
        self.assertTrue(np.isnan(ecg_features(np.zeros(60 * FS), FS)["ecg_hr_mean"]))
        x = ecg_wave(75, 60); x[: 20 * FS] = np.nan
        self.assertTrue(np.isnan(ecg_features(x, FS)["ecg_hr_mean"]))

    def test_resp_waveform_and_flat_sensor(self):
        t = np.arange(60 * FS) / FS
        self.assertAlmostEqual(resp_features(np.sin(2 * np.pi * 0.25 * t), FS)["resp_rr_mean"], 15, delta=1)
        flat = resp_features(np.random.default_rng(0).normal(0, 1e-3, 60 * FS), FS)
        self.assertTrue(np.isnan(flat["resp_rr_mean"]) and flat["resp_max_gap_s"] >= 55)

    def test_resp_numeric_fallback(self):
        self.assertAlmostEqual(resp_features(numeric=[14, 15, 16, 15, 14, 15])["resp_rr_mean"], 14.8, delta=.1)

    def test_spo2_artifacts_and_slope(self):
        v = 98 - np.arange(60) * 0.1; v[10] = 40
        f = spo2_features(v, 2.0)
        self.assertLess(f["spo2_slope_per_min"], -2); self.assertLess(f["spo2_valid_frac"], 1)

    def test_ppg(self):
        t = np.arange(60 * FS) / FS
        self.assertAlmostEqual(ppg_features(np.sin(2 * np.pi * 1.2 * t), FS)["ppg_pulse_mean"], 72, delta=3)


class Windows(unittest.TestCase):
    def test_sustained(self):
        self.assertEqual(sustained(np.array([1, 1, 0, 1, 1, 1, 1], bool), 3).tolist(), [0, 0, 0, 1, 1, 1, 1])

    def test_labels_look_ahead_and_skip_events(self):
        dt, n = 2.0, 1800  # one hour
        spo2 = np.full(n, 97.0); spo2[900:960] = 85.0  # event starts at t = 1800 s, lasts 120 s
        case = dict(fs=FS, num_dt=dt, patient="p", spo2=spo2, rr=np.full(n, 15.0))
        d = case_windows(case, CFG)
        H = CFG["window"]["horizon_min"] * 60
        pos = d[d.label == 1]
        self.assertTrue(len(pos) > 0 and (pos.onset_time == 1800).all() and (pos.t_end < 1800).all() and (pos.t_end >= 1800 - H).all())
        quiet = CFG["window"]["quiet_before_min"] * 60
        self.assertFalse(((d.t_end > 1800) & (d.t_end < 1920 + quiet)).any())  # inside/just after event dropped


class Pipeline(unittest.TestCase):
    def test_history_offline_equals_online(self):
        lags = CFG["window"]["history_lags"]
        rng = np.random.default_rng(1)
        df = pd.DataFrame(dict(patient="a", t_end=np.arange(30) * 30.0, spo2_mean=rng.normal(95, 1, 30), ecg_hr_mean=rng.normal(80, 5, 30)))
        off = add_history(df, lags)
        rows = df.to_dict("records")
        for i in (12, 25):
            on = history_row(rows[:i], rows[i], lags)
            for k in lags:
                self.assertAlmostEqual(on[f"spo2_mean_d{k}"], off.loc[i, f"spo2_mean_d{k}"]) if i >= k else self.assertTrue(np.isnan(on[f"spo2_mean_d{k}"]))

    def test_patient_split_has_no_leakage(self):
        df = pd.DataFrame(dict(patient=np.repeat([f"p{i}" for i in range(40)], 5), label=np.repeat([0, 1] * 20, 5)))
        s = patient_split(df, 1, .2, .2)
        ids = [set(v.patient) for v in s.values()]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def toy_table(seed=0, n_pat=40, per=60):
    rng = np.random.default_rng(seed)
    pat = np.repeat([f"p{i}" for i in range(n_pat)], per)
    n = len(pat)
    spo2, hr, rr, temp = rng.normal(96, 3, n), rng.normal(80, 12, n), rng.normal(16, 4, n), rng.normal(36.8, .5, n)
    z = -4 + .5 * (94 - spo2) + .04 * (hr - 80) + .1 * (rr - 16)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return pd.DataFrame(dict(patient=pat, t_end=np.tile(np.arange(per) * 30.0, n_pat), label=y, onset_time=np.where(y == 1, 1e4, np.nan),
                             ecg_hr_mean=hr, spo2_mean=spo2, resp_rr_mean=rr, temp_mean=temp))


class Fusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = toy_table()
        s = patient_split(d, 1, .2, .25)
        cls.s = s
        cls.models = {k: FusionRiskModel(k, 1, .25).fit(s["train"], s["val"]) for k in ("early", "late")}

    def test_levels_monotone_and_thresholds_ordered(self):
        for m in self.models.values():
            self.assertEqual(m.thresholds_, sorted(m.thresholds_))
            p = np.linspace(0, 1, 50)
            self.assertTrue((np.diff(m.level(p)) >= 0).all())

    def test_missing_modality_still_predicts(self):
        te = self.s["test"].copy()
        te["spo2_mean"] = np.nan
        for m in self.models.values():
            self.assertTrue(np.isfinite(m.predict_proba(te)).all())

    def test_signal_beats_chance_on_held_out_patients(self):
        from sklearn.metrics import roc_auc_score
        te = self.s["test"]
        for m in self.models.values():
            self.assertGreater(roc_auc_score(te.label, m.predict_proba(te)), .7)

    def test_contributions_flag_absent_modality(self):
        row = self.s["test"].iloc[[0]].copy(); row["resp_rr_mean"] = np.nan
        c = self.models["late"].contributions(row)
        self.assertIsNone(c["resp"]); self.assertIsNotNone(c["oxy"])


class Live(unittest.TestCase):
    def test_engine_emits_signal_every_chunk_and_assessment_per_stride(self):
        d = toy_table(); s = patient_split(d, 1, .2, .25)
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "m.joblib"); FusionRiskModel("late", 1, .25).fit(s["train"], s["val"]).save(p)
            eng, outs = LiveEngine(p, CFG), []
            ecg = ecg_wave(80, 130)
            for sec in range(130):
                ch = dict(patient="x", t=sec + 1.0, fs=FS, num_dt=2.0, ecg=ecg[sec * FS:(sec + 1) * FS].tolist(), ppg=None, resp=None,
                          spo2=[97] if sec % 2 == 0 else [], rr=[15] if sec % 2 == 0 else [], temp=None, thermal=None)
                sig, ass = eng.push(ch)
                self.assertEqual(sig["type"], "signal")
                if ass: outs.append(ass)
            self.assertTrue(2 <= len(outs) <= 3, len(outs))
            a = outs[-1]
            self.assertIn("Not a diagnosis", a["disclaimer"]); self.assertIn("temp", a["missing"]); self.assertTrue(0 <= a["probability"] <= 1)
            self.assertAlmostEqual(a["measurements"]["hr"], 80, delta=3)


if __name__ == "__main__":
    unittest.main()
