"""Multimodal fusion risk model.

Two architectures are trained and compared on held-out patients:
  early: ONE gradient-boosted model over all features (missing modalities are NaN, which the model handles natively)
  late : one model per modality (ECG, oxygen/pulse, respiration, temperature) -> logistic stacker on their log-odds
Both are trained with random modality dropout so a missing sensor at deployment is something the model has seen.
Output is a calibrated probability of an adverse event within the horizon, mapped to levels 0-4 by operating points
measured on validation patients (target false-positive rates from the config), not by hand-picked cut-offs."""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold
from ..pipeline.features import MODALITIES, META, modality_cols

LEVELS = ["NORMAL", "ATTENTION", "WARNING", "HIGH RISK", "EMERGENCY"]
DISCLAIMER = "AI risk estimate for review by qualified clinicians. Not a diagnosis."
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def feature_columns(df):
    return [c for c in df.columns if c not in META and c != "subject"]


def drop_modalities(X, rng, p):
    """Copy of X where each modality is blanked per row with probability p (never all four)."""
    X = X.copy()
    mods = list(MODALITIES)
    gone = rng.random((len(X), len(mods))) < p
    gone[gone.all(axis=1), rng.integers(0, len(mods))] = False
    for j, m in enumerate(mods):
        X.loc[gone[:, j], modality_cols(X.columns, m)] = np.nan
    return X


class FusionRiskModel:
    def __init__(self, kind="late", seed=7, dropout=0.25, target_fpr=None):
        self.kind, self.seed, self.dropout = kind, seed, dropout
        self.target_fpr = {int(k): v for k, v in (target_fpr or {"1": .25, "2": .10, "3": .03, "4": .01}).items()}

    def _hgb(self):
        return HGB(max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0, class_weight="balanced", random_state=self.seed)

    def _logits(self, X):
        """Late fusion inputs: per-modality log-odds and presence flags (absent modality -> 0 and flag 0)."""
        L, P = np.zeros((len(X), len(self.mods_))), np.zeros((len(X), len(self.mods_)))
        for j, m in enumerate(self.mods_):
            cols = self.cols_[m]
            present = X[cols].notna().any(axis=1).to_numpy()
            P[:, j] = present
            if present.any():
                L[present, j] = logit(self.models_[m].predict_proba(X.loc[present, cols])[:, 1])
        return np.hstack([L, P])

    def fit(self, train, val):
        rng = np.random.default_rng(self.seed)
        self.cols_all_ = feature_columns(train)
        self.mods_ = list(MODALITIES)
        self.cols_ = {m: modality_cols(self.cols_all_, m) for m in self.mods_}
        Xt, yt, gt = train[self.cols_all_], train["label"].to_numpy(), train["patient"].to_numpy()
        self.ref_ = {"median": Xt.median(), "iqr": (Xt.quantile(.75) - Xt.quantile(.25)).replace(0, 1)}
        if self.kind == "early":
            Xa = pd.concat([Xt, drop_modalities(Xt, rng, self.dropout)], ignore_index=True)
            self.model_ = self._hgb().fit(Xa, np.r_[yt, yt])
        else:
            self.models_ = {}
            oof = np.zeros((len(Xt), len(self.mods_)))
            pres = np.zeros_like(oof)
            for j, m in enumerate(self.mods_):  # out-of-fold log-odds so the stacker never sees fitted-on-itself scores
                cols, ok = self.cols_[m], Xt[self.cols_[m]].notna().any(axis=1).to_numpy()
                pres[:, j] = ok
                for tr, te in GroupKFold(3).split(Xt, yt, gt):
                    tr, te = tr[ok[tr]], te[ok[te]]
                    if len(np.unique(yt[tr])) == 2 and len(te):
                        oof[te, j] = logit(self._hgb().fit(Xt.iloc[tr][cols], yt[tr]).predict_proba(Xt.iloc[te][cols])[:, 1])
                if ok.sum() and len(np.unique(yt[ok])) == 2:
                    self.models_[m] = self._hgb().fit(Xt[ok][cols], yt[ok])
            S = np.hstack([oof, pres])
            gone = rng.random(pres.shape) < self.dropout  # stacker also learns to cope with absent modalities
            gone[gone.all(axis=1), 0] = False
            S2 = S.copy()
            S2[:, :len(self.mods_)][gone] = 0
            S2[:, len(self.mods_):][gone] = 0
            self.stack_ = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(np.vstack([S, S2]), np.r_[yt, yt])
        raw = self._raw(val[self.cols_all_])
        yv = val["label"].to_numpy()
        self.cal_ = IsotonicRegression(out_of_bounds="clip").fit(raw, yv)
        pv = self.cal_.predict(raw)
        fpr, _, thr = roc_curve(yv, pv)
        th = [thr[np.searchsorted(fpr, self.target_fpr[k], side="right") - 1] for k in (1, 2, 3, 4)]
        self.thresholds_ = list(np.maximum.accumulate(th))
        return self

    def _raw(self, X):
        if self.kind == "early":
            return self.model_.predict_proba(X)[:, 1]
        return self.stack_.predict_proba(self._logits(X))[:, 1]

    def predict_proba(self, df):
        return self.cal_.predict(self._raw(df[self.cols_all_]))

    def level(self, p):
        return np.searchsorted(np.array(self.thresholds_), p, side="right").astype(int)

    def contributions(self, row: pd.DataFrame):
        """Model-agnostic attribution: change in log-odds when one modality is removed (single-row frame)."""
        X = row[self.cols_all_]
        full = logit(self._raw(X))[0]
        out = {}
        for m in self.mods_:
            if X[self.cols_[m]].notna().any(axis=1).iloc[0]:
                Y = X.copy()
                Y[self.cols_[m]] = np.nan
                out[m] = float(full - logit(self._raw(Y))[0])
            else:
                out[m] = None
        return out

    def top_features(self, row: pd.DataFrame, k=2):
        """Per modality: the features that deviate most from the training median (robust z-score)."""
        x = row[self.cols_all_].iloc[0]
        z = (x - self.ref_["median"]) / self.ref_["iqr"]
        return {m: [(c, float(x[c]), float(z[c])) for c in z[self.cols_[m]].abs().sort_values(ascending=False).index[:k] if np.isfinite(z[c])]
                for m in self.mods_}

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
