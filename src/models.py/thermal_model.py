"""Trained thermal -> temperature mapping (replaces any hand-set skin-to-core offset).
Fits on a real thermal dataset with reference temperatures, validated leave-subject-out. At serving time its output
feeds the same `temp_*` features the fusion model learned from monitor core-temperature numerics.
Limit: fusion learned from invasive/probe temperature, thermal gives a camera estimate, so check the error you measure."""
from __future__ import annotations
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, cross_val_predict
from ..signal.thermal import REGION, thermal_region_features


class ThermalTempModel:
    def fit(self, table):
        X, y, g = table[REGION], table["ref_temp_c"].to_numpy(), table["subject"].to_numpy()
        mk = lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05)
        pred = cross_val_predict(mk(), X, y, groups=g, cv=GroupKFold(min(5, len(set(g)))))
        self.cv_mae_c_ = float(mean_absolute_error(y, pred))
        self.model_ = mk().fit(X, y)
        return self

    def predict(self, frame):
        f = thermal_region_features(frame)
        if not np.isfinite(list(f.values())).any():
            return np.nan
        import pandas as pd
        return float(self.model_.predict(pd.DataFrame([f])[REGION])[0])

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
