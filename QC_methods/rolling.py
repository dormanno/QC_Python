from typing import List, Dict
from collections import defaultdict, deque
import numpy as np
import pandas as pd

import ColumnNames as COL_NAME
from QC_methods.QC_Base import QCMethod

class RollingZQC(QCMethod):
    """
    Rolling Z-score per trade per feature using a trailing buffer of fixed length.
    Fit(): warm-up the buffers with TRAIN.
    score_day(): compute max |Z| across features, clipped at z_cap -> [0,1].
    After scoring, call update_state(day_df) to append today's values to buffers.
    """

    def __init__(self, features: List[str], window: int = 20, z_cap: float = 6.0):
        self.features = features
        self.window = window
        self.z_cap = z_cap
        self._eps = 1e-8
        self.buffers: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.window))
        )

    def fit(self, train_df: pd.DataFrame) -> None:
        for _, r in train_df[[COL_NAME.TRADE] + self.features + [COL_NAME.DATE]].sort_values(COL_NAME.DATE).iterrows():
            t = r[COL_NAME.TRADE]
            for f in self.features:
                v = float(r[f]) if pd.notna(r[f]) else np.nan
                if np.isfinite(v):
                    self.buffers[t][f].append(v)

    def score_day(self, day_df: pd.DataFrame) -> pd.Series:
        vals = []
        for idx, row in day_df.iterrows():
            t = row[COL_NAME.TRADE]
            z_max = 0.0
            for f in self.features:
                buf = self.buffers[t][f]
                if len(buf) >= 5:
                    arr = np.asarray(buf, dtype=float)
                    mu = np.nanmean(arr)
                    sd = np.nanstd(arr, ddof=1) if len(arr) > 1 else np.nan
                    z = (float(row[f]) - mu) / (sd + self._eps) if np.isfinite(sd) and sd > 0 else 0.0
                    z_max = max(z_max, abs(z))
            vals.append(float(np.clip(z_max / self.z_cap, 0.0, 1.0)))
        return pd.Series(vals, index=day_df.index, name="Rolling_score")

    def update_state(self, day_df: pd.DataFrame) -> None:
        # Append today's values AFTER scoring (no look-ahead)
        for _, r in day_df[[COL_NAME.TRADE] + self.features].iterrows():
            t = r[COL_NAME.TRADE]
            for f in self.features:
                v = float(r[f]) if pd.notna(r[f]) else np.nan
                if np.isfinite(v):
                    self.buffers[t][f].append(v)
