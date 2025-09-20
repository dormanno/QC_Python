from typing import List, Dict
import numpy as np
import pandas as pd

import ColumnNames as Column
from QC_methods.QC_Base import QCMethod

class RobustZQC(QCMethod):
    """
    Per-trade robust Z-score using TRAIN medians and MADs.
    Score = max_abs_robust_z over selected features, clipped at z_cap and mapped to [0,1].
    """

    def __init__(self, features: List[str], z_cap: float = 6.0):
        self.features = features
        self.z_cap = z_cap
        self.median: pd.DataFrame | None = None
        self.mad: pd.DataFrame | None = None
        self._eps = 1e-8

    def fit(self, train_df: pd.DataFrame) -> None:
        g = train_df.groupby(Column.TRADE)
        self.median = g[self.features].median()
        self.mad = g[self.features].apply(lambda x: (x - x.median()).abs().median()).replace(0.0, np.nan)

    def score_day(self, day_df: pd.DataFrame) -> pd.Series:
        assert self.median is not None and self.mad is not None
        vals = []
        for idx, row in day_df.iterrows():
            t = row[Column.TRADE]
            med = self.median.loc[t, self.features]
            mad = self.mad.loc[t, self.features]
            z = (row[self.features] - med) / (1.4826 * mad + self._eps)
            z_max = float(np.nanmax(np.abs(z.values.astype(float)))) if len(z) else 0.0
            vals.append(np.clip(z_max / self.z_cap, 0.0, 1.0))
        s = pd.Series(vals, index=day_df.index, name=Column.ROBUST_Z_SCORE)
        return s
