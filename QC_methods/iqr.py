from typing import List
import numpy as np
import pandas as pd

import ColumnNames as Column
from QC_methods.QC_Base import QCMethod

class IQRQC(QCMethod):
    """
    Per-trade Tukey fences based score:
      Score = fraction of features outside [Q1-1.5*IQR, Q3+1.5*IQR] in [0,1].
    """

    def __init__(self, features: List[str]):
        self.features = features
        self.q1: pd.DataFrame | None = None
        self.q3: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        g = train_df.groupby(Column.TRADE)
        self.q1 = g[self.features].quantile(0.25)
        self.q3 = g[self.features].quantile(0.75)

    def score_day(self, day_df: pd.DataFrame) -> pd.Series:
        assert self.q1 is not None and self.q3 is not None
        vals = []
        for idx, row in day_df.iterrows():
            t = row[Column.TRADE]
            if t not in self.q1.index:
                vals.append(0.0)  # Default score for unseen trades
            else:
                q1 = self.q1.loc[t, self.features]
                q3 = self.q3.loc[t, self.features]
                iqr = (q3 - q1).replace(0.0, np.nan)
                lo = q1 - 1.5 * iqr
                hi = q3 + 1.5 * iqr
                x = row[self.features].astype(float)
                viol = ((x < lo) | (x > hi)).astype(float)
                vals.append(float(np.nanmean(viol.values)) if len(viol) else 0.0)
        return pd.Series(vals, index=day_df.index, name=Column.IQR_SCORE)
