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
        """
        Initialize IQRQC with specified features for IQR-based outlier detection.
        
        Args:
            features (List[str]): List of feature column names to use for IQR-based scoring.
        """
        self.features = features
        self.q1: pd.DataFrame | None = None
        self.q3: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Fit the IQR model by computing Q1 and Q3 (25th and 75th percentiles) per trade.
        
        Args:
            train_df (pd.DataFrame): Training DataFrame containing TRADE column and feature columns.
                                    Computes quartiles per trade for the specified features.
        
        Returns:
            None: Stores computed Q1 and Q3 as instance variables.
        """
        g = train_df.groupby(Column.TRADE)
        self.q1 = g[self.features].quantile(0.25)
        self.q3 = g[self.features].quantile(0.75)

    def score_day(self, day_df: pd.DataFrame) -> pd.Series:
        """
        Compute Tukey fence-based outlier scores for each row.
        
        Calculates the fraction of features outside the [Q1-1.5*IQR, Q3+1.5*IQR] bounds
        (standard Tukey fence definition for outliers).
        
        Args:
            day_df (pd.DataFrame): DataFrame containing TRADE column and feature columns to score.
        
        Returns:
            pd.Series: Series of outlier scores (values in [0,1]) indexed by day_df's index,
                      with column name IQR_SCORE. Score represents the fraction of features
                      that violate Tukey fence bounds. Returns 0.0 for unseen trades.
        
        Raises:
            AssertionError: If fit() has not been called (Q1 and Q3 are None).
        """
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
