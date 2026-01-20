from typing import List, Dict
import numpy as np
import pandas as pd

from QC_methods.qc_base import QCMethod

class RobustZScoreQC(QCMethod):
    """
    Per-trade robust Z-score using TRAIN medians and MADs.
    Score = max_abs_robust_z over selected features, clipped at z_cap and mapped to [0,1].
    """

    def __init__(self, *, features: List[str], identity_column: str, score_name: str, z_cap: float = 6.0):
        """
        Initialize RobustZScoreQC with specified features and z-score cap.
        
        Args:
            features (List[str]): List of feature column names to use for robust Z-score calculation.
            identity_column (str): Column name for trade/entity identifier.
            z_cap (float, optional): Maximum Z-score value used for clipping. Defaults to 6.0.
        """
        super().__init__(score_name=score_name)
        self.features = features
        self.identity_column = identity_column
        self.z_cap = z_cap
        self.median: pd.DataFrame | None = None
        self.mad: pd.DataFrame | None = None
        self._eps = 1e-8

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Fit the RobustZScoreQC model by computing median and MAD (Median Absolute Deviation) for each trade.
        
        Args:
            train_df (pd.DataFrame): Training DataFrame containing identity_column and feature columns.
                                    Computes per-trade median and MAD for the specified features.
        
        Returns:
            None: Stores computed median and mad as instance variables.
        """
        g = train_df.groupby(self.identity_column)
        self.median = g[self.features].median()
        self.mad = g[self.features].apply(lambda x: (x - x.median()).abs().median()).replace(0.0, np.nan)

    def score_day(self, day_df: pd.DataFrame) -> pd.Series:
        """
        Compute robust Z-scores for each row in the provided DataFrame.
        
        Calculates per-trade robust Z-scores using previously fitted medians and MADs.
        The robust Z-score is computed as max(|Z|) across all features, clipped to [0,1] range.
        
        Args:
            day_df (pd.DataFrame): DataFrame containing identity_column and feature columns to score.
        
        Returns:
            pd.Series: Series of normalized robust Z-scores (values in [0,1]) indexed by day_df's index,
                      with column name ROBUST_Z_SCORE. Returns 0.0 for unseen trades.
        
        Raises:
            AssertionError: If fit() has not been called (median and mad are None).
        """
        assert self.median is not None and self.mad is not None
        vals = []
        for idx, row in day_df.iterrows():
            t = row[self.identity_column]
            if t not in self.median.index:
                vals.append(0.0)  # Default score for unseen trades
            else:
                med = self.median.loc[t, self.features]
                mad = self.mad.loc[t, self.features]
                z = (row[self.features] - med) / (1.4826 * mad + self._eps)
                z_max = float(np.nanmax(np.abs(z.values.astype(float)))) if len(z) else 0.0
                vals.append(np.clip(z_max / self.z_cap, 0.0, 1.0))
        s = pd.Series(vals, index=day_df.index, name=self.ScoreName)
        return s
