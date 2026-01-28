from typing import List, Dict
from collections import defaultdict, deque
import numpy as np
import pandas as pd

from QC_methods.qc_base import StatefulQCMethod

class RollingZScoreQC(StatefulQCMethod):
    """
    Rolling Z-score per trade per feature using a trailing buffer of fixed length.
    Fit(): warm-up the buffers with TRAIN.
    score_day(): compute max |Z| across features, clipped at z_cap -> [0,1].
    After scoring, call update_state(day_df) to append today's values to buffers.
    """

    def __init__(self, *, features: List[str], identity_column: str, temporal_column: str, score_name: str, window: int = 20, z_cap: float = 6.0):
        """
        Initialize RollingZScoreQC with features and rolling window parameters.
        
        Args:
            features (List[str]): List of feature column names to compute rolling Z-scores for.
            identity_column (str): Column name for trade/entity identifier.
            temporal_column (str): Column name for date/temporal identifier.
            window (int, optional): Size of the rolling window buffer for each trade-feature. Defaults to 20.
            z_cap (float, optional): Maximum Z-score value used for clipping. Defaults to 6.0.
        """
        super().__init__(score_name=score_name)
        self.features = features
        self.identity_column = identity_column
        self.temporal_column = temporal_column
        self.window = window
        self.z_cap = z_cap
        self._eps = 1e-8
        self.buffers: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.window))
        )

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Warm-up rolling buffers with training data, sorted by temporal column.
        
        Args:
            train_df (pd.DataFrame): Training DataFrame containing identity_column, temporal_column, and feature columns.
                                    Data is sorted by temporal_column before processing to maintain temporal order.
        
        Returns:
            None: Populates the rolling buffers for each trade-feature combination with training values.
        """
        for _, r in train_df[[self.identity_column] + self.features + [self.temporal_column]].sort_values(self.temporal_column).iterrows():
            t = r[self.identity_column]
            for f in self.features:
                v = float(r[f]) if pd.notna(r[f]) else np.nan
                if np.isfinite(v):
                    self.buffers[t][f].append(v)

    def _score_day_impl(self, day_df: pd.DataFrame) -> pd.Series:
        """
        Compute rolling Z-scores for each row, using the historical window of buffered values.
        
        Calculates Z-scores based on mean and std of the rolling window for each trade-feature.
        Returns the maximum absolute Z-score across features, clipped to [0,1] range.
        Requires at least 5 values in buffer to compute statistics.
        
        Args:
            day_df (pd.DataFrame): DataFrame containing identity_column and feature columns to score.
        
        Returns:
            pd.Series: Series of normalized rolling Z-scores (values in [0,1]) indexed by day_df's index,
                      with column name ROLLING_SCORE. Returns 0.0 for unseen trades or insufficient buffer.
        
        Notes:
            - Call update_state() after scoring to add today's values to buffers (prevents look-ahead bias).
        """
        vals = []
        for idx, row in day_df.iterrows():
            t = row[self.identity_column]
            if t not in self.buffers:
                vals.append(0.0)  # Default score for unseen trades
            else:
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
        return pd.Series(vals, index=day_df.index, name=self.ScoreName)

    def _update_state_impl(self, day_df: pd.DataFrame) -> None:
        """
        Update rolling buffers by appending today's feature values after scoring.
        
        This should be called AFTER score_day() to prevent look-ahead bias. Buffers automatically
        maintain their fixed size (window) by dropping oldest values when full.
        
        Args:
            day_df (pd.DataFrame): DataFrame containing identity_column and feature columns.
                                  Only finite values are appended to the buffers.
        
        Returns:
            None: Updates internal buffers in-place.
        """
        # Append today's values AFTER scoring (no look-ahead)
        for _, r in day_df[[self.identity_column] + self.features].iterrows():
            t = r[self.identity_column]
            for f in self.features:
                v = float(r[f]) if pd.notna(r[f]) else np.nan
                if np.isfinite(v):
                    self.buffers[t][f].append(v)
