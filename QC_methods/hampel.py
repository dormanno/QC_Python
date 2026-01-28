from typing import List, Dict
from collections import defaultdict, deque
import numpy as np
import pandas as pd

from QC_methods.qc_base import StatefulQCMethod

class HampelFilterQC(StatefulQCMethod):
    """
    Hampel Filter per trade per feature using a trailing buffer of fixed length.
    
    The Hampel filter is a robust outlier detection method that uses median and MAD 
    (Median Absolute Deviation) instead of mean and standard deviation. This makes it
    more resistant to outliers in the rolling window.
    
    Fit(): warm-up the buffers with TRAIN data.
    score_day(): compute max |robust_Z| across features, clipped at threshold -> [0,1].
    After scoring, call update_state(day_df) to append today's values to buffers.
    """

    def __init__(self, *, features: List[str], identity_column: str, temporal_column: str, 
                 score_name: str, window: int = 20, threshold: float = 3.0):
        """
        Initialize HampelFilterQC with features and rolling window parameters.
        
        Args:
            features (List[str]): List of feature column names to apply Hampel filter to.
            identity_column (str): Column name for trade/entity identifier.
            temporal_column (str): Column name for date/temporal identifier.
            score_name (str): Name for the output score column.
            window (int, optional): Size of the rolling window buffer for each trade-feature. Defaults to 20.
            threshold (float, optional): Threshold for robust Z-score (typically 3.0). Defaults to 3.0.
        """
        super().__init__(score_name=score_name)
        self.features = features
        self.identity_column = identity_column
        self.temporal_column = temporal_column
        self.window = window
        self.threshold = threshold
        self._eps = 1e-8
        # Buffers store historical values for each trade-feature combination
        self.buffers: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.window))
        )

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Warm-up rolling buffers with training data, sorted by temporal column.
        
        Args:
            train_df (pd.DataFrame): Training DataFrame containing identity_column, temporal_column, 
                                    and feature columns. Data is sorted by temporal_column before 
                                    processing to maintain temporal order.
        
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
        Compute Hampel filter scores for each row, using the historical window of buffered values.
        
        The Hampel filter calculates robust Z-scores based on median and MAD of the rolling window.
        Robust Z-score = (value - median) / (1.4826 * MAD)
        The constant 1.4826 makes MAD consistent with standard deviation for normal distributions.
        
        Returns the maximum absolute robust Z-score across features, clipped to [0,1] range.
        Requires at least 5 values in buffer to compute statistics.
        
        Args:
            day_df (pd.DataFrame): DataFrame containing identity_column and feature columns to score.
        
        Returns:
            pd.Series: Series of normalized Hampel filter scores (values in [0,1]) indexed by day_df's index.
                      Returns 0.0 for unseen trades or insufficient buffer.
        
        Notes:
            - Call update_state() after scoring to add today's values to buffers (prevents look-ahead bias).
            - The score represents the maximum deviation across all features, normalized by the threshold.
        """
        vals = []
        for idx, row in day_df.iterrows():
            t = row[self.identity_column]
            if t not in self.buffers:
                vals.append(0.0)  # Default score for unseen trades
            else:
                robust_z_max = 0.0
                for f in self.features:
                    buf = self.buffers[t][f]
                    if len(buf) >= 5:
                        arr = np.asarray(buf, dtype=float)
                        median = np.nanmedian(arr)
                        # Compute MAD (Median Absolute Deviation)
                        mad = np.nanmedian(np.abs(arr - median))
                        
                        v = row[f]
                        # Compute robust Z-score using Hampel method
                        # The constant 1.4826 makes MAD consistent with std for normal distributions
                        if np.isfinite(mad) and mad > 0 and pd.notna(v):                            
                            robust_z = abs(float(row[f]) - median) / (1.4826 * mad + self._eps)
                        else:
                            robust_z = 0.0
                        
                        robust_z_max = max(robust_z_max, robust_z)
                
                # Normalize to [0,1] using the threshold
                vals.append(float(np.clip(robust_z_max / self.threshold, 0.0, 1.0)))
        
        return pd.Series(vals, index=day_df.index, name=self.ScoreName)

    def _update_state_impl(self, day_df: pd.DataFrame) -> None:
        """
        Update rolling buffers by appending today's feature values after scoring.
        
        This should be called AFTER score_day() to prevent look-ahead bias. Buffers automatically
        maintain their fixed size (window) by dropping oldest values when full.
        
        Args:
            day_df (pd.DataFrame): DataFrame containing identity_column and feature columns.
        
        Returns:
            None: Updates the rolling buffers with new values from day_df.
        """
        for _, row in day_df.iterrows():
            t = row[self.identity_column]
            for f in self.features:
                v = float(row[f]) if pd.notna(row[f]) else np.nan
                if np.isfinite(v):
                    self.buffers[t][f].append(v)
