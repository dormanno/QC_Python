from typing import List, Optional
import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

from column_names import main_column
from QC_methods.qc_base import StatefulQCMethod

logger = logging.getLogger(__name__)

# Absolute minimum neighbors for LOF to produce meaningful density estimates.
# LOF with k=2 can still detect points in sparse vs dense regions.
_MIN_NEIGHBORS = 2

# Minimum observations required for LOF fitting after filtering sparse days.
_MIN_OBSERVATIONS = 20


class LOFQC(StatefulQCMethod):
    """
    Local Outlier Factor (LOF) based QC method.
    
    LOF detects outliers by measuring the local deviation of density of a given sample
    with respect to its neighbors. Samples with substantially lower density than their
    neighbors are considered outliers.
    
    Returns anomaly scores in [0,1] where higher values indicate more anomalous samples.
    """

    def __init__(self,
                 *,
                 features: List[str],
                 identity_column: str,
                 score_name: str,
                 n_neighbors: int = 20,
                 max_window_size: int = 100,
                 contamination: float = 0.1,
                 use_robust_scaler: bool = True):
        """
        Initialize LOFQC with specified features and LOF parameters.
        
        Args:
            features (List[str]): List of feature column names to use for LOF calculation.
            identity_column (str): Column name for trade/entity identifier.
            score_name (str): Name for the output score column.
            n_neighbors (int, optional): Number of neighbors to use for LOF computation. Defaults to 20.
            contamination (float, optional): Expected proportion of outliers in the dataset. Defaults to 0.1.
            use_robust_scaler (bool, optional): Whether to apply RobustScaler preprocessing. Defaults to True.
        """
        super().__init__(score_name=score_name)
        self.features = features
        self.identity_column = identity_column
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.use_robust_scaler = use_robust_scaler
        self._max_window_size = max_window_size
        
        self._lof: Optional[LocalOutlierFactor] = None
        self._scaler: Optional[RobustScaler] = None
        self._train_X: Optional[np.ndarray] = None
        self._buffer_df: Optional[pd.DataFrame] = None        

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Fit the LOF model on training data.
        
        Builds feature matrix from training window, optionally applies RobustScaler,
        and fits the LocalOutlierFactor model for novelty detection.
        Sets self.isFit = True on success, False on failure.
        
        Args:
            train_df (pd.DataFrame): Training DataFrame containing identity_column and feature columns.
        
        Returns:
            None: Stores fitted scaler and LOF model as instance variables.
        """
        try:
            # 1) Filter out days with insufficient observations
            row_valid = ~train_df[self.features].isna().all(axis=1)
            if main_column.DATE in train_df.columns:
                day_counts = train_df.loc[row_valid].groupby(main_column.DATE).size()
                valid_dates = day_counts[day_counts >= _MIN_OBSERVATIONS].index
                filtered_df = train_df.loc[row_valid & train_df[main_column.DATE].isin(valid_dates)].copy()
                excluded_days = train_df[main_column.DATE].nunique() - len(valid_dates)
                if excluded_days > 0:
                    logger.info(
                        f"LOF excluded {excluded_days} day(s) with < {_MIN_OBSERVATIONS} observations"
                    )
            else:
                filtered_df = train_df.loc[row_valid].copy()

            # 2) Extract features
            X_raw = filtered_df[self.features].astype(float).values

            # 3) Select rows with at least one observed feature
            # Feature is usable if it has at least one non-NaN value
            feature_valid_mask = ~np.isnan(X_raw).all(axis=0)
            X_raw = X_raw[:, feature_valid_mask]
            # this mask will be used during scoring to select same features
            self._active_features_mask = feature_valid_mask

            # Defensive guard: ensure enough samples remain after filtering
            min_required = max(_MIN_OBSERVATIONS, _MIN_NEIGHBORS + 1)
            if X_raw.shape[0] < min_required:
                unique_ids = filtered_df[self.identity_column].nunique() if self.identity_column in filtered_df.columns else "N/A"
                raise ValueError(
                    f"Insufficient non-missing observations to fit LOF after filtering sparse days. "
                    f"Required: {min_required} (minimum {_MIN_OBSERVATIONS} observations), "
                    f"Got: {X_raw.shape[0]} rows. "
                    f"Dataset info: {len(filtered_df)} rows after filtering (from {len(train_df)} total), "
                    f"{unique_ids} unique {self.identity_column}, "
                    f"{X_raw.shape[1]} active features (after filtering all-NaN columns). "
                    f"Features: {[f for f, valid in zip(self.features, feature_valid_mask) if valid]}"
                )

            # Compute effective n_neighbors: cap at n_samples - 1, floor at _MIN_NEIGHBORS
            effective_n_neighbors = max(_MIN_NEIGHBORS, min(self.n_neighbors, X_raw.shape[0] - 1))
            if effective_n_neighbors < self.n_neighbors:
                logger.info(
                    f"LOF n_neighbors reduced from {self.n_neighbors} to {effective_n_neighbors} "
                    f"(dataset has {X_raw.shape[0]} rows)"
                )
            
            # 3. Compute feature-wise medians on training data        
            self._feature_medians = np.nanmedian(X_raw, axis=0)        

            # 4. Impute missing values using training medians
            X_fit_clean = np.where(
                np.isnan(X_raw),
                self._feature_medians,
                X_raw
            )        

            # 5. Apply robust scaling (after imputation)
            if self.use_robust_scaler:
                self._scaler = RobustScaler(with_centering=True, with_scaling=True)
                X_fit_clean = self._scaler.fit_transform(X_fit_clean)
            else:
                self._scaler = None    
            # 6. Fit LOF model in novelty mode
            self._lof = LocalOutlierFactor(
                n_neighbors=effective_n_neighbors,
                contamination=self.contamination,
                novelty=True,
                n_jobs=-1
            )

            self._lof.fit(X_fit_clean)

            # 7. Store fitted reference data (optional, for diagnostics)
            self._train_X = X_fit_clean
            
            # Mark as successfully fitted
            self.isFit = True
            logger.info(f"LOF fit successfully with {X_fit_clean.shape[0]} observations and {effective_n_neighbors} neighbors")
        except Exception as e:
            self.isFit = False
            logger.warning(f"LOF fit failed: {e}")
    
    def _update_state_impl(self, day_df):
        """
        Update stateful LOF model with new day's data.
        
        Maintains a rolling buffer of recent data up to max_window_size,
        refits the LOF model on the updated buffer.
        
        Args:
            day_df (pd.DataFrame): DataFrame for the new day containing identity_column and feature columns.
        
        Returns:
            None: Updates internal buffer and refits LOF model.
        """
        if self._buffer_df is None:
            self._buffer_df = day_df.copy()
        else:
            self._buffer_df = pd.concat([self._buffer_df, day_df], axis=0, ignore_index=True)
        
        # Keep only the most recent max_window_size rows
        if len(self._buffer_df) > self._max_window_size:
            self._buffer_df = self._buffer_df.iloc[-self._max_window_size :]#.reset_index(drop=True)
        
        # Refit LOF on updated buffer
        self.fit(self._buffer_df)

    def _score_day_impl(self, day_df: pd.DataFrame) -> pd.Series:
        """
        Compute LOF-based outlier scores for each row.
        
        Transforms features using the fitted scaler and computes LOF scores.
        Scores are normalized to [0,1] range where higher values indicate more anomalous samples.
        
        Args:
            day_df (pd.DataFrame): DataFrame containing identity_column and feature columns to score.
        
        Returns:
            pd.Series: Series of outlier scores (values in [0,1]) indexed by day_df's index,
                      with the specified score name. Higher scores indicate more anomalous samples.
                      Returns 0.0 for rows with missing features.
        
        Raises:
            AssertionError: If fit() has not been called (LOF model is None).
        """
        assert self._lof is not None, "Must call fit() before score_day()"        
        
        # Extract features
        X_raw = day_df[self.features].astype(float).values        
        X_raw = X_raw[:, self._active_features_mask]

        all_nan_mask = np.isnan(X_raw).all(axis=1)

        # 1. Impute FIRST using training medians (which have 0.0 fallback for all-NaN features)
        X_clean = np.where(
            np.isnan(X_raw),
            self._feature_medians,
            X_raw
        )       

        # 2. Scale AFTER imputation
        if self._scaler is not None:
            X_clean = self._scaler.transform(X_clean)

        # Convert to [0,1]: more negative decision = higher score        
        decision = self._lof.decision_function(X_clean)
        raw_score = -decision

        # If every row is all-NaN, return zeros
        valid_mask = ~all_nan_mask

        finite_mask = valid_mask & np.isfinite(raw_score)

        if not finite_mask.any():
            return pd.Series(0.0, index=day_df.index, name=self.ScoreName)

        p95 = np.percentile(raw_score[finite_mask], 95)
        if not np.isfinite(p95) or p95 <= 0:
            p95 = 1.0
        scores = np.clip(raw_score / p95, 0, 1)
        scores[~np.isfinite(scores)] = 0.0
        scores[all_nan_mask] = 0.0

        return pd.Series(scores, index=day_df.index, name=self.ScoreName)