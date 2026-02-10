"""Score normalization utilities for QC method outputs."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


class ScoreNormalizer:
    """
    Score-level quantile normalizer.

    For each score column, stores sorted non-NaN training values and maps new
    values to quantile ranks in [0, 1]. NaNs are preserved. If a column has
    fewer than `min_samples` non-NaN values, a neutral fallback is returned.
    """

    def __init__(self, min_samples: int = 2, fallback_value: float = 0.5) -> None:
        """
        Args:
            min_samples: Minimum number of non-NaN samples required to compute
                ECDF quantiles.
            fallback_value: Value used when a column has insufficient samples.
        """
        self.min_samples = min_samples
        self.fallback_value = fallback_value
        self._train_values: Dict[str, np.ndarray] = {}
        self._fitted = False

    def fit(self, train_scores_df: pd.DataFrame) -> "ScoreNormalizer":
        """
        Fit the normalizer on training scores.

        Args:
            train_scores_df: DataFrame of raw method scores for the training set.

        Returns:
            self
        """
        self._train_values = {}
        for col in train_scores_df.columns:
            values = train_scores_df[col].dropna().astype(float).to_numpy()
            if values.size:
                values = np.sort(values)
            self._train_values[col] = values
        self._fitted = True
        return self

    def transform(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw scores into quantile ranks.

        Args:
            scores_df: DataFrame of raw method scores to normalize.

        Returns:
            DataFrame of normalized scores, preserving index/columns.
        """
        if not self._fitted:
            raise ValueError("ScoreNormalizer must be fitted before calling transform().")

        normalized = pd.DataFrame(index=scores_df.index)

        for col in scores_df.columns:
            values = scores_df[col].astype(float).to_numpy()
            nan_mask = np.isnan(values)

            train_vals = self._train_values.get(col, np.array([], dtype=float))
            if train_vals.size < self.min_samples:
                out = np.full_like(values, self.fallback_value, dtype=float)
            else:
                ranks = np.searchsorted(train_vals, values, side="right")
                out = ranks / float(train_vals.size)
                out = np.clip(out, 0.0, 1.0)

            out[nan_mask] = np.nan
            normalized[col] = out

        return normalized