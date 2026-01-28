import numpy as np
import pandas as pd
from collections import deque
from typing import List, Optional

from QC_methods.qc_base import StatefulQCMethod

class ECDFQC(StatefulQCMethod):
    """
    Empirical Cumulative Distribution Function based QC method.

    For each feature:
      - builds empirical distribution from historical values
      - scores new observations by tail probability
    """

    def __init__(
        self,
        features: List[str],
        score_name: str,
        window: Optional[int] = None,  # None = expanding window
        min_samples: int = 20,
        default_score: float = 1.0 # neutral score to be used if score cannot be computed (due to insufficient data/history)
    ):
        super().__init__(score_name)
        self._features = features
        self._window = window
        self._min_samples = min_samples
        self._default_score = default_score

        # per-feature historical buffers
        self._history = {
            f: deque(maxlen=window) if window is not None else []
            for f in features
        }

    # ---------- FIT ----------

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Initialize ECDF state using TRAIN window only.
        """
        for f in self._features:
            values = train_df[f].dropna().values.tolist()
            if self._window is not None:
                self._history[f].extend(values[-self._window:])
            else:
                self._history[f].extend(values)

    # ---------- SCORING ----------

    def _score_day_impl(self, day_df: pd.DataFrame) -> pd.Series:
        """
        Compute ECDF-based score per row.
        """
        scores = []

        for _, row in day_df.iterrows():
            feature_scores = []

            for f in self._features:
                x = row[f]
                hist = np.asarray(self._history[f])

                if np.isnan(x) or len(hist) < self._min_samples:
                    # feature_scores.append(np.nan)
                    continue

                # ECDF
                F_x = np.mean(hist <= x)

                # two-sided tail probability
                score = 2.0 * min(F_x, 1.0 - F_x)
                feature_scores.append(score)
            
            if len(feature_scores) == 0:
                scores.append(self._default_score)  # neutral score
            else:
                scores.append(float(np.mean(feature_scores)))

        return pd.Series(scores, index=day_df.index, name=self.ScoreName)

    # ---------- STATE UPDATE ----------

    def _update_state_impl(self, day_df: pd.DataFrame) -> None:
        """
        Update ECDF history AFTER scoring (no leakage).
        """
        for f in self._features:
            values = day_df[f].dropna().values.tolist()
            if self._window is not None:
                self._history[f].extend(values[-self._window:])
            else:
                self._history[f].extend(values)
