from typing import Optional, List, Literal
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

import ColumnNames as Column
from QC_methods.QC_Base import QCMethod

FeatureMode = Literal["time_series", "cross_sectional"]

@dataclass
class _IFInput:
    X: np.ndarray                 # (n_samples, n_features)
    ids: pd.DataFrame             # columns: TradeName, Date
    feature_names: list           # list of feature names used
    X_pipeline: Optional[Pipeline]  # fitted preprocessing pipeline (e.g., RobustScaler)

class IsolationForestQC(QCMethod):
    """
    Self-contained Isolation Forest QC.
    Embeds:
      - feature matrix builder (mode, optional per-trade normalization, RobustScaler),
      - fit(),
      - score_day() returning a Series in [0,1] named 'IF_score'.

    Notes:
    - Orchestrator may already apply per-trade denominators. In that case
      pass per_trade_normalize=False (recommended to avoid re-fitting on OOS).
    """

    def __init__(self,
                 mode: FeatureMode = "time_series",
                 per_trade_normalize: bool = False,
                 use_robust_scaler: bool = True,
                 n_estimators: int = 200,
                 max_samples: int = 256,
                 contamination: Optional[float] = 0.01,
                 random_state: int = 42):
        self.mode = mode
        self.per_trade_normalize = per_trade_normalize
        self.use_robust_scaler = use_robust_scaler
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

        # learned state
        self._pipeline: Optional[Pipeline] = None
        self._feature_names: Optional[List[str]] = None
        self._clf: Optional[IsolationForest] = None

    # -----------------------------
    # Internal: build the feature X
    # -----------------------------
    def _build_iforest_matrix(self, df: pd.DataFrame) -> _IFInput:
        """
        Reproduces your original builder:
        - feature set aligned with ColumnNames.py and ReadInput.engineer_features
        - optional per-trade normalization by median |level| within each trade
        - optional RobustScaler in a Pipeline
        """
        base_feats = [
            Column.START,
            *Column.PNL_SLICES,
            Column.TOTAL, Column.EXPLAINED, Column.UNEXPLAINED,
            Column.TOTAL_JUMP, Column.UNEXPLAINED_JUMP
        ]

        work = df.copy()

        if self.per_trade_normalize:
            level_feats = [Column.START, Column.END, *Column.PNL_SLICES,
                           Column.TOTAL, Column.EXPLAINED, Column.UNEXPLAINED]
            def _norm_group(g: pd.DataFrame) -> pd.DataFrame:
                denom = g[level_feats].abs().median().replace(0, np.nan)
                g[level_feats] = g[level_feats].div(denom, axis=1)
                return g
            work = work.groupby(Column.TRADE, group_keys=False).apply(_norm_group)

        feat_df = work[[Column.TRADE, Column.DATE] + base_feats] \
                    .dropna(how="all", subset=base_feats).copy()

        feature_names = base_feats
        X_raw = feat_df[feature_names].astype(float).values
        ids = feat_df[[Column.TRADE, Column.DATE]].copy()

        steps = []
        if self.use_robust_scaler:
            steps.append(("robust", RobustScaler(with_centering=True, with_scaling=True)))
        X_pipeline = Pipeline(steps) if steps else None

        if X_pipeline is not None:
            X = X_pipeline.fit_transform(X_raw)
        else:
            X = X_raw

        return _IFInput(X=X, ids=ids, feature_names=feature_names, X_pipeline=X_pipeline)

    # -----------
    # Public API
    # -----------
    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Fit on TRAIN window only (no leakage).
        Stores feature_names and fitted pipeline for later OOS transforms.
        """
        train_input = self._build_iforest_matrix(train_df)
        self._feature_names = train_input.feature_names
        self._pipeline = train_input.X_pipeline

        clf = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,  # may be 'auto' if you wish
            random_state=self.random_state,
            n_jobs=-1,
        )
        clf.fit(train_input.X)
        self._clf = clf

    def _score_matrix(self, X: np.ndarray) -> pd.Series:
        """
        Mirrors your score logic:
          - decision_function (higher = more normal)
          - rank to 0..1 'anomaly_intensity' (higher = worse)
        Returns a pandas Series in [0,1].
        """
        assert self._clf is not None
        decf = self._clf.decision_function(X)
        ranks = pd.Series(decf).rank(method="average")  # low decf => more anomalous
        intensity = 1.0 - (ranks - 1) / (len(ranks) - 1 + 1e-12)
        return intensity.rename(Column.IF_SCORE)

    def score_day(self, day_df: pd.DataFrame) -> pd.Series:
        """
        Transform and score a single OOS day (rows = trades for that day).
        Uses the TRAIN-fitted pipeline and feature order.
        """
        assert self._feature_names is not None
        X_raw = day_df[self._feature_names].astype(float).values
        if self._pipeline is not None:
            X = self._pipeline.transform(X_raw)
        else:
            X = X_raw

        s = self._score_matrix(X)
        s.index = day_df.index
        return s
