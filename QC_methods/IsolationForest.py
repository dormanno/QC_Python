from typing import Optional, List, Literal
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

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
                 *,
                 base_feats: List[str],
                 identity_column: str,
                 temporal_column: str,
                 score_name: str,
                 mode: FeatureMode = "time_series",
                 per_trade_normalize: bool = False,
                 use_robust_scaler: bool = True,
                 n_estimators: int = 200,
                 max_samples: int = 256,
                 contamination: Optional[float] = 0.01,
                 random_state: int = 42):
        """
        Initialize IsolationForestQC with model and preprocessing parameters.
        
        Args:
            base_feats (List[str]): List of feature column names to extract from the DataFrame.
            identity_column (str): Column name for trade/entity identifier.
            temporal_column (str): Column name for date/temporal identifier.
            mode (FeatureMode, optional): Feature extraction mode, either "time_series" or "cross_sectional".
                                         Defaults to "time_series".
            per_trade_normalize (bool, optional): Whether to normalize features per trade using median |level|.
                                                 Defaults to False. Not recommended if orchestrator already normalizes.
            use_robust_scaler (bool, optional): Whether to apply RobustScaler preprocessing. Defaults to True.
            n_estimators (int, optional): Number of isolation trees. Defaults to 200.
            max_samples (int, optional): Maximum samples per tree. Defaults to 256.
            contamination (Optional[float], optional): Expected contamination rate for the forest.
                                                       Defaults to 0.01. Can be 'auto'.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        super().__init__(score_name=score_name)
        self.base_feats = base_feats
        self.identity_column = identity_column
        self.temporal_column = temporal_column
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
        Build the feature matrix for Isolation Forest from raw trade data.
        
        Extracts and optionally normalizes features according to mode and preprocessing settings.
        Applies per-trade normalization (if enabled) by dividing by median |level| per trade.
        Optionally applies RobustScaler for standardization.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing TRADE, DATE columns and feature columns
                             matching those in ColumnNames.py.
        
        Returns:
            _IFInput: Dataclass containing:
                - X: Feature matrix (n_samples, n_features) after preprocessing
                - ids: DataFrame with TRADE and DATE for each sample
                - feature_names: List of feature names used
                - X_pipeline: Fitted preprocessing pipeline (e.g., RobustScaler) or None
        """
        work = df.copy()

        if self.per_trade_normalize:            
            def _norm_group(g: pd.DataFrame) -> pd.DataFrame:
                denom = g[self.base_feats].abs().median().replace(0, np.nan)
                g[self.base_feats] = g[self.base_feats].div(denom, axis=1)
                return g
            work = work.groupby(self.identity_column, group_keys=False).apply(_norm_group)

        feat_df = work[[self.identity_column, self.temporal_column] + self.base_feats] \
                    .dropna(how="all", subset=self.base_feats).copy()

        feature_names = self.base_feats
        X_raw = feat_df[feature_names].astype(float).values
        ids = feat_df[[self.identity_column, self.temporal_column]].copy()

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
        Fit the Isolation Forest model on training data without data leakage.
        
        Builds the feature matrix from TRAIN window only, fits the preprocessing pipeline
        (e.g., RobustScaler), and trains the Isolation Forest classifier. Stores feature names
        and fitted pipeline for consistent transformation of out-of-sample data.
        
        Args:
            train_df (pd.DataFrame): Training DataFrame containing TRADE, DATE, and feature columns.
        
        Returns:
            None: Stores fitted pipeline, feature names, and classifier as instance variables.
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
        Score a feature matrix using the fitted Isolation Forest classifier.
        
        Converts decision function values (higher = more normal) to anomaly intensity scores
        ranked in [0,1] range where higher values indicate more anomalous samples.
        
        Args:
            X (np.ndarray): Preprocessed feature matrix of shape (n_samples, n_features).
        
        Returns:
            pd.Series: Series of anomaly intensities in [0,1] with column name IF_SCORE.
                      Higher values indicate more anomalous samples.
        
        Raises:
            AssertionError: If fit() has not been called (classifier is None).
        """
        assert self._clf is not None
        decf = self._clf.decision_function(X)
        ranks = pd.Series(decf).rank(method="average")  # low decf => more anomalous
        intensity = 1.0 - (ranks - 1) / (len(ranks) - 1 + 1e-12)
        return intensity.rename(self.ScoreName)

    def score_day(self, day_df: pd.DataFrame) -> pd.Series:
        """
        Transform and score a single out-of-sample day using the fitted model.
        
        Extracts features from the provided day DataFrame, applies the TRAIN-fitted preprocessing
        pipeline (e.g., RobustScaler), and computes anomaly intensity scores using the fitted
        Isolation Forest classifier.
        
        Args:
            day_df (pd.DataFrame): DataFrame containing TRADE column and feature columns to score.
                                  Rows represent trades for a single day.
        
        Returns:
            pd.Series: Series of anomaly intensity scores (values in [0,1]) indexed by day_df's index,
                      with column name IF_SCORE. Higher values indicate more anomalous trades.
        
        Raises:
            AssertionError: If fit() has not been called (feature_names or classifier are None).
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
