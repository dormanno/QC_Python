"""QC Engine that encapsulates QC methods and scoring logic."""

import copy
import logging
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

from column_names import main_column, qc_column
from QC_methods.qc_method_definitions import QCMethodDefinition, QCMethodDefinitions
from QC_methods import IsolationForestQC, RobustZScoreQC, IQRQC, RollingZScoreQC, LOFQC, ECDFQC
from QC_methods.hampel import HampelFilterQC
from QC_methods.stale_value_filter import StaleValueFilterQC
from QC_methods.qc_base import StatefulQCMethod
from Engine.aggregator import ConsensusMode, ScoreAggregator
from Engine.score_normalizer import ScoreNormalizer

logger = logging.getLogger(__name__)


class QCEngine:
    """Encapsulates QC methods and aggregation logic.
    
    Responsible for:
    - Instantiating QC methods with appropriate configurations
    - Managing the aggregator with weights
    - Fitting methods on training data
    - Scoring data with all methods and aggregating results
    - Updating stateful methods
    """
    
    def __init__(self,
                 qc_features: List[str],
                 methods_config: dict[QCMethodDefinition, float],
                 roll_window: int = 20,
                 score_normalizer: Optional[ScoreNormalizer] = None,
                 consensus: ConsensusMode | str = ConsensusMode.NONE,
                 filters: Optional[List[QCMethodDefinition]] = None):
        """Initialize QC Engine with features, method configuration, and window size.
        
        Args:
            qc_features (List[str]): Features to use for QC scoring.
            methods_config (dict[QCMethod, float]): Dictionary mapping QCMethod instances to weights.
                Keys are QCMethod instances (e.g., QCMethods.ISOLATION_FOREST).
                Values are weights (floats) for aggregation.
                Only methods included in this dict will be enabled.
            roll_window (int): Window size for rolling methods.
            score_normalizer (ScoreNormalizer): Quantile normalizer instance for score-level normalization.
            consensus (ConsensusMode | str): Consensus mode for aggregation.
            filters (List[QCMethodDefinition]): Optional list of filter methods to apply.
        """
        self.qc_features = qc_features
        self.roll_window = roll_window
        self.methods_config = methods_config
        self.filters = filters if filters is not None else []
        if score_normalizer is None:
            raise ValueError("score_normalizer must be provided when constructing QCEngine.")
        self.score_normalizer = score_normalizer
        
        # Validate that all keys are QCMethod instances
        available_methods = {m.name for m in QCMethodDefinitions.all_methods()}
        for method in methods_config.keys():
            if not isinstance(method, QCMethodDefinition):
                raise TypeError(
                    f"All keys in methods_config must be QCMethod instances. "
                    f"Got: {type(method)}. Use QCMethods.ISOLATION_FOREST, etc."
                )
            if method.name not in available_methods:
                raise ValueError(
                    f"Unknown method: {method.name}. "
                    f"Valid options: {list(available_methods)}"
                )
        
        # Initialize QC methods
        self.qc_methods = self._instantiate_qc_methods()
        
        # Initialize filter methods
        self.qc_filters = self._instantiate_filters()

        # Skip normalization for methods already producing quantiles (e.g., ECDF)
        self._skip_normalization_cols = {
            method.ScoreName for method in self.qc_methods.values()
            if getattr(method, "RequiresNormalization", True) is False
        }
        
        # Convert methods_config to weights dict (QCMethod -> score_column_name)
        weights = {method.score_name: weight 
                   for method, weight in methods_config.items()}
        
        # Initialize aggregator
        self.aggregator = ScoreAggregator(weights=weights, consensus=consensus)
        
        logger.info(f"QC Engine initialized with {len(self.qc_methods)} methods: {list(self.qc_methods.keys())}")
        logger.info(f"QC Engine initialized with {len(self.qc_filters)} filters: {list(self.qc_filters.keys())}")
    
    def _instantiate_qc_methods(self) -> Dict:
        """Instantiate QC scoring methods based on methods_config.
        
        Returns:
            Dict: Dictionary of QC method instances keyed by method name.
        """
        # Define all available methods with their QCMethod keys
        all_methods = {
            QCMethodDefinitions.ISOLATION_FOREST.name: IsolationForestQC(
                base_feats=self.qc_features,
                identity_column=main_column.TRADE,
                temporal_column=main_column.DATE,
                score_name=qc_column.IF_SCORE,
                mode="time_series",
                per_trade_normalize=False,
                use_robust_scaler=True,
                n_estimators=200,
                max_samples=256,
                contamination=0.01,
            ),
            QCMethodDefinitions.ROBUST_Z.name: RobustZScoreQC(
                features=self.qc_features,
                identity_column=main_column.TRADE,
                score_name=qc_column.ROBUST_Z_SCORE,
            ),
            QCMethodDefinitions.IQR.name: IQRQC(
                features=self.qc_features,
                identity_column=main_column.TRADE,
                score_name=qc_column.IQR_SCORE,
            ),
            QCMethodDefinitions.ROLLING.name: RollingZScoreQC(
                features=self.qc_features,
                identity_column=main_column.TRADE,
                temporal_column=main_column.DATE,
                score_name=qc_column.ROLLING_SCORE,
                window=self.roll_window
            ),
            QCMethodDefinitions.LOF.name: LOFQC(
                features=self.qc_features,
                identity_column=main_column.TRADE,
                score_name=qc_column.LOF_SCORE,
                n_neighbors=self.roll_window,
                contamination=0.1,
                use_robust_scaler=True
            ),
            QCMethodDefinitions.ECDF.name: ECDFQC(
                features=self.qc_features,
                score_name=qc_column.ECDF_SCORE,
            ),
            QCMethodDefinitions.HAMPEL.name: HampelFilterQC(
                features=self.qc_features,
                identity_column=main_column.TRADE,
                temporal_column=main_column.DATE,
                score_name=qc_column.HAMPEL_SCORE,
                window=self.roll_window,
                threshold=3.0
            )
        }
        
        # Only instantiate methods specified in methods_config
        enabled_method_names = {method.name for method in self.methods_config.keys()}
        methods = {name: impl for name, impl in all_methods.items() if name in enabled_method_names}
        
        logger.info(f"Instantiated {len(methods)} QC methods: {list(methods.keys())}")
        return methods
    
    def _instantiate_filters(self) -> Dict:
        """Instantiate QC filter methods based on filters list.
        
        Returns:
            Dict: Dictionary of filter method instances keyed by method name.
        """
        # Define all available filter methods
        all_filters = {
            QCMethodDefinitions.STALE_VALUE.name: StaleValueFilterQC(
                features=self.qc_features,
                identity_column=main_column.TRADE,
                temporal_column=main_column.DATE,
                score_name=qc_column.STALE_SCORE
            )
        }
        
        # Only instantiate filters specified in self.filters
        enabled_filter_names = {filter_def.name for filter_def in self.filters}
        filters = {name: impl for name, impl in all_filters.items() if name in enabled_filter_names}
        
        logger.info(f"Instantiated {len(filters)} filter methods: {list(filters.keys())}")
        return filters
    
    def get_score_columns(self) -> List[str]:
        """Get list of all score columns that will be generated by this engine.
        
        Returns:
            List[str]: List of score column names including individual method scores,
                      aggregated score, and QC flag.
        """
        # Get score names from all configured methods
        method_score_cols = [method.ScoreName for method in self.qc_methods.values()]
        
        # Add aggregated score and flag columns
        all_score_cols = method_score_cols + [qc_column.AGGREGATED_SCORE, qc_column.QC_FLAG]
        
        return all_score_cols
    
    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit all QC methods and filters on training data.
        
        Args:
            train_data (pd.DataFrame): Training data to fit methods on.
        """
        for name, method in self.qc_methods.items():
            method.fit(train_data)
            logger.info(f"Fitted {name} on training data")
        
        for name, filter_method in self.qc_filters.items():
            filter_method.fit(train_data)
            logger.info(f"Fitted filter {name} on training data")

        # Fit score normalizer on TRAIN scores (no look-ahead in scoring order)
        self._fit_score_normalizer(train_data)
    
    def score_day(self, day_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Score a single day's data with all QC methods and filters, then aggregate.
        
        Args:
            day_data (pd.DataFrame): Data for one day (already normalized). Must contain values for all features for all trades for a given day.
        
        Returns:
            Tuple containing:
                - day_scores (pd.DataFrame): Individual method scores
                - aggregated_score (pd.Series): Weighted aggregated score (or 1 if any filter returns 1)
                - qc_flag (pd.DataFrame): Traffic-light flags (GREEN/AMBER/RED)
        """
        # Score with each method dynamically (raw)
        day_scores_raw = self._score_day_raw(day_data, self.qc_methods)

        # Normalize scores (skip ECDF)
        day_scores = self._normalize_scores(day_scores_raw)
        
        # Apply filters: if any filter returns 1, set aggregated_score to 1
        aggregated_score = self.aggregator.combine(day_scores)
        
        if self.qc_filters:
            filter_scores = self._score_day_raw(day_data, self.qc_filters)
            # For each filter column, if it returns 1, override aggregated_score to 1
            for col in filter_scores.columns:
                filter_flagged = filter_scores[col] == 1.0
                aggregated_score[filter_flagged] = 1.0
        
        qc_flag = self.aggregator.map_to_flag(aggregated_score, simpleMode=True).to_frame()
        
        return day_scores, aggregated_score, qc_flag
    
    def update_stateful_methods(self, day_data: pd.DataFrame) -> None:
        """Update state for all stateful QC methods and filters.
        
        Should be called AFTER scoring to prevent look-ahead bias.
        
        Args:
            day_data (pd.DataFrame): Data to incorporate into stateful methods.
        """
        for method in self.qc_methods.values():
            if isinstance(method, StatefulQCMethod):
                method.update_state(day_data)
        
        for filter_method in self.qc_filters.values():
            if isinstance(filter_method, StatefulQCMethod):
                filter_method.update_state(day_data)

    def _score_day_raw(self, day_data: pd.DataFrame, methods: Dict) -> pd.DataFrame:
        """Score a single day and return raw method scores."""
        scores = {}
        for method in methods.values():
            scores[method.ScoreName] = method.score_day(day_data)
        day_scores = pd.concat(scores.values(), axis=1)
        day_scores.columns = scores.keys()
        return day_scores

    def _update_stateful_methods_on(self, day_data: pd.DataFrame, methods: Dict) -> None:
        """Update state for all stateful methods in the provided collection."""
        for method in methods.values():
            if isinstance(method, StatefulQCMethod):
                method.update_state(day_data)

    def _fit_score_normalizer(self, train_data: pd.DataFrame) -> None:
        """Fit quantile normalizer using TRAIN scores in chronological order."""
        train_dates = train_data[main_column.DATE].drop_duplicates().sort_values().to_list()
        if len(train_dates) == 0:
            raise ValueError("Training data must include at least one date for score normalization.")

        # Use copies to avoid mutating method state used for OOS scoring
        methods_for_scoring = copy.deepcopy(self.qc_methods)
        train_scores = []

        for d in train_dates:
            day_df = train_data.loc[train_data[main_column.DATE] == d].copy()
            day_scores = self._score_day_raw(day_df, methods_for_scoring)
            train_scores.append(day_scores)
            self._update_stateful_methods_on(day_df, methods_for_scoring)

        train_scores_df = pd.concat(train_scores, axis=0)
        self.score_normalizer.fit(train_scores_df)

    def _normalize_scores(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize scores using quantile ranks, skipping already-quantile columns."""
        normalized = self.score_normalizer.transform(scores_df)

        for col in self._skip_normalization_cols:
            if col in scores_df.columns:
                normalized[col] = scores_df[col]

        return normalized
