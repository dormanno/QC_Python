"""QC Engine that encapsulates QC methods and scoring logic."""

import logging
from typing import List, Dict
import pandas as pd

from column_names import main_column, qc_column
from QC_methods import IsolationForestQC, RobustZScoreQC, IQRQC, RollingZScoreQC, LOFQC, ECDFQC
from QC_methods.hampel import HampelFilterQC
from QC_methods.qc_base import StatefulQCMethod
from Engine.aggregator import ScoreAggregator

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
                 weights: dict[str, float],
                 roll_window: int = 20):
        """Initialize QC Engine with features, weights, and configuration.
        
        Args:
            qc_features (List[str]): Features to use for QC scoring.
            weights (dict[str, float]): Dictionary mapping score column names to weights.
                Expected keys: qc_column score names (IF_score, RobustZ_score, etc.)
            roll_window (int): Window size for rolling methods.
        """
        self.qc_features = qc_features
        self.roll_window = roll_window
        
        # Initialize aggregator
        self.aggregator = ScoreAggregator(weights=weights)
        
        # Initialize QC methods
        self.qc_methods = self._instantiate_qc_methods()
        logger.info(f"QC Engine initialized with {len(self.qc_methods)} methods")
    
    def _instantiate_qc_methods(self) -> Dict:
        """Instantiate all QC scoring methods.
        
        Returns:
            Dict: Dictionary of QC method instances keyed by method type.
        """
        methods = {
            'isolation_forest': IsolationForestQC(
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
            'robust_z': RobustZScoreQC(
                features=self.qc_features,
                identity_column=main_column.TRADE,
                score_name=qc_column.ROBUST_Z_SCORE,
            ),
            'iqr': IQRQC(
                features=self.qc_features,
                identity_column=main_column.TRADE,
                score_name=qc_column.IQR_SCORE,
            ),
            'rolling': RollingZScoreQC(
                features=self.qc_features,
                identity_column=main_column.TRADE,
                temporal_column=main_column.DATE,
                score_name=qc_column.ROLLING_SCORE,
                window=self.roll_window
            ),
            'lof': LOFQC(
                features=self.qc_features,
                identity_column=main_column.TRADE,
                score_name=qc_column.LOF_SCORE,
                n_neighbors=self.roll_window,
                contamination=0.1,
                use_robust_scaler=True
            ),
            'ecdf': ECDFQC(
                features=self.qc_features,
                score_name=qc_column.ECDF_SCORE,
            ),
            'hampel': HampelFilterQC(
                features=self.qc_features,
                identity_column=main_column.TRADE,
                temporal_column=main_column.DATE,
                score_name=qc_column.HAMPEL_SCORE,
                window=self.roll_window,
                threshold=3.0
            )
        }
        
        logger.info(f"Instantiated {len(methods)} QC methods: {list(methods.keys())}")
        return methods
    
    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit all QC methods on training data.
        
        Args:
            train_data (pd.DataFrame): Training data to fit methods on.
        """
        for name, method in self.qc_methods.items():
            method.fit(train_data)
            logger.info(f"Fitted {name} on training data")
    
    def score_day(self, day_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Score a single day's data with all QC methods and aggregate.
        
        Args:
            day_data (pd.DataFrame): Data for one day (already normalized). Must contain values for all features for all trades for a given day.
        
        Returns:
            Tuple containing:
                - day_scores (pd.DataFrame): Individual method scores
                - aggregated_score (pd.Series): Weighted aggregated score
                - qc_flag (pd.DataFrame): Traffic-light flags (GREEN/AMBER/RED)
        """
        # Score with each method dynamically
        scores = {method.ScoreName: method.score_day(day_data) 
                  for method in self.qc_methods.values()}
        
        # Aggregate scores
        day_scores = pd.concat(scores.values(), axis=1)
        day_scores.columns = scores.keys()
        aggregated_score = self.aggregator.combine(day_scores)
        qc_flag = self.aggregator.map_to_flag(aggregated_score).to_frame()
        
        return day_scores, aggregated_score, qc_flag
    
    def update_stateful_methods(self, day_data: pd.DataFrame) -> None:
        """Update state for all stateful QC methods.
        
        Should be called AFTER scoring to prevent look-ahead bias.
        
        Args:
            day_data (pd.DataFrame): Data to incorporate into stateful methods.
        """
        for method in self.qc_methods.values():
            if isinstance(method, StatefulQCMethod):
                method.update_state(day_data)
