"""QC Engine that encapsulates QC methods and scoring logic."""

import logging
from typing import List, Dict
import pandas as pd

from column_names import main_column, qc_column
from QC_methods.qc_method_definitions import QCMethodDefinition, QCMethodDefinitions
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
                 methods_config: dict[QCMethodDefinition, float],
                 roll_window: int = 20):
        """Initialize QC Engine with features, method configuration, and window size.
        
        Args:
            qc_features (List[str]): Features to use for QC scoring.
            methods_config (dict[QCMethod, float]): Dictionary mapping QCMethod instances to weights.
                Keys are QCMethod instances (e.g., QCMethods.ISOLATION_FOREST).
                Values are weights (floats) for aggregation.
                Only methods included in this dict will be enabled.
            roll_window (int): Window size for rolling methods.
        """
        self.qc_features = qc_features
        self.roll_window = roll_window
        self.methods_config = methods_config
        
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
        
        # Convert methods_config to weights dict (QCMethod -> score_column_name)
        weights = {method.score_name: weight 
                   for method, weight in methods_config.items()}
        
        # Initialize aggregator
        self.aggregator = ScoreAggregator(weights=weights)
        
        logger.info(f"QC Engine initialized with {len(self.qc_methods)} methods: {list(self.qc_methods.keys())}")
    
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
