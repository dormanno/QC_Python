import pandas as pd
from typing import Literal
from column_names import qc_column

FLAG = Literal["GREEN", "AMBER", "RED"]

class ScoreAggregator:
    """
    Linear weighted aggregator; weights sum to 1.
    """

    def __init__(self,
                 weights: dict[str, float],
                 *,
                 amber_lo: float = 0.85,
                 red_lo: float = 0.95
                 ):
        """
        Args:
            weights: Dictionary mapping score column names to weights. 
                     Keys should be qc_column score names (IF_score, RobustZ_score, etc.)
                     Can include any subset of available methods.
            amber_lo: Lower threshold for amber flag
            red_lo: Lower threshold for red flag
        """
        available_keys = {
            qc_column.IF_SCORE,
            qc_column.ROBUST_Z_SCORE,
            qc_column.ROLLING_SCORE,
            qc_column.IQR_SCORE,
            qc_column.LOF_SCORE,
            qc_column.ECDF_SCORE,
            qc_column.HAMPEL_SCORE
        }
        
        # Validate that all provided keys are valid score names
        invalid_keys = set(weights.keys()) - available_keys
        if invalid_keys:
            raise ValueError(f"Invalid score names: {invalid_keys}. Valid options: {available_keys}")
        
        # Validate weights sum to 1
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-9:
            raise ValueError(f"Weights must sum to 1. Current sum: {weight_sum}")
        if not (0.0 <= amber_lo < red_lo <= 1.0):
            raise ValueError("Require 0 ≤ amber_lo < red_lo ≤ 1.")
        
        self.weights = weights
        self.amber_lo, self.red_lo = amber_lo, red_lo

    def combine(self, df: pd.DataFrame) -> pd.Series:
        """Combine scores from configured methods using weighted average.
        
        Args:
            df: DataFrame containing score columns for configured methods
            
        Returns:
            Series with aggregated scores
        """
        # Start with zeros
        result = pd.Series(0.0, index=df.index)
        
        # Add weighted contribution from each configured method
        for score_column, weight in self.weights.items():
            if score_column in df.columns:
                result += weight * df[score_column]
            else:
                raise ValueError(f"Expected score column '{score_column}' not found in DataFrame")
        
        return result.rename(qc_column.AGGREGATED_SCORE)

    def map_to_flag(self, agg: pd.Series) -> pd.Series:
        """
        Map aggregated anomaly intensity in [0,1] to traffic-light flags:
          [0, amber_lo)  -> GREEN
          [amber_lo, red_lo) -> AMBER
          [red_lo, 1]   -> RED
        Returns a pandas Categorical with ordered categories (GREEN < AMBER < RED).
        """
        cats = pd.cut(
            agg.astype(float),
            bins=[-1e-12, self.amber_lo, self.red_lo, 1.0 + 1e-12],
            labels=["GREEN", "AMBER", "RED"],
            right=False, include_lowest=True
        ).astype("category")
        cats = cats.cat.set_categories(["GREEN", "AMBER", "RED"], ordered=True)
        return cats.rename(qc_column.QC_FLAG)
