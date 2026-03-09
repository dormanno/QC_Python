"""Family-level score aggregation strategies."""
from enum import Enum
from typing import List, Tuple

import pandas as pd


class FamilyAggregationMode(str, Enum):
    """Mode for aggregating scores across feature families."""
    NOISY_OR = "noisy_or"
    MAX = "max"


class FamilyAggregator:
    """Aggregates family-level scores into a single overall score.
    
    Supports two aggregation modes:
    - NOISY_OR: Weighted noisy-OR combination: 1 - prod((1 - score_i)^weight_i)
    - MAX: Weighted maximum: max(weight_i * score_i) element-wise
    """
    
    def __init__(self, mode: FamilyAggregationMode | str = FamilyAggregationMode.NOISY_OR):
        """Initialize family aggregator.
        
        Args:
            mode: Aggregation mode (NOISY_OR or MAX)
        """
        if isinstance(mode, str):
            mode = FamilyAggregationMode(mode)
        self.mode = mode
    
    def aggregate(self, family_scores: List[Tuple[float, pd.Series]]) -> pd.Series:
        """Aggregate family scores into overall score.
        
        Args:
            family_scores: List of (weight, aggregated_score_series) tuples,
                one per feature family.
        
        Returns:
            Series with overall aggregated scores in [0, 1].
        """
        if not family_scores:
            raise ValueError("family_scores list cannot be empty")
        
        # Get index from first family's scores
        index = family_scores[0][1].index
        
        if self.mode == FamilyAggregationMode.NOISY_OR:
            return self._aggregate_noisy_or(family_scores, index)
        elif self.mode == FamilyAggregationMode.MAX:
            return self._aggregate_max(family_scores, index)
        else:
            raise ValueError(f"Unknown aggregation mode: {self.mode}")
    
    def _aggregate_noisy_or(self, family_scores: List[Tuple[float, pd.Series]], index) -> pd.Series:
        """Weighted noisy-OR aggregation: 1 - prod((1 - score_i)^weight_i)"""
        survival = pd.Series(1.0, index=index)
        for weight, agg in family_scores:
            survival *= (1 - agg).clip(lower=0.0) ** weight
        overall = (1 - survival).clip(lower=0.0, upper=1.0)
        return overall
    
    def _aggregate_max(self, family_scores: List[Tuple[float, pd.Series]], index) -> pd.Series:
        """Weighted maximum aggregation: max(weight_i * score_i) element-wise."""
        # Apply weights to each family score
        weighted_scores = pd.DataFrame({
            i: weight * agg for i, (weight, agg) in enumerate(family_scores)
        })
        # Take element-wise maximum across families
        overall = weighted_scores.max(axis=1).clip(lower=0.0, upper=1.0)
        return overall
