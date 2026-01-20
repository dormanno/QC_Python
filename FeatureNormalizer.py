"""Feature normalization utilities for QC pipeline."""

import numpy as np
import pandas as pd
from typing import List
from ColumnNames import main_column


class FeatureNormalizer:
    """Handles per-trade feature normalization using training data statistics.
    
    Computes median absolute denominators per trade from training data and applies
    them to normalize features, improving stability for downstream QC methods.
    """
    
    def __init__(self, features: List[str], identity_column: str = main_column.TRADE):
        """Initialize the normalizer.
        
        Args:
            features (List[str]): Features to normalize.
            identity_column (str): Column identifying unique entities (e.g., TradeID).
        """
        self.features = features
        self.identity_column = identity_column
        self.denominators = None
    
    def fit(self, train_df: pd.DataFrame) -> 'FeatureNormalizer':
        """Compute per-trade median absolute denominators from training data.
        
        Args:
            train_df (pd.DataFrame): Training data.
        
        Returns:
            self: For method chaining.
        """
        self.denominators = (
            train_df.groupby(self.identity_column)[self.features]
            .apply(lambda g: g.abs().median())
            .reset_index()
        )
        self.denominators.columns = [self.identity_column] + [
            f"{c}__den" for c in self.features
        ]
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply denominators to normalize features.
        
        Args:
            df (pd.DataFrame): Data to normalize.
        
        Returns:
            pd.DataFrame: Normalized data.
        
        Raises:
            ValueError: If fit() has not been called first.
        """
        if self.denominators is None:
            raise ValueError("Must call fit() before transform()")
        
        normalized = df.merge(self.denominators, on=self.identity_column, how="left")
        for c in self.features:
            dcol = f"{c}__den"
            normalized[c] = normalized[c] / normalized[dcol].replace(0, np.nan)
        
        return normalized.drop(columns=[f"{c}__den" for c in self.features])
    
    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method to fit and transform in one step.
        
        Args:
            train_df (pd.DataFrame): Training data.
        
        Returns:
            pd.DataFrame: Normalized training data.
        """
        self.fit(train_df)
        return self.transform(train_df)
