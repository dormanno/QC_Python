from typing import Dict, List
import numpy as np
import pandas as pd

from QC_methods.qc_base import StatefulQCMethod


class StaleValueFilterQC(StatefulQCMethod):
    """
    Stale Value Filter QC Method - detects when all feature values are identical to previous day.
    
    This stateful method tracks feature values for each trade across days and flags cases where
    all non-NaN, non-zero values remain unchanged from the previous day.
    
    Fit(): Initialize with the last available date from training data for each trade.
    score_day(): Compare current day values to previous day values for each trade.
    update_state(): Store current day values for next iteration.
    """

    def __init__(
        self,
        *,
        features: List[str],
        identity_column: str,
        temporal_column: str,
        score_name: str
    ):
        """
        Initialize StaleValueFilterQC.

        Args:
            features (List[str]): List of feature column names to monitor for stale values.
            identity_column (str): Column name for trade/entity identifier.
            temporal_column (str): Column name for date/temporal identifier.
            score_name (str): Name for the output score column.
        """
        super().__init__(score_name=score_name)
        self.features = features
        self.identity_column = identity_column
        self.temporal_column = temporal_column
        
        # Store the last seen values for each trade: {trade_id: {feature: value}}
        self.last_values: Dict[str, Dict[str, float]] = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Initialize state with the last available date in training data.
        
        For each trade, stores the feature values from the last temporal point in training data.
        These will be used as the "previous day" for the first scoring iteration.

        Args:
            train_df (pd.DataFrame): Training DataFrame containing identity_column, temporal_column,
                                    and feature columns.
        """
        # Get the last (most recent) date in the training data
        last_date = train_df[self.temporal_column].max()
        
        # Filter for only the last date
        last_day_df = train_df[train_df[self.temporal_column] == last_date]
        
        # Store values for each trade
        for _, row in last_day_df.iterrows():
            trade_id = row[self.identity_column]
            self.last_values[trade_id] = {}
            
            for feature in self.features:
                value = row[feature]
                # Store the value (NaN, 0, or otherwise)
                self.last_values[trade_id][feature] = value

    def _score_day_impl(self, day_df: pd.DataFrame) -> pd.Series:
        """
        Score each row by comparing feature values to previous day.
        
        A row is scored as 1 (stale) if:
        - For all comparable features (non-NaN AND non-zero on both days): values are equal
        
        A row is scored as 0 (not stale) if:
        - No previous day data exists for the trade, OR
        - No comparable features exist, OR
        - Any comparable feature differs from previous day
        
        Features are excluded from comparison if they are NaN or 0 on either day.

        Args:
            day_df (pd.DataFrame): DataFrame containing rows to score.

        Returns:
            pd.Series: Series of scores (0 or 1) aligned to day_df.index.
        """
        scores = []
        
        for _, row in day_df.iterrows():
            trade_id = row[self.identity_column]
            
            # If no previous data for this trade, consider as not stale
            if trade_id not in self.last_values:
                scores.append(0.0)
                continue
            
            previous_values = self.last_values[trade_id]
            
            # Collect comparable features (non-NaN, non-zero in BOTH current and previous day)
            comparable_features = []
            all_equal = True
            
            for feature in self.features:
                current_val = row[feature]
                previous_val = previous_values.get(feature, np.nan)
                
                # Skip if NaN or zero in either day
                if pd.isna(current_val) or pd.isna(previous_val):
                    continue
                if current_val == 0 or previous_val == 0:
                    continue
                
                # This feature is comparable
                comparable_features.append(feature)
                
                # Check if values are equal
                if current_val != previous_val:
                    all_equal = False
            
            # If no comparable features, consider as not stale
            if len(comparable_features) == 0:
                scores.append(0.0)
            # If all comparable features are equal, score as 1 (stale)
            elif all_equal:
                scores.append(1.0)
            # Otherwise score as 0 (not stale)
            else:
                scores.append(0.0)
        
        return pd.Series(scores, index=day_df.index, name=self.ScoreName)

    def _update_state_impl(self, day_df: pd.DataFrame) -> None:
        """
        Update internal state with current day's values.
        
        Stores feature values for each trade, which will be used as "previous day"
        when scoring the next day.

        Args:
            day_df (pd.DataFrame): DataFrame containing the day's data to incorporate into state.
        """
        for _, row in day_df.iterrows():
            trade_id = row[self.identity_column]
            
            # Initialize or update the trade's stored values
            if trade_id not in self.last_values:
                self.last_values[trade_id] = {}
            
            for feature in self.features:
                value = row[feature]
                self.last_values[trade_id][feature] = value
