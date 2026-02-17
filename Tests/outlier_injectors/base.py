"""
Base class for outlier injection in testing QC methods.
Provides shared functionality for statistical computation and trade/date selection.

Notation:
- x: original feature value for a given TradeID-Date row
- MAD_seg(x): robust scale computed on Train for that TradeType × Family (per-feature)
- IQR_seg(x): IQR-based scale computed on Train for that TradeType (per-feature)
- k: severity parameter (3=Small, 6=Medium, 12=High, 24=Extreme)
- U(a,b): uniform random draw between a and b
- ε: small noise term (e.g., Normal(0, 0.1*MAD))
- s: sign, randomly ±1
- α: relative coefficient for Index trade types
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List
from abc import ABC, abstractmethod
from column_names import main_column


class OutlierInjector(ABC):
    """
    Abstract base class for injecting synthetic outliers into datasets for QC testing.
    
    Implements shared utility methods for trade/date selection, statistic computation,
    and other common functionality. Subclasses implement dataset-specific injection scenarios.
    """
    
    # Severity levels (multipliers for MAD/IQR)
    SEVERITY_SMALL = 3
    SEVERITY_MEDIUM = 6
    SEVERITY_HIGH = 12
    SEVERITY_EXTREME = 24
    
    def __init__(self, random_seed: Optional[int] = 42):
        """
        Initialize the injector.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Will store MAD/IQR statistics computed from train data
        self._mad_stats: Dict[Tuple[str, str], float] = {}
        self._iqr_stats: Dict[Tuple[str, str], float] = {}
    
    @abstractmethod
    def inject(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject outliers into the dataset.
        
        This method must be implemented by subclasses to apply
        dataset-specific injection scenarios.
        
        Args:
            dataset: The dataset to inject outliers into
            
        Returns:
            The modified dataset with outliers injected
        """
        pass

    def _ensure_record_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure RecordType column exists; default missing values to "OOS".
        """
        if main_column.RECORD_TYPE not in df.columns:
            df[main_column.RECORD_TYPE] = "OOS"
        else:
            df[main_column.RECORD_TYPE] = df[main_column.RECORD_TYPE].fillna("OOS")
        return df

    def _get_train_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return Train rows if present, otherwise fall back to full dataset.
        """
        if main_column.RECORD_TYPE in df.columns:
            train_data = df[df[main_column.RECORD_TYPE] == "Train"]
            if not train_data.empty:
                return train_data
        return df

    def _eligible_mask(self, df: pd.DataFrame) -> pd.Series:
        """
        Eligible rows are OOS (RecordType != "Train") and not already injected.
        """
        if main_column.RECORD_TYPE not in df.columns:
            return pd.Series([True] * len(df), index=df.index)

        record_series = df[main_column.RECORD_TYPE].astype(str).fillna("")
        return (record_series != "Train") & (~record_series.str.startswith("Injected_")) & (~record_series.str.startswith("CD_"))
    
    def _compute_mad_stats(self, train_data: pd.DataFrame, features: List[str]):
        """
        Compute MAD (Median Absolute Deviation) for each feature × TradeType.
        
        Args:
            train_data: Training data (RecordType == "Train")
            features: List of feature column names to compute MAD for
        """
        self._mad_stats.clear()
        
        for trade_type in train_data[main_column.TRADE_TYPE].unique():
            type_data = train_data[train_data[main_column.TRADE_TYPE] == trade_type]
            
            for feature in features:
                if feature in type_data.columns:
                    values = type_data[feature].dropna()
                    if len(values) > 0:
                        median = values.median()
                        mad = np.median(np.abs(values - median))
                        # Use 1.4826 to make MAD consistent with std for normal dist
                        mad_scaled = mad * 1.4826
                        self._mad_stats[(trade_type, feature)] = mad_scaled if mad_scaled > 0 else 1.0
    
    def _compute_iqr_stats(self, train_data: pd.DataFrame, features: List[str]):
        """
        Compute IQR (Interquartile Range) for each feature × TradeType.
        
        Args:
            train_data: Training data (RecordType == "Train")
            features: List of feature column names to compute IQR for
        """
        self._iqr_stats.clear()
        
        for trade_type in train_data[main_column.TRADE_TYPE].unique():
            type_data = train_data[train_data[main_column.TRADE_TYPE] == trade_type]
            
            for feature in features:
                if feature in type_data.columns:
                    values = type_data[feature].dropna()
                    if len(values) > 0:
                        q75, q25 = np.percentile(values, [75, 25])
                        iqr = q75 - q25
                        self._iqr_stats[(trade_type, feature)] = iqr if iqr > 0 else 1.0
    
    def _get_mad(self, trade_type: str, feature: str) -> float:
        """Get MAD for a specific TradeType × feature."""
        return self._mad_stats.get((trade_type, feature), 1.0)
    
    def _get_iqr(self, trade_type: str, feature: str) -> float:
        """Get IQR for a specific TradeType × feature."""
        return self._iqr_stats.get((trade_type, feature), 1.0)
    
    def _random_sign(self) -> int:
        """Return random sign: +1 or -1."""
        return np.random.choice([-1, 1])
    
    def _select_random_trades(self, dataset: pd.DataFrame, trade_count: int) -> List[str]:
        """
        Select random trades by exact count.
        
        Args:
            dataset: Dataset to select from
            trade_count: Exact number of trades to select
            
        Returns:
            List of selected TradeIDs
        """
        eligible_data = dataset[self._eligible_mask(dataset)]
        all_trades = eligible_data[main_column.TRADE].unique()
        if len(all_trades) == 0:
            return []
        n_trades = min(max(1, trade_count), len(all_trades))
        return np.random.choice(all_trades, size=n_trades, replace=False).tolist()
    
    def _select_random_dates(self, dataset: pd.DataFrame, n_days_min: int, n_days_max: int) -> List:
        """
        Select random dates from OOS data.
        
        Args:
            dataset: Dataset to select from
            n_days_min: Minimum number of days
            n_days_max: Maximum number of days
            
        Returns:
            List of selected dates
        """
        eligible_data = dataset[self._eligible_mask(dataset)]
        all_dates = sorted(eligible_data[main_column.DATE].unique())
        if len(all_dates) == 0:
            return []
        n_days = np.random.randint(n_days_min, n_days_max + 1)
        n_days = min(n_days, len(all_dates))
        return np.random.choice(all_dates, size=n_days, replace=False).tolist()
