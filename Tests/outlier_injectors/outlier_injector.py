"""
Outlier Injectors for testing QC methods.
This module provides base and dataset-specific injector classes for synthetic outlier injection.

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
from column_names import pnl_column, cds_column, main_column


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
        return (record_series != "Train") & (~record_series.str.startswith("Injected_"))
    
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


class PnLOutlierInjector(OutlierInjector):
    """
    PnL-specific outlier injector implementing 10 data quality scenarios.
    
    Implements various data quality issue scenarios based on real-world problems
    in PnL reporting systems.
    """
    
    # Re-export severity constants for convenience in default parameters
    SEVERITY_SMALL = OutlierInjector.SEVERITY_SMALL
    SEVERITY_MEDIUM = OutlierInjector.SEVERITY_MEDIUM
    SEVERITY_HIGH = OutlierInjector.SEVERITY_HIGH
    SEVERITY_EXTREME = OutlierInjector.SEVERITY_EXTREME
    
    def inject(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject outliers into the PnL dataset.
        
        Applies all 10 PnL scenarios sequentially, each respecting eligible OOS rows.
        
        Args:
            dataset: The PnL dataset to inject outliers into
            
        Returns:
            The modified dataset with outliers injected
        """
        df = dataset.copy()
        df = self._ensure_record_type(df)

        # Apply all scenarios sequentially, each respecting eligible OOS rows
        df = self.inject_pv_eod_spike_point(df, trade_count=10)
        df = self.inject_pv_eod_step_change(df, trade_count=4)
        df = self.inject_pv_eod_stale_copy(df, trade_count=4)
        df = self.inject_pv_eod_scale_error(df, trade_count=5)
        df = self.inject_pv_eod_sign_flip(df, trade_count=5)
        df = self.inject_slice_credit_single_spike(df, trade_count=10)
        df = self.inject_slice_credit_single_stale(df, trade_count=4)
        df = self.inject_slice_reallocation_bug(df, trade_count=10)
        df = self.inject_total_pnl_identity_break(df, trade_count=5)
        df = self.inject_cross_family_inconsistency(df, trade_count=5)

        return df
    
    # ========================================================================
    # Scenario 1: PV_EoD_Spike_Point
    # ========================================================================
    def inject_pv_eod_spike_point(self, dataset: pd.DataFrame, severity: float = SEVERITY_MEDIUM,
                                  trade_count: int = None) -> pd.DataFrame:
        """
        Inject single-day spikes in End_PV for 1-3% of trades.
        
        Formula: End_PV' = End_PV + s*k*MAD_seg(End_PV)
        
        Real-world: One-off bad market data snapshot, transient calc glitch.
        
        Args:
            dataset: Input dataset
            severity: Severity multiplier (k)
            
        Returns:
            Dataset with injected spikes
        """
        df = dataset.copy()
        
        # Compute MAD stats if not already done
        if not self._mad_stats:
            train_data = self._get_train_data(df)
            self._compute_mad_stats(train_data, [pnl_column.END])
        
        # Select trades and dates
        selected_trades = self._select_random_trades(df, trade_count)
        eligible_mask = self._eligible_mask(df)
        
        for trade_id in selected_trades:
            # Select one random date for this trade
            trade_dates = self._select_random_dates(
                df[df[main_column.TRADE] == trade_id], 1, 1
            )
            if not trade_dates:
                continue
            
            date = trade_dates[0]
            mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == date)
            mask &= eligible_mask
            
            if mask.sum() > 0:
                trade_type = df.loc[mask, main_column.TRADE_TYPE].iloc[0]
                mad = self._get_mad(trade_type, pnl_column.END)
                sign = self._random_sign()
                
                df.loc[mask, pnl_column.END] += sign * severity * mad
                df.loc[mask, main_column.RECORD_TYPE] = "Injected_PV_Spike"
        
        return df
    
    # ========================================================================
    # Scenario 2: PV_EoD_StepChange
    # ========================================================================
    def inject_pv_eod_step_change(self, dataset: pd.DataFrame, severity: float = SEVERITY_MEDIUM,
                                  trade_count: int = None) -> pd.DataFrame:
        """
        Inject persistent step change in End_PV for 0.5-2% of trades, 10+ days.
        
        Formula: For t≥t₀: End_PV' = End_PV + s*k*MAD_seg(End_PV)
        
        Real-world: Persistent mapping/config/model change affecting subset of trades.
        
        Args:
            dataset: Input dataset
            severity: Severity multiplier (k)
            
        Returns:
            Dataset with injected step changes
        """
        df = dataset.copy()
        
        if not self._mad_stats:
            train_data = self._get_train_data(df)
            self._compute_mad_stats(train_data, [pnl_column.END])
        
        selected_trades = self._select_random_trades(df, trade_count)
        eligible_mask = self._eligible_mask(df)
        
        for trade_id in selected_trades:
            trade_data = df[df[main_column.TRADE] == trade_id]
            eligible_trade_data = trade_data[self._eligible_mask(trade_data)]
            oos_dates = sorted(eligible_trade_data[main_column.DATE].unique())
            
            if len(oos_dates) < 10:
                continue
            
            # Pick a start date and affect all dates from that point
            start_idx = np.random.randint(0, len(oos_dates) - 10)
            affected_dates = oos_dates[start_idx:]
            
            trade_type = trade_data[main_column.TRADE_TYPE].iloc[0]
            mad = self._get_mad(trade_type, pnl_column.END)
            sign = self._random_sign()
            shift = sign * severity * mad
            
            mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE].isin(affected_dates))
            mask &= eligible_mask
            df.loc[mask, pnl_column.END] += shift
            df.loc[mask, main_column.RECORD_TYPE] = "Injected_PV_Step"
        
        return df
    
    # ========================================================================
    # Scenario 3: PV_EoD_StaleCopy
    # ========================================================================
    def inject_pv_eod_stale_copy(self, dataset: pd.DataFrame, trade_count: int = None) -> pd.DataFrame:
        """
        Inject stale/stuck values in End_PV for 1-2% of trades, 5-15 days.
        
        Formula: End_PV(t)' = End_PV(t-1)
        
        Real-world: Stuck feed / valuations not updating but still being published.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with stale values
        """
        df = dataset.copy()
        selected_trades = self._select_random_trades(df, trade_count)
        eligible_mask = self._eligible_mask(df)
        
        for trade_id in selected_trades:
            trade_data = df[df[main_column.TRADE] == trade_id].sort_values(main_column.DATE)
            oos_data = trade_data[self._eligible_mask(trade_data)]
            
            if len(oos_data) < 5:
                continue
            
            # Select 5-15 consecutive days
            n_days = np.random.randint(5, min(16, len(oos_data) + 1))
            start_idx = np.random.randint(0, len(oos_data) - n_days + 1)
            
            affected_indices = oos_data.iloc[start_idx:start_idx + n_days].index
            
            # Copy previous day's value for each affected day
            for i, idx in enumerate(affected_indices):
                if i == 0:
                    # Get the value from the day before the stale period
                    prev_idx = oos_data.index[start_idx - 1] if start_idx > 0 else idx
                    stale_value = df.loc[prev_idx, pnl_column.END]
                
                if eligible_mask.loc[idx]:
                    df.loc[idx, pnl_column.END] = stale_value
                    df.loc[idx, main_column.RECORD_TYPE] = "Injected_PV_Stale"
        
        return df
    
    # ========================================================================
    # Scenario 4: PV_EoD_ScaleError_x100
    # ========================================================================
    def inject_pv_eod_scale_error(self, dataset: pd.DataFrame, scale_factor: float = 100.0,
                                  trade_count: int = None) -> pd.DataFrame:
        """
        Inject scaling error in End_PV for 0.5-1% of trades, 1 day.
        
        Formula: End_PV' = End_PV * 100 (or /100)
        
        Real-world: Unit/currency/notional scaling error, decimal placement bug.
        
        Args:
            dataset: Input dataset
            scale_factor: Scale multiplier (100 or 0.01)
            
        Returns:
            Dataset with scale errors
        """
        df = dataset.copy()
        selected_trades = self._select_random_trades(df, trade_count)
        eligible_mask = self._eligible_mask(df)
        
        for trade_id in selected_trades:
            trade_dates = self._select_random_dates(
                df[df[main_column.TRADE] == trade_id], 1, 1
            )
            if not trade_dates:
                continue
            
            date = trade_dates[0]
            mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == date)
            mask &= eligible_mask
            
            # Randomly choose *100 or /100
            factor = scale_factor if np.random.random() > 0.5 else 1.0 / scale_factor
            df.loc[mask, pnl_column.END] *= factor
            df.loc[mask, main_column.RECORD_TYPE] = "Injected_PV_Scale"
        
        return df
    
    # ========================================================================
    # Scenario 5: PV_EoD_SignFlip
    # ========================================================================
    def inject_pv_eod_sign_flip(self, dataset: pd.DataFrame, trade_count: int = None) -> pd.DataFrame:
        """
        Inject sign flip in End_PV for 0.5-1% of trades, 1 day.
        
        Formula: End_PV' = -End_PV
        
        Real-world: Sign convention inversion, wrong direction, mapping inversion.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with sign flips
        """
        df = dataset.copy()
        selected_trades = self._select_random_trades(df, trade_count)
        eligible_mask = self._eligible_mask(df)
        
        for trade_id in selected_trades:
            trade_dates = self._select_random_dates(
                df[df[main_column.TRADE] == trade_id], 1, 1
            )
            if not trade_dates:
                continue
            
            date = trade_dates[0]
            mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == date)
            mask &= eligible_mask
            df.loc[mask, pnl_column.END] *= -1
            df.loc[mask, main_column.RECORD_TYPE] = "Injected_PV_SignFlip"
        
        return df
    
    # ========================================================================
    # Scenario 6: Slice_CreditSingle_Spike_Point
    # ========================================================================
    def inject_slice_credit_single_spike(self, dataset: pd.DataFrame, severity: float = SEVERITY_MEDIUM,
                                         trade_count: int = None) -> pd.DataFrame:
        """
        Inject single-day spike in Credit_Single_PnL for 1-3% of trades.
        
        Formula: Credit_Single_PnL' = Credit_Single_PnL + s*k*MAD_seg(Credit_Single_PnL)
        
        Real-world: Component PnL spike due to factor mispricing, wrong bucket.
        
        Args:
            dataset: Input dataset
            severity: Severity multiplier (k)
            
        Returns:
            Dataset with injected spikes
        """
        df = dataset.copy()
        
        if not self._mad_stats:
            train_data = self._get_train_data(df)
            self._compute_mad_stats(train_data, [pnl_column.CREDIT_SINGLE])
        
        selected_trades = self._select_random_trades(df, trade_count)
        eligible_mask = self._eligible_mask(df)
        
        for trade_id in selected_trades:
            trade_dates = self._select_random_dates(
                df[df[main_column.TRADE] == trade_id], 1, 1
            )
            if not trade_dates:
                continue
            
            date = trade_dates[0]
            mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == date)
            mask &= eligible_mask
            
            if mask.sum() > 0:
                trade_type = df.loc[mask, main_column.TRADE_TYPE].iloc[0]
                mad = self._get_mad(trade_type, pnl_column.CREDIT_SINGLE)
                sign = self._random_sign()
                
                df.loc[mask, pnl_column.CREDIT_SINGLE] += sign * severity * mad
                df.loc[mask, main_column.RECORD_TYPE] = "Injected_Slice_Spike"
        
        return df
    
    # ========================================================================
    # Scenario 7: Slice_CreditSingle_StaleCopy
    # ========================================================================
    def inject_slice_credit_single_stale(self, dataset: pd.DataFrame, trade_count: int = None) -> pd.DataFrame:
        """
        Inject stale values in Credit_Single_PnL for 1-2% of trades, 5-15 days.
        
        Formula: Credit_Single_PnL(t)' = Credit_Single_PnL(t-1)
        
        Real-world: One pricer module stuck while others update normally.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with stale slice values
        """
        df = dataset.copy()
        selected_trades = self._select_random_trades(df, trade_count)
        eligible_mask = self._eligible_mask(df)
        
        for trade_id in selected_trades:
            trade_data = df[df[main_column.TRADE] == trade_id].sort_values(main_column.DATE)
            oos_data = trade_data[self._eligible_mask(trade_data)]
            
            if len(oos_data) < 5:
                continue
            
            n_days = np.random.randint(5, min(16, len(oos_data) + 1))
            start_idx = np.random.randint(0, len(oos_data) - n_days + 1)
            affected_indices = oos_data.iloc[start_idx:start_idx + n_days].index
            
            for i, idx in enumerate(affected_indices):
                if i == 0:
                    prev_idx = oos_data.index[start_idx - 1] if start_idx > 0 else idx
                    stale_value = df.loc[prev_idx, pnl_column.CREDIT_SINGLE]
                
                if eligible_mask.loc[idx]:
                    df.loc[idx, pnl_column.CREDIT_SINGLE] = stale_value
                    df.loc[idx, main_column.RECORD_TYPE] = "Injected_Slice_Stale"
        
        return df
    
    # ========================================================================
    # Scenario 8: Slice_Reallocation_Bug
    # ========================================================================
    def inject_slice_reallocation_bug(self, dataset: pd.DataFrame, severity: float = SEVERITY_MEDIUM,
                                      trade_count: int = None) -> pd.DataFrame:
        """
        Inject misallocation between Credit_Single_PnL and a donor slice.
        
        Formula: 
            Δ = s*k*MAD_seg(Credit_Single_PnL)
            Credit_Single_PnL' = Credit_Single_PnL + Δ
            DonorSlice' = DonorSlice - Δ
        
        Real-world: Misallocation between components - totals look fine but slice mix is wrong.
        
        Args:
            dataset: Input dataset
            severity: Severity multiplier (k)
            
        Returns:
            Dataset with reallocation bugs
        """
        df = dataset.copy()
        
        if not self._mad_stats:
            train_data = self._get_train_data(df)
            self._compute_mad_stats(train_data, [pnl_column.CREDIT_SINGLE])
        
        selected_trades = self._select_random_trades(df, trade_count)
        eligible_mask = self._eligible_mask(df)
        
        # Choose donor slice (could be Rates, Model, etc.)
        donor_slices = [pnl_column.RATES, pnl_column.MODEL, pnl_column.MISC]
        
        for trade_id in selected_trades:
            trade_dates = self._select_random_dates(
                df[df[main_column.TRADE] == trade_id], 1, 1
            )
            if not trade_dates:
                continue
            
            date = trade_dates[0]
            mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == date)
            mask &= eligible_mask
            
            if mask.sum() > 0:
                trade_type = df.loc[mask, main_column.TRADE_TYPE].iloc[0]
                mad = self._get_mad(trade_type, pnl_column.CREDIT_SINGLE)
                sign = self._random_sign()
                delta = sign * severity * mad
                
                # Pick a donor slice randomly
                donor = np.random.choice(donor_slices)
                
                df.loc[mask, pnl_column.CREDIT_SINGLE] += delta
                df.loc[mask, donor] -= delta
                df.loc[mask, main_column.RECORD_TYPE] = "Injected_Reallocation"
        
        return df
    
    # ========================================================================
    # Scenario 9: TotalPnL_IdentityBreak
    # ========================================================================
    def inject_total_pnl_identity_break(self, dataset: pd.DataFrame, severity: float = SEVERITY_MEDIUM,
                                        trade_count: int = None) -> pd.DataFrame:
        """
        Break reconciliation by changing Credit_Single_PnL without updating Total_PnL.
        
        Formula: Credit_Single_PnL' = Credit_Single_PnL + Δ, but keep Total_PnL unchanged
        
        Real-world: Broken reconciliation/aggregation, stale totals, downstream join failure.
        
        Args:
            dataset: Input dataset
            severity: Severity multiplier (k)
            
        Returns:
            Dataset with identity breaks
        """
        df = dataset.copy()
        
        if not self._mad_stats:
            train_data = self._get_train_data(df)
            self._compute_mad_stats(train_data, [pnl_column.CREDIT_SINGLE])
        
        selected_trades = self._select_random_trades(df, trade_count)
        eligible_mask = self._eligible_mask(df)
        
        for trade_id in selected_trades:
            trade_dates = self._select_random_dates(
                df[df[main_column.TRADE] == trade_id], 1, 1
            )
            if not trade_dates:
                continue
            
            date = trade_dates[0]
            mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == date)
            mask &= eligible_mask
            
            if mask.sum() > 0:
                trade_type = df.loc[mask, main_column.TRADE_TYPE].iloc[0]
                mad = self._get_mad(trade_type, pnl_column.CREDIT_SINGLE)
                sign = self._random_sign()
                
                # Change slice but NOT the total
                df.loc[mask, pnl_column.CREDIT_SINGLE] += sign * severity * mad
                # Total_PnL remains unchanged, breaking the identity
                df.loc[mask, main_column.RECORD_TYPE] = "Injected_IdentityBreak"
        
        return df
    
    # ========================================================================
    # Scenario 10: CrossFamily_Inconsistency_PVvsPnL
    # ========================================================================
    def inject_cross_family_inconsistency(self, dataset: pd.DataFrame, severity: float = SEVERITY_MEDIUM,
                                          trade_count: int = None) -> pd.DataFrame:
        """
        Create inconsistency between End_PV and Total_PnL.
        
        Formula: End_PV' = End_PV + s*k*MAD_seg(End_PV), Total_PnL' = Total_PnL + ε (small noise)
        
        Real-world: PV feed/date alignment issue - PV changes but PnL pipeline doesn't match.
        
        Args:
            dataset: Input dataset
            severity: Severity multiplier (k)
            
        Returns:
            Dataset with cross-family inconsistencies
        """
        df = dataset.copy()
        
        if not self._mad_stats:
            train_data = self._get_train_data(df)
            self._compute_mad_stats(train_data, [pnl_column.END, pnl_column.TOTAL])
        
        selected_trades = self._select_random_trades(df, trade_count)
        eligible_mask = self._eligible_mask(df)
        
        for trade_id in selected_trades:
            trade_dates = self._select_random_dates(
                df[df[main_column.TRADE] == trade_id], 1, 1
            )
            if not trade_dates:
                continue
            
            date = trade_dates[0]
            mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == date)
            mask &= eligible_mask
            
            if mask.sum() > 0:
                trade_type = df.loc[mask, main_column.TRADE_TYPE].iloc[0]
                mad_end = self._get_mad(trade_type, pnl_column.END)
                mad_total = self._get_mad(trade_type, pnl_column.TOTAL)
                sign = self._random_sign()
                
                # Large change to End_PV
                df.loc[mask, pnl_column.END] += sign * severity * mad_end
                
                # Small noise to Total_PnL (inconsistent)
                epsilon = np.random.normal(0, 0.1 * mad_total)
                df.loc[mask, pnl_column.TOTAL] += epsilon
                df.loc[mask, main_column.RECORD_TYPE] = "Injected_CrossFamily"
        
        return df


class CdsOutlierInjector(OutlierInjector):
    """
    Credit Delta Single (CDS)-specific outlier injector implementing 8 scenarios.
    
    Implements various data quality and risk modeling scenarios based on real-world problems
    in credit risk systems.
    """
    
    def inject(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject outliers into the CDS dataset.
        
        Applies all 8 CDS scenarios sequentially, each respecting eligible OOS rows and trade types.
        
        Args:
            dataset: The CDS dataset to inject outliers into
            
        Returns:
            The modified dataset with outliers injected
        """
        df = dataset.copy()
        df = self._ensure_record_type(df)
        
        # Compute statistics once for all scenarios
        train_data = self._get_train_data(df)
        self._compute_mad_stats(train_data, [cds_column.CREDIT_DELTA_SINGLE])
        
        # Apply all scenarios sequentially
        df = self.inject_cd_drift(df, k=6, alpha=0.5)
        df = self.inject_cd_stale_value(df)
        df = self.inject_cd_cluster_shock_3d(df, k=6, alpha=0.5)
        df = self.inject_cd_trade_type_wide_shock(df, k=6, alpha=0.5)
        df = self.inject_cd_point_shock(df, k=6, alpha=0.5)
        df = self.inject_cd_sign_flip(df)
        df = self.inject_cd_scale_error(df)
        df = self.inject_cd_sudden_zero(df)
        
        return df
    
    # ========================================================================
    # Scenario 1: CD_Drift - Linear drift over T days
    # ========================================================================
    def inject_cd_drift(self, dataset: pd.DataFrame, k: float = 6, alpha: float = 0.5) -> pd.DataFrame:
        """
        Inject linear drift in CD values (gradual deterioration / creeping issue).
        
        Formula: For i=0..T−1: Δ(t+i)' = Δ(t+i) + (i/(T−1))·coefficient·Scale
        - Basis/Basket/SingleName/Tranche: use k·Scale_trade
        - Index: use α·Range_trade
        - Tranche: excluded (skip)
        
        Args:
            dataset: Input dataset
            k: Additive coefficient for non-Index trades
            alpha: Relative coefficient for Index trades
            
        Returns:
            Dataset with drift injections
        """
        df = dataset.copy()
        eligible_mask = self._eligible_mask(df)
        
        scenario_config = {
            'Basis': {'trades': 1, 'days': 15},
            'Basket': {'trades': 1, 'days': 10},
            'Index': {'trades': 1, 'days': 15},
            'SingleName': {'trades': 1, 'days': 15},
            # 'Tranche' is excluded
        }
        
        for trade_type, config in scenario_config.items():
            type_data = df[df[main_column.TRADE_TYPE] == trade_type]
            type_eligible = type_data[eligible_mask.loc[type_data.index]]
            
            if type_eligible.empty:
                continue
            
            # Select random trades of this type
            all_trades = type_eligible[main_column.TRADE].unique()
            n_trades = min(config['trades'], len(all_trades))
            selected_trades = np.random.choice(all_trades, size=n_trades, replace=False).tolist()
            
            for trade_id in selected_trades:
                trade_data = df[(df[main_column.TRADE] == trade_id) & 
                               (df[main_column.TRADE_TYPE] == trade_type)].sort_values(main_column.DATE)
                oos_data = trade_data[eligible_mask.loc[trade_data.index]]
                
                if len(oos_data) < config['days']:
                    continue
                
                # Select last consecutive days
                T = config['days']
                last_dates = sorted(oos_data[main_column.DATE].unique())[-T:]
                
                for i, date in enumerate(last_dates):
                    mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == date)
                    mask &= eligible_mask
                    
                    if mask.sum() > 0:
                        coefficient = k if trade_type != 'Index' else alpha
                        scale_attr = 'scale_trade' if trade_type != 'Index' else 'range_trade'
                        scale = self._get_mad(trade_type, cds_column.CREDIT_DELTA_SINGLE)  # Using MAD as proxy
                        
                        drift = (i / (T - 1)) * coefficient * scale if T > 1 else 0
                        df.loc[mask, cds_column.CREDIT_DELTA_SINGLE] += drift
                        df.loc[mask, main_column.RECORD_TYPE] = "CD_Drift"
        
        return df
    
    # ========================================================================
    # Scenario 2: CD_StaleValue - Stuck values
    # ========================================================================
    def inject_cd_stale_value(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject stale/stuck values (frozen feed).
        
        Formula: Δ(t)' = Δ(t−1)
        
        Applied to 1 trade per trade type for 5 first consecutive days.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with stale values
        """
        df = dataset.copy()
        eligible_mask = self._eligible_mask(df)
        
        scenario_config = {
            'Basis': {'trades': 1, 'days': 5},
            'Basket': {'trades': 1, 'days': 5},
            'Index': {'trades': 1, 'days': 5},
            'SingleName': {'trades': 1, 'days': 5},
            'Tranche': {'trades': 1, 'days': 5},
        }
        
        for trade_type, config in scenario_config.items():
            type_data = df[df[main_column.TRADE_TYPE] == trade_type]
            type_eligible = type_data[eligible_mask.loc[type_data.index]]
            
            if type_eligible.empty:
                continue
            
            all_trades = type_eligible[main_column.TRADE].unique()
            n_trades = min(config['trades'], len(all_trades))
            selected_trades = np.random.choice(all_trades, size=n_trades, replace=False).tolist()
            
            for trade_id in selected_trades:
                trade_data = df[(df[main_column.TRADE] == trade_id) & 
                               (df[main_column.TRADE_TYPE] == trade_type)].sort_values(main_column.DATE)
                oos_data = trade_data[eligible_mask.loc[trade_data.index]]
                
                if len(oos_data) < config['days']:
                    continue
                
                # Select first consecutive days
                first_dates = sorted(oos_data[main_column.DATE].unique())[:config['days']]
                
                # Get value from day before first day
                all_dates = sorted(trade_data[main_column.DATE].unique())
                first_date_idx = all_dates.index(first_dates[0])
                stale_value = df[(df[main_column.TRADE] == trade_id) & 
                                (df[main_column.DATE] == all_dates[first_date_idx - 1])][cds_column.CREDIT_DELTA_SINGLE]
                
                if stale_value.empty:
                    continue
                
                stale_value = stale_value.iloc[0]
                
                for date in first_dates:
                    mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == date)
                    mask &= eligible_mask
                    
                    if mask.sum() > 0:
                        df.loc[mask, cds_column.CREDIT_DELTA_SINGLE] = stale_value
                        df.loc[mask, main_column.RECORD_TYPE] = "CD_StaleValue"
        
        return df
    
    # ========================================================================
    # Scenario 3: CD_ClusterShock_3d - Short feed issue
    # ========================================================================
    def inject_cd_cluster_shock_3d(self, dataset: pd.DataFrame, k: float = 6, alpha: float = 0.5) -> pd.DataFrame:
        """
        Inject short-term cluster shock (fallback curve) for 3 days.
        
        Formula: For i=0..2: Δ(t+i)' = Δ(t+i) + k·Scale_tt or α·Range_tt
        
        Args:
            dataset: Input dataset
            k: Additive coefficient for non-Index trades
            alpha: Relative coefficient for Index trades
            
        Returns:
            Dataset with cluster shocks
        """
        df = dataset.copy()
        eligible_mask = self._eligible_mask(df)
        
        scenario_config = {
            'Basis': {'trades': 1, 'days': 3},
            'Basket': {'trades': 1, 'days': 3},
            'Index': {'trades': 1, 'days': 3},
            'SingleName': {'trades': 1, 'days': 3},
            'Tranche': {'trades': 1, 'days': 3},
        }
        
        for trade_type, config in scenario_config.items():
            type_data = df[df[main_column.TRADE_TYPE] == trade_type]
            type_eligible = type_data[eligible_mask.loc[type_data.index]]
            
            if type_eligible.empty:
                continue
            
            all_trades = type_eligible[main_column.TRADE].unique()
            n_trades = min(config['trades'], len(all_trades))
            selected_trades = np.random.choice(all_trades, size=n_trades, replace=False).tolist()
            
            for trade_id in selected_trades:
                trade_data = df[(df[main_column.TRADE] == trade_id) & 
                               (df[main_column.TRADE_TYPE] == trade_type)].sort_values(main_column.DATE)
                oos_data = trade_data[eligible_mask.loc[trade_data.index]]
                
                if len(oos_data) < config['days']:
                    continue
                
                # Select random consecutive days
                all_oos_dates = sorted(oos_data[main_column.DATE].unique())
                start_idx = np.random.randint(0, len(all_oos_dates) - config['days'] + 1)
                affected_dates = all_oos_dates[start_idx:start_idx + config['days']]
                
                coefficient = k if trade_type != 'Index' else alpha
                scale = self._get_mad(trade_type, cds_column.CREDIT_DELTA_SINGLE)
                shock = coefficient * scale
                
                for date in affected_dates:
                    mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == date)
                    mask &= eligible_mask
                    
                    if mask.sum() > 0:
                        df.loc[mask, cds_column.CREDIT_DELTA_SINGLE] += shock
                        df.loc[mask, main_column.RECORD_TYPE] = "CD_ClusterShock_3d"
        
        return df
    
    # ========================================================================
    # Scenario 4: CD_TradeTypeWide_Shock - Systemic shock
    # ========================================================================
    def inject_cd_trade_type_wide_shock(self, dataset: pd.DataFrame, k: float = 6, alpha: float = 0.5) -> pd.DataFrame:
        """
        Inject systemic shock affecting a percentage of trades of each type (1 day).
        
        Formula: Δ' = Δ + k·Scale_tt or α·Range_tt
        
        Args:
            dataset: Input dataset
            k: Additive coefficient for non-Index trades
            alpha: Relative coefficient for Index trades
            
        Returns:
            Dataset with systemic shocks
        """
        df = dataset.copy()
        eligible_mask = self._eligible_mask(df)
        
        scenario_config = {
            'Basis': {'percent': 0.5},
            'Basket': {'percent': 1.0},
            'Index': {'percent': 0.5},
            'SingleName': {'percent': 0.5},
            'Tranche': {'percent': 1.0},
        }
        
        for trade_type, config in scenario_config.items():
            type_data = df[df[main_column.TRADE_TYPE] == trade_type]
            type_eligible = type_data[eligible_mask.loc[type_data.index]]
            
            if type_eligible.empty:
                continue
            
            # Calculate number of trades based on percentage
            all_trades = type_eligible[main_column.TRADE].unique()
            n_trades = max(1, int(len(all_trades) * config['percent']))
            selected_trades = np.random.choice(all_trades, size=n_trades, replace=False).tolist()
            
            # Select random day
            all_oos_dates = sorted(type_eligible[main_column.DATE].unique())
            if not all_oos_dates:
                continue
            shock_date = np.random.choice(all_oos_dates)
            
            coefficient = k if trade_type != 'Index' else alpha
            scale = self._get_mad(trade_type, cds_column.CREDIT_DELTA_SINGLE)
            shock = coefficient * scale
            
            for trade_id in selected_trades:
                mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == shock_date)
                mask &= eligible_mask
                
                if mask.sum() > 0:
                    df.loc[mask, cds_column.CREDIT_DELTA_SINGLE] += shock
                    df.loc[mask, main_column.RECORD_TYPE] = "CD_TradeTypeWide_Shock"
        
        return df
    
    # ========================================================================
    # Scenario 5: CD_PointShock - One-day bad snapshot
    # ========================================================================
    def inject_cd_point_shock(self, dataset: pd.DataFrame, k: float = 6, alpha: float = 0.5) -> pd.DataFrame:
        """
        Inject one-day point shocks (transient glitch).
        
        Formula: Δ' = Δ + k·Scale_tt or α·Range_tt
        
        Args:
            dataset: Input dataset
            k: Additive coefficient for non-Index trades
            alpha: Relative coefficient for Index trades
            
        Returns:
            Dataset with point shocks
        """
        df = dataset.copy()
        eligible_mask = self._eligible_mask(df)
        
        scenario_config = {
            'Basis': {'trades': 3},
            'Basket': {'trades': 1},
            'Index': {'trades': 2},
            'SingleName': {'trades': 3},
            'Tranche': {'trades': 1},
        }
        
        for trade_type, config in scenario_config.items():
            type_data = df[df[main_column.TRADE_TYPE] == trade_type]
            type_eligible = type_data[eligible_mask.loc[type_data.index]]
            
            if type_eligible.empty:
                continue
            
            all_trades = type_eligible[main_column.TRADE].unique()
            n_trades = min(config['trades'], len(all_trades))
            selected_trades = np.random.choice(all_trades, size=n_trades, replace=False).tolist()
            
            for trade_id in selected_trades:
                trade_data = df[(df[main_column.TRADE] == trade_id) & 
                               (df[main_column.TRADE_TYPE] == trade_type)]
                trade_eligible = trade_data[eligible_mask.loc[trade_data.index]]
                
                if trade_eligible.empty:
                    continue
                
                # Select random date
                all_dates = sorted(trade_eligible[main_column.DATE].unique())
                shock_date = np.random.choice(all_dates)
                
                mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == shock_date)
                mask &= eligible_mask
                
                if mask.sum() > 0:
                    coefficient = k if trade_type != 'Index' else alpha
                    scale = self._get_mad(trade_type, cds_column.CREDIT_DELTA_SINGLE)
                    shock = coefficient * scale
                    
                    df.loc[mask, cds_column.CREDIT_DELTA_SINGLE] += shock
                    df.loc[mask, main_column.RECORD_TYPE] = "CD_PointShock"
        
        return df
    
    # ========================================================================
    # Scenario 6: CD_SignFlip - Sign convention bug
    # ========================================================================
    def inject_cd_sign_flip(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject sign flip (sign convention bug, 1 day per trade).
        
        Formula: Δ' = −Δ
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with sign flips
        """
        df = dataset.copy()
        eligible_mask = self._eligible_mask(df)
        
        scenario_config = {
            'Basis': {'trades': 3},
            'Basket': {'trades': 1},
            'Index': {'trades': 2},
            'SingleName': {'trades': 3},
            'Tranche': {'trades': 1},
        }
        
        for trade_type, config in scenario_config.items():
            type_data = df[df[main_column.TRADE_TYPE] == trade_type]
            type_eligible = type_data[eligible_mask.loc[type_data.index]]
            
            if type_eligible.empty:
                continue
            
            all_trades = type_eligible[main_column.TRADE].unique()
            n_trades = min(config['trades'], len(all_trades))
            selected_trades = np.random.choice(all_trades, size=n_trades, replace=False).tolist()
            
            for trade_id in selected_trades:
                trade_data = df[(df[main_column.TRADE] == trade_id) & 
                               (df[main_column.TRADE_TYPE] == trade_type)]
                trade_eligible = trade_data[eligible_mask.loc[trade_data.index]]
                
                if trade_eligible.empty:
                    continue
                
                all_dates = sorted(trade_eligible[main_column.DATE].unique())
                flip_date = np.random.choice(all_dates)
                
                mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == flip_date)
                mask &= eligible_mask
                
                if mask.sum() > 0:
                    df.loc[mask, cds_column.CREDIT_DELTA_SINGLE] *= -1
                    df.loc[mask, main_column.RECORD_TYPE] = "CD_SignFlip"
        
        return df
    
    # ========================================================================
    # Scenario 7: CD_ScaleError - Unit/decimal/notional scale bug
    # ========================================================================
    def inject_cd_scale_error(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject scale error (unit/decimal/notional bug, 1 day per trade).
        
        Formula: Δ' = multiplier·Δ (typically 100x or 1000x for Index)
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with scale errors
        """
        df = dataset.copy()
        eligible_mask = self._eligible_mask(df)
        
        scenario_config = {
            'Basis': {'trades': 3, 'scale': 100},
            'Basket': {'trades': 1, 'scale': 100},
            'Index': {'trades': 2, 'scale': 1000},
            'SingleName': {'trades': 3, 'scale': 100},
            'Tranche': {'trades': 1, 'scale': 100},
        }
        
        for trade_type, config in scenario_config.items():
            type_data = df[df[main_column.TRADE_TYPE] == trade_type]
            type_eligible = type_data[eligible_mask.loc[type_data.index]]
            
            if type_eligible.empty:
                continue
            
            all_trades = type_eligible[main_column.TRADE].unique()
            n_trades = min(config['trades'], len(all_trades))
            selected_trades = np.random.choice(all_trades, size=n_trades, replace=False).tolist()
            
            for trade_id in selected_trades:
                trade_data = df[(df[main_column.TRADE] == trade_id) & 
                               (df[main_column.TRADE_TYPE] == trade_type)]
                trade_eligible = trade_data[eligible_mask.loc[trade_data.index]]
                
                if trade_eligible.empty:
                    continue
                
                all_dates = sorted(trade_eligible[main_column.DATE].unique())
                error_date = np.random.choice(all_dates)
                
                mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == error_date)
                mask &= eligible_mask
                
                if mask.sum() > 0:
                    df.loc[mask, cds_column.CREDIT_DELTA_SINGLE] *= config['scale']
                    df.loc[mask, main_column.RECORD_TYPE] = "CD_ScaleError"
        
        return df
    
    # ========================================================================
    # Scenario 8: CD_SuddenZero - Valuation error leading to 0
    # ========================================================================
    def inject_cd_sudden_zero(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject sudden zero values (valuation error, 1 day per trade).
        
        Formula: Δ' = 0
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with sudden zeros
        """
        df = dataset.copy()
        eligible_mask = self._eligible_mask(df)
        
        scenario_config = {
            'Basis': {'trades': 3},
            'Basket': {'trades': 1},
            'Index': {'trades': 2},
            'SingleName': {'trades': 3},
            'Tranche': {'trades': 1},
        }
        
        for trade_type, config in scenario_config.items():
            type_data = df[df[main_column.TRADE_TYPE] == trade_type]
            type_eligible = type_data[eligible_mask.loc[type_data.index]]
            
            if type_eligible.empty:
                continue
            
            all_trades = type_eligible[main_column.TRADE].unique()
            n_trades = min(config['trades'], len(all_trades))
            selected_trades = np.random.choice(all_trades, size=n_trades, replace=False).tolist()
            
            for trade_id in selected_trades:
                trade_data = df[(df[main_column.TRADE] == trade_id) & 
                               (df[main_column.TRADE_TYPE] == trade_type)]
                trade_eligible = trade_data[eligible_mask.loc[trade_data.index]]
                
                if trade_eligible.empty:
                    continue
                
                all_dates = sorted(trade_eligible[main_column.DATE].unique())
                zero_date = np.random.choice(all_dates)
                
                mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == zero_date)
                mask &= eligible_mask
                
                if mask.sum() > 0:
                    df.loc[mask, cds_column.CREDIT_DELTA_SINGLE] = 0
                    df.loc[mask, main_column.RECORD_TYPE] = "CD_SuddenZero"
        
        return df

