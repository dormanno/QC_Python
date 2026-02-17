"""
CDS (Credit Delta Single) outlier injector implementing 8 data quality scenarios.

Implements 8 specific data quality issue scenarios for Credit Delta Single (CDS)
datasets as specified in CDS_InjectionScenarios.md.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from column_names import cds_column, main_column
from .base import OutlierInjector


class CdsOutlierInjector(OutlierInjector):
    """
    CDS (Credit Delta Single) outlier injector implementing 8 data quality scenarios.
    
    Implements 8 specific data quality issue scenarios for Credit Delta Single (CDS)
    datasets as specified in CDS_InjectionScenarios.md.
    """
    
    # Re-export severity constants for convenience in default parameters
    SEVERITY_SMALL = OutlierInjector.SEVERITY_SMALL
    SEVERITY_MEDIUM = OutlierInjector.SEVERITY_MEDIUM
    SEVERITY_HIGH = OutlierInjector.SEVERITY_HIGH
    SEVERITY_EXTREME = OutlierInjector.SEVERITY_EXTREME
    
    def inject(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject CDS outliers into the dataset.
        
        Applies all 8 CDS scenarios sequentially, each respecting eligible OOS rows.
        
        Args:
            dataset: The CDS dataset to inject outliers into
            
        Returns:
            The modified dataset with outliers injected
        """
        df = dataset.copy()
        df = self._ensure_record_type(df)

        # Apply all scenarios sequentially
        df = self.inject_cd_drift(df)
        df = self.inject_cd_stale_value(df)
        df = self.inject_cd_cluster_shock_3d(df)
        df = self.inject_cd_trade_type_wide_shock(df)
        df = self.inject_cd_point_shock(df)
        df = self.inject_cd_sign_flip(df)
        df = self.inject_cd_scale_error(df)
        df = self.inject_cd_sudden_zero(df)

        return df
    
    def _get_trade_type_days(self) -> Dict[str, int]:
        """Get T (number of days) for each trade type from CDS scenarios spec."""
        return {
            "Basis": 20,
            "Basket": 15,
            "Index": 10,
            "SingleName": 30,
            "Tranche": 25,
        }
    
    def _get_trade_type_trade_counts(self) -> Dict[str, int]:
        """Get # trades for each scenario/trade type combination from CDS scenarios spec."""
        # Key: (scenario, trade_type) -> # trades to inject
        return {
            ("CD_Drift", "Basis"): 1,
            ("CD_Drift", "Basket"): 1,
            ("CD_Drift", "Index"): 1,
            ("CD_Drift", "SingleName"): 1,
            ("CD_Drift", "Tranche"): 0,  # Excluded
            ("CD_StaleValue", "Basis"): 1,
            ("CD_StaleValue", "Basket"): 1,
            ("CD_StaleValue", "Index"): 1,
            ("CD_StaleValue", "SingleName"): 1,
            ("CD_StaleValue", "Tranche"): 1,
            ("CD_ClusterShock_3d", "Basis"): 1,
            ("CD_ClusterShock_3d", "Basket"): 1,
            ("CD_ClusterShock_3d", "Index"): 1,
            ("CD_ClusterShock_3d", "SingleName"): 1,
            ("CD_ClusterShock_3d", "Tranche"): 1,
            ("CD_TradeTypeWide_Shock", "Basis"): 0.5,  # 50% on first OOS date
            ("CD_TradeTypeWide_Shock", "Basket"): 1.0,  # 100%
            ("CD_TradeTypeWide_Shock", "Index"): 0.5,  # 50%
            ("CD_TradeTypeWide_Shock", "SingleName"): 0.5,  # 50%
            ("CD_TradeTypeWide_Shock", "Tranche"): 1.0,  # 100%
            ("CD_PointShock", "Basis"): 3,
            ("CD_PointShock", "Basket"): 1,
            ("CD_PointShock", "Index"): 2,
            ("CD_PointShock", "SingleName"): 3,
            ("CD_PointShock", "Tranche"): 1,
            ("CD_SignFlip", "Basis"): 3,
            ("CD_SignFlip", "Basket"): 1,
            ("CD_SignFlip", "Index"): 2,
            ("CD_SignFlip", "SingleName"): 3,
            ("CD_SignFlip", "Tranche"): 1,
            ("CD_ScaleError", "Basis"): 3,
            ("CD_ScaleError", "Basket"): 1,
            ("CD_ScaleError", "Index"): 2,
            ("CD_ScaleError", "SingleName"): 3,
            ("CD_ScaleError", "Tranche"): 1,
            ("CD_SuddenZero", "Basis"): 3,
            ("CD_SuddenZero", "Basket"): 1,
            ("CD_SuddenZero", "Index"): 2,
            ("CD_SuddenZero", "SingleName"): 3,
            ("CD_SuddenZero", "Tranche"): 1,
        }
    
    def _get_drift_days_by_type(self) -> Dict[str, int]:
        """Get # days for CD_Drift by trade type from spec."""
        return {
            "Basis": 15,
            "Basket": 10,
            "Index": 15,
            "SingleName": 15,
            "Tranche": 0,  # Excluded
        }
    
    def _get_stale_days(self) -> int:
        """Get # days for CD_StaleValue (same for all trade types)."""
        return 5
    
    # ========================================================================
    # Scenario 1: CD_Drift
    # ========================================================================
    def inject_cd_drift(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject linear drift in Credit Delta over last T consecutive days.
        
        Per trade type spec: select 1 trade, apply drift over T days = T records.
        Formula: Δ'(i) = Δ(i) + (i/(T-1))·k·Scale_trade for i=0..T-1
        
        Real-world: Gradual model degradation, stale curve, wrong rates assumption.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with drift injected
        """
        df = dataset.copy()
        
        if not self._mad_stats:
            train_data = self._get_train_data(df)
            self._compute_mad_stats(train_data, [cds_column.CREDIT_DELTA_SINGLE])
        
        drift_days_by_type = self._get_drift_days_by_type()
        eligible_mask = self._eligible_mask(df)
        
        # Apply to exactly 1 trade per trade type (spec says "1 trade" per type)
        for trade_type in df[main_column.TRADE_TYPE].unique():
            T = drift_days_by_type.get(trade_type, 0)
            if T == 0:  # Tranche is excluded
                continue
            
            # Get all trades of this type and their eligible dates
            type_trades = df[(df[main_column.TRADE_TYPE] == trade_type) & eligible_mask][main_column.TRADE].unique()
            if len(type_trades) == 0:
                continue
            
            # Select 1 random trade of this type
            selected_trade = np.random.choice(type_trades)
            
            trade_data = df[df[main_column.TRADE] == selected_trade].sort_values(main_column.DATE)
            eligible_trade = trade_data[eligible_mask[trade_data.index]]
            
            if len(eligible_trade) < T:
                continue  # Not enough days for this trade
            
            # Get last T dates for this trade
            all_eligible_dates = sorted(eligible_trade[main_column.DATE].unique())
            last_dates = all_eligible_dates[-T:]
            
            # Get MAD for this trade
            mad = self._get_mad(trade_type, cds_column.CREDIT_DELTA_SINGLE)
            sign = self._random_sign()
            
            # Apply drift formula: Δ'(i) = Δ(i) + (i/(T-1))·k·MAD for i=0..T-1
            for i, date in enumerate(last_dates):
                mask = (df[main_column.TRADE] == selected_trade) & (df[main_column.DATE] == date)
                mask &= eligible_mask
                
                if mask.sum() > 0:
                    # Linear scaling from 0 to 1 over T days
                    scale_factor = i / (T - 1) if T > 1 else 0
                    drift = sign * scale_factor * self.SEVERITY_MEDIUM * mad
                    df.loc[mask, cds_column.CREDIT_DELTA_SINGLE] += drift
                    df.loc[mask, main_column.RECORD_TYPE] = "CD_Drift"
        
        return df
    
    # ========================================================================
    # Scenario 2: CD_StaleValue
    # ========================================================================
    def inject_cd_stale_value(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject stale/stuck values over first 5 consecutive days per trade type.
        
        Per trade type spec: select 1 trade, copy Δ(t-1) for 5 consecutive days = 5 records.
        Formula: Δ'(t) = Δ(t-1) (copy previous day's value)
        
        Real-world: Valuations not updating, feed stuck, source frozen.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with stale values
        """
        df = dataset.copy()
        eligible_mask = self._eligible_mask(df)
        stale_days = self._get_stale_days()
        
        # Apply to exactly 1 trade per trade type
        for trade_type in df[main_column.TRADE_TYPE].unique():
            # Get all trades of this type with eligible dates
            type_trades = df[(df[main_column.TRADE_TYPE] == trade_type) & eligible_mask][main_column.TRADE].unique()
            if len(type_trades) == 0:
                continue
            
            # Select 1 random trade of this type
            selected_trade = np.random.choice(type_trades)
            
            trade_data = df[df[main_column.TRADE] == selected_trade].sort_values(main_column.DATE)
            eligible_trade = trade_data[eligible_mask[trade_data.index]]
            
            if len(eligible_trade) < stale_days + 1:
                continue  # Need at least stale_days + 1 to have a value to copy
            
            # Get first N eligible dates
            all_eligible_dates = sorted(eligible_trade[main_column.DATE].unique())
            first_dates = all_eligible_dates[:stale_days]
            
            # Get the stale value from the day before the stale period
            stale_value = df[(df[main_column.TRADE] == selected_trade) & 
                            (df[main_column.DATE] == all_eligible_dates[0])][cds_column.CREDIT_DELTA_SINGLE].iloc[0]
            
            # Apply stale value to first N dates
            for date in first_dates:
                mask = (df[main_column.TRADE] == selected_trade) & (df[main_column.DATE] == date)
                mask &= eligible_mask
                
                if mask.sum() > 0:
                    df.loc[mask, cds_column.CREDIT_DELTA_SINGLE] = stale_value
                    df.loc[mask, main_column.RECORD_TYPE] = "CD_StaleValue"
        
        return df
    
    # ========================================================================
    # Scenario 3: CD_ClusterShock_3d
    # ========================================================================
    def inject_cd_cluster_shock_3d(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject 3-day cluster shock.
        
        Per trade type spec: select 1 trade, apply shock over 3 consecutive days = 3 records.
        Formula: Δ'(t) = Δ(t) + k·MAD_tt for 3 consecutive days
        
        Real-world: Market event/shock spanning multiple days, rate spike, credit event.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with cluster shocks
        """
        df = dataset.copy()
        
        if not self._mad_stats:
            train_data = self._get_train_data(df)
            self._compute_mad_stats(train_data, [cds_column.CREDIT_DELTA_SINGLE])
        
        eligible_mask = self._eligible_mask(df)
        shock_days = 3
        
        # Apply to exactly 1 trade per trade type
        for trade_type in df[main_column.TRADE_TYPE].unique():
            # Get all trades of this type with eligible dates
            type_trades = df[(df[main_column.TRADE_TYPE] == trade_type) & eligible_mask][main_column.TRADE].unique()
            if len(type_trades) == 0:
                continue
            
            # Select 1 random trade of this type
            selected_trade = np.random.choice(type_trades)
            
            trade_data = df[df[main_column.TRADE] == selected_trade].sort_values(main_column.DATE)
            eligible_trade = trade_data[eligible_mask[trade_data.index]]
            
            if len(eligible_trade) < shock_days:
                continue
            
            # Select 3 consecutive random dates from eligible dates
            all_eligible_dates = sorted(eligible_trade[main_column.DATE].unique())
            if len(all_eligible_dates) < shock_days:
                continue
            
            start_idx = np.random.randint(0, len(all_eligible_dates) - shock_days + 1)
            shock_dates = all_eligible_dates[start_idx:start_idx + shock_days]
            
            # Get MAD for this trade type
            mad = self._get_mad(trade_type, cds_column.CREDIT_DELTA_SINGLE)
            sign = self._random_sign()
            shock = sign * self.SEVERITY_MEDIUM * mad
            
            # Apply shock to 3 consecutive dates
            for date in shock_dates:
                mask = (df[main_column.TRADE] == selected_trade) & (df[main_column.DATE] == date)
                mask &= eligible_mask
                
                if mask.sum() > 0:
                    df.loc[mask, cds_column.CREDIT_DELTA_SINGLE] += shock
                    df.loc[mask, main_column.RECORD_TYPE] = "CD_ClusterShock_3d"
        
        return df
    
    # ========================================================================
    # Scenario 4: CD_TradeTypeWide_Shock
    # ========================================================================
    def inject_cd_trade_type_wide_shock(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject systemic shock on first OOS date across % of each trade type.
        
        Per trade type spec percentage: count trades on FIRST OOS date (including all OOS),
        but only inject on rows that haven't been injected yet. Select backup trades if
        needed to hit the target count.
        Formula: Δ' = Δ + k·MAD_tt for selected trades on first OOS date
        
        Example: If 100 Basis trades exist on first OOS date, 50% = 50 trades, 1 day each = 50 records.
        
        Real-world: Market-wide repricing event affecting risk category, credit curve shift.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with trade-type-wide shocks
        """
        df = dataset.copy()
        
        if not self._mad_stats:
            train_data = self._get_train_data(df)
            self._compute_mad_stats(train_data, [cds_column.CREDIT_DELTA_SINGLE])
        
        eligible_mask = self._eligible_mask(df)
        trade_type_counts = self._get_trade_type_trade_counts()
        
        # Find first OOS date (include both original OOS and already-injected rows)
        oos_mask = df[main_column.RECORD_TYPE] != "Train"
        oos_dates = sorted(df[oos_mask][main_column.DATE].unique())
        if len(oos_dates) == 0:
            return df
        
        first_oos_date = oos_dates[0]
        
        # For each trade type, select percentage of trades present on first OOS date
        for trade_type in df[main_column.TRADE_TYPE].unique():
            pct = trade_type_counts.get(("CD_TradeTypeWide_Shock", trade_type), 0.5)
            
            # Get trades of this type on the first OOS date (including already-injected, for counting)
            first_date_mask = (df[main_column.DATE] == first_oos_date) & \
                            (df[main_column.TRADE_TYPE] == trade_type) & \
                            oos_mask
            
            trades_on_first_date = df[first_date_mask][main_column.TRADE].unique()
            if len(trades_on_first_date) == 0:
                continue
            
            # Select percentage of these trades (target count)
            target_count = max(1, int(len(trades_on_first_date) * pct))
            
            # Shuffle trades to randomize selection
            shuffled_trades = np.random.permutation(trades_on_first_date)
            
            # Get MAD for this trade type
            mad = self._get_mad(trade_type, cds_column.CREDIT_DELTA_SINGLE)
            sign = self._random_sign()
            shock = sign * self.SEVERITY_MEDIUM * mad
            
            # Inject until we reach target_count, using backup trades if needed
            injected_count = 0
            for trade_id in shuffled_trades:
                if injected_count >= target_count:
                    break
                
                # Try to inject on first OOS date
                mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == first_oos_date)
                mask &= eligible_mask
                
                if mask.sum() > 0:
                    df.loc[mask, cds_column.CREDIT_DELTA_SINGLE] += shock
                    df.loc[mask, main_column.RECORD_TYPE] = "CD_TradeTypeWide_Shock"
                    injected_count += 1
                else:
                    # If first OOS date row is already injected, try any eligible date for this trade
                    trade_mask = (df[main_column.TRADE] == trade_id) & eligible_mask
                    if trade_mask.sum() > 0:
                        # Find first eligible row for this trade and inject
                        trade_rows = df[trade_mask].iloc[:1]
                        idx = trade_rows.index[0]
                        df.loc[idx, cds_column.CREDIT_DELTA_SINGLE] += shock
                        df.loc[idx, main_column.RECORD_TYPE] = "CD_TradeTypeWide_Shock"
                        injected_count += 1
        
        return df
    
    # ========================================================================
    # Scenario 5: CD_PointShock
    # ========================================================================
    def inject_cd_point_shock(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject isolated 1-day shocks for specified trades per type.
        
        Per trade type spec: Basis:3, Basket:1, Index:2, SingleName:3, Tranche:1 trades on random day.
        Formula: Δ' = Δ + k·MAD_tt
        
        Real-world: One-off bad market data snapshot, transient calc glitch, data feed artifact.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with point shocks
        """
        df = dataset.copy()
        
        if not self._mad_stats:
            train_data = self._get_train_data(df)
            self._compute_mad_stats(train_data, [cds_column.CREDIT_DELTA_SINGLE])
        
        eligible_mask = self._eligible_mask(df)
        trade_type_counts = self._get_trade_type_trade_counts()
        
        # Apply to specified number of trades per trade type
        for trade_type in df[main_column.TRADE_TYPE].unique():
            trade_count = int(trade_type_counts.get(("CD_PointShock", trade_type), 0))
            if trade_count == 0:
                continue
            
            # Get all trades of this type with eligible dates
            type_trades = df[(df[main_column.TRADE_TYPE] == trade_type) & eligible_mask][main_column.TRADE].unique()
            if len(type_trades) == 0:
                continue
            
            # Select exactly N trades of this type
            n_to_inject = min(trade_count, len(type_trades))
            selected_trades = np.random.choice(type_trades, size=n_to_inject, replace=False)
            
            # Get MAD for this trade type
            mad = self._get_mad(trade_type, cds_column.CREDIT_DELTA_SINGLE)
            sign = self._random_sign()
            shock = sign * self.SEVERITY_MEDIUM * mad
            
            # Apply shock on 1 random day for each selected trade
            for trade_id in selected_trades:
                trade_data = df[(df[main_column.TRADE] == trade_id) & eligible_mask]
                if len(trade_data) == 0:
                    continue
                
                # Pick 1 random date for this trade
                random_date = np.random.choice(trade_data[main_column.DATE].unique())
                
                mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == random_date)
                mask &= eligible_mask
                
                if mask.sum() > 0:
                    df.loc[mask, cds_column.CREDIT_DELTA_SINGLE] += shock
                    df.loc[mask, main_column.RECORD_TYPE] = "CD_PointShock"
        
        return df
    
    # ========================================================================
    # Scenario 6: CD_SignFlip
    # ========================================================================
    def inject_cd_sign_flip(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject sign flip for specified trades per type on 1 random day.
        
        Per trade type spec: Basis:3, Basket:1, Index:2, SingleName:3, Tranche:1 trades.
        Formula: Δ' = -Δ
        
        Real-world: Sign convention inversion, wrong direction, mapping inversion bug.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with sign flips
        """
        df = dataset.copy()
        eligible_mask = self._eligible_mask(df)
        trade_type_counts = self._get_trade_type_trade_counts()
        
        # Apply to specified number of trades per trade type
        for trade_type in df[main_column.TRADE_TYPE].unique():
            trade_count = int(trade_type_counts.get(("CD_SignFlip", trade_type), 0))
            if trade_count == 0:
                continue
            
            # Get all trades of this type with eligible dates
            type_trades = df[(df[main_column.TRADE_TYPE] == trade_type) & eligible_mask][main_column.TRADE].unique()
            if len(type_trades) == 0:
                continue
            
            # Select exactly N trades of this type
            n_to_inject = min(trade_count, len(type_trades))
            selected_trades = np.random.choice(type_trades, size=n_to_inject, replace=False)
            
            # Apply sign flip on 1 random day for each selected trade
            for trade_id in selected_trades:
                trade_data = df[(df[main_column.TRADE] == trade_id) & eligible_mask]
                if len(trade_data) == 0:
                    continue
                
                # Pick 1 random date for this trade
                random_date = np.random.choice(trade_data[main_column.DATE].unique())
                
                mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == random_date)
                mask &= eligible_mask
                
                if mask.sum() > 0:
                    df.loc[mask, cds_column.CREDIT_DELTA_SINGLE] *= -1
                    df.loc[mask, main_column.RECORD_TYPE] = "CD_SignFlip"
        
        return df
    
    # ========================================================================
    # Scenario 7: CD_ScaleError
    # ========================================================================
    def inject_cd_scale_error(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject scale/unit error for specified trades per type on 1 random day.
        
        Per trade type spec: Basis:3, Basket:1, Index:2, SingleName:3, Tranche:1 trades.
        Trade-type specific scaling:
        - Index: 1000× or /1000 (basis point vs percentage confusion)
        - Others: 100× or /100 (currency/notional scaling error)
        
        Real-world: Unit/currency/notional scaling error, decimal placement bug.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with scale errors
        """
        df = dataset.copy()
        eligible_mask = self._eligible_mask(df)
        trade_type_counts = self._get_trade_type_trade_counts()
        
        # Apply to specified number of trades per trade type
        for trade_type in df[main_column.TRADE_TYPE].unique():
            trade_count = int(trade_type_counts.get(("CD_ScaleError", trade_type), 0))
            if trade_count == 0:
                continue
            
            # Get all trades of this type with eligible dates
            type_trades = df[(df[main_column.TRADE_TYPE] == trade_type) & eligible_mask][main_column.TRADE].unique()
            if len(type_trades) == 0:
                continue
            
            # Select exactly N trades of this type
            n_to_inject = min(trade_count, len(type_trades))
            selected_trades = np.random.choice(type_trades, size=n_to_inject, replace=False)
            
            # Determine scale factor (Index uses 1000, others use 100)
            base_scale = 1000.0 if trade_type == "Index" else 100.0
            
            # Apply scale error on 1 random day for each selected trade
            for trade_id in selected_trades:
                trade_data = df[(df[main_column.TRADE] == trade_id) & eligible_mask]
                if len(trade_data) == 0:
                    continue
                
                # Pick 1 random date for this trade
                random_date = np.random.choice(trade_data[main_column.DATE].unique())
                
                mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == random_date)
                mask &= eligible_mask
                
                if mask.sum() > 0:
                    # Randomly choose ×scale or /scale
                    factor = base_scale if np.random.random() > 0.5 else 1.0 / base_scale
                    df.loc[mask, cds_column.CREDIT_DELTA_SINGLE] *= factor
                    df.loc[mask, main_column.RECORD_TYPE] = "CD_ScaleError"
        
        return df
    
    # ========================================================================
    # Scenario 8: CD_SuddenZero
    # ========================================================================
    def inject_cd_sudden_zero(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject sudden zero values for specified trades per type on 1 random day.
        
        Per trade type spec: Basis:3, Basket:1, Index:2, SingleName:3, Tranche:1 trades.
        Formula: Δ' = 0
        
        Real-world: Valuation error, dead code path, null/missing value, convention inversion.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with sudden zeros
        """
        df = dataset.copy()
        eligible_mask = self._eligible_mask(df)
        trade_type_counts = self._get_trade_type_trade_counts()
        
        # Apply to specified number of trades per trade type
        for trade_type in df[main_column.TRADE_TYPE].unique():
            trade_count = int(trade_type_counts.get(("CD_SuddenZero", trade_type), 0))
            if trade_count == 0:
                continue
            
            # Get all trades of this type with eligible dates
            type_trades = df[(df[main_column.TRADE_TYPE] == trade_type) & eligible_mask][main_column.TRADE].unique()
            if len(type_trades) == 0:
                continue
            
            # Select exactly N trades of this type
            n_to_inject = min(trade_count, len(type_trades))
            selected_trades = np.random.choice(type_trades, size=n_to_inject, replace=False)
            
            # Apply zero injection on 1 random day for each selected trade
            for trade_id in selected_trades:
                trade_data = df[(df[main_column.TRADE] == trade_id) & eligible_mask]
                if len(trade_data) == 0:
                    continue
                
                # Pick 1 random date for this trade
                random_date = np.random.choice(trade_data[main_column.DATE].unique())
                
                mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == random_date)
                mask &= eligible_mask
                
                if mask.sum() > 0:
                    df.loc[mask, cds_column.CREDIT_DELTA_SINGLE] = 0.0
                    df.loc[mask, main_column.RECORD_TYPE] = "CD_SuddenZero"
        
        return df
