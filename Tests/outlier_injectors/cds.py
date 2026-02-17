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
    
    def _get_trade_type_shock_percentage(self) -> Dict[str, float]:
        """Get shock percentage for CD_TradeTypeWideShock scenario."""
        return {
            "Basis": 0.025,      # 2.5% of Basis trades
            "Basket": 0.03,      # 3% of Basket trades
            "Index": 0.02,       # 2% of Index trades
            "SingleName": 0.04,  # 4% of SingleName trades
            "Tranche": 0.035,    # 3.5% of Tranche trades
        }
    
    # ========================================================================
    # Scenario 1: CD_Drift
    # ========================================================================
    def inject_cd_drift(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject linear drift in Credit Delta over last T consecutive days.
        
        Parameters T (# days) vary by trade type (Basis:20, Basket:15, Index:10, etc).
        Drift formula: Δ'(d) = Δ(d) + α·d·Range, d ∈ [1,T], Range = max(Δ) - min(Δ)
        
        Real-world: Gradual model degradation, stale curve, wrong rates assumption.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with drift injected
        """
        df = dataset.copy()
        
        if not self._iqr_stats:
            train_data = self._get_train_data(df)
            self._compute_iqr_stats(train_data, [cds_column.CREDIT_DELTA_SINGLE])
        
        # Trade-type specific parameters
        days_by_type = self._get_trade_type_days()
        
        eligible_mask = self._eligible_mask(df)
        trade_ids = df[df[main_column.RECORD_TYPE] != "Train"][main_column.TRADE].unique()
        
        for trade_id in trade_ids:
            trade_data = df[df[main_column.TRADE] == trade_id].sort_values(main_column.DATE)
            eligible_trade = trade_data[eligible_mask[trade_data.index]]
            
            if len(eligible_trade) < 1:
                continue
            
            trade_type = trade_data[main_column.TRADE_TYPE].iloc[0]
            T = days_by_type.get(trade_type, 20)
            
            # Get last T dates
            last_dates = sorted(eligible_trade[main_column.DATE].unique())[-T:]
            
            # Compute range from IQR stats
            iqr = self._get_iqr(trade_type, cds_column.CREDIT_DELTA_SINGLE)
            range_val = iqr  # Range estimate from IQR
            
            alpha = 0.2 / T  # Scale factor for realistic drift
            
            for d, date in enumerate(last_dates, start=1):
                mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == date)
                mask &= eligible_mask
                
                if mask.sum() > 0:
                    sign = self._random_sign()
                    drift = sign * alpha * d * range_val
                    df.loc[mask, cds_column.CREDIT_DELTA_SINGLE] += drift
                    df.loc[mask, main_column.RECORD_TYPE] = "CD_Drift"
        
        return df
    
    # ========================================================================
    # Scenario 2: CD_StaleValue
    # ========================================================================
    def inject_cd_stale_value(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject stale/stuck values in Credit Delta over first consecutive days.
        
        Affected period is first consecutive days in OOS data per trade.
        Formula: Δ'(t) = Δ(t-1) (copy previous day's value)
        
        Real-world: Valuations not updating, feed stuck, source frozen.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with stale values
        """
        df = dataset.copy()
        eligible_mask = self._eligible_mask(df)
        trade_ids = df[df[main_column.RECORD_TYPE] != "Train"][main_column.TRADE].unique()
        
        for trade_id in trade_ids:
            trade_data = df[df[main_column.TRADE] == trade_id].sort_values(main_column.DATE)
            eligible_trade = trade_data[eligible_mask[trade_data.index]]
            
            if len(eligible_trade) < 3:
                continue
            
            # Select random number of first consecutive stale days (3-10)
            n_stale = np.random.randint(3, min(11, len(eligible_trade)))
            eligible_dates = sorted(eligible_trade[main_column.DATE].unique())
            stale_dates = eligible_dates[:n_stale]
            
            # Get the value from before the stale period
            stale_value = df[(df[main_column.TRADE] == trade_id) & 
                            (df[main_column.DATE] == stale_dates[0])][cds_column.CREDIT_DELTA_SINGLE].iloc[0]
            
            for date in stale_dates:
                mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == date)
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
        Inject 3-day cluster shock in Credit Delta.
        
        Randomly select 3 consecutive days in OOS range per trade.
        Formula: Δ'(t) = Δ(t) + s·k·IQR_seg(Δ), for t ∈ [shock_start, shock_start+2]
        
        Real-world: Market event/shock spanning multiple days, rate spike, credit event.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with cluster shocks
        """
        df = dataset.copy()
        
        if not self._iqr_stats:
            train_data = self._get_train_data(df)
            self._compute_iqr_stats(train_data, [cds_column.CREDIT_DELTA_SINGLE])
        
        eligible_mask = self._eligible_mask(df)
        trade_ids = df[df[main_column.RECORD_TYPE] != "Train"][main_column.TRADE].unique()
        
        for trade_id in trade_ids:
            trade_data = df[df[main_column.TRADE] == trade_id].sort_values(main_column.DATE)
            eligible_trade = trade_data[eligible_mask[trade_data.index]]
            
            if len(eligible_trade) < 3:
                continue
            
            # Select random 3 consecutive dates
            eligible_dates = sorted(eligible_trade[main_column.DATE].unique())
            if len(eligible_dates) < 3:
                continue
            
            start_idx = np.random.randint(0, len(eligible_dates) - 2)
            shock_dates = eligible_dates[start_idx:start_idx + 3]
            
            trade_type = trade_data[main_column.TRADE_TYPE].iloc[0]
            severity = self.SEVERITY_MEDIUM
            iqr = self._get_iqr(trade_type, cds_column.CREDIT_DELTA_SINGLE)
            sign = self._random_sign()
            shock = sign * severity * iqr
            
            for date in shock_dates:
                mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == date)
                mask &= eligible_mask
                
                if mask.sum() > 0:
                    df.loc[mask, cds_column.CREDIT_DELTA_SINGLE] += shock
                    df.loc[mask, main_column.RECORD_TYPE] = "CD_ClusterShock_3d"
        
        return df
    
    # ========================================================================
    # Scenario 4: CD_TradeTypeWideShock
    # ========================================================================
    def inject_cd_trade_type_wide_shock(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject systemic shock across percentage of each trade type (1 day).
        
        Shock percentage varies by trade type:
        - SingleName: 4%, Basket: 3%, Tranche: 3.5%, Basis: 2.5%, Index: 2%
        
        Formula: Δ' = Δ + s·k·IQR_seg(Δ), applied to selected % of trades of type
        
        Real-world: Market-wide repricing event affecting risk category, credit curve shift.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with trade-type-wide shocks
        """
        df = dataset.copy()
        
        if not self._iqr_stats:
            train_data = self._get_train_data(df)
            self._compute_iqr_stats(train_data, [cds_column.CREDIT_DELTA_SINGLE])
        
        eligible_mask = self._eligible_mask(df)
        shock_pct = self._get_trade_type_shock_percentage()
        severity = self.SEVERITY_MEDIUM
        
        # Pick a single date for the shock
        shock_date_candidates = df[eligible_mask][main_column.DATE].unique()
        if len(shock_date_candidates) == 0:
            return df
        
        shock_date = np.random.choice(shock_date_candidates)
        
        # For each trade type, select percentage of trades and shock them on shock_date
        for trade_type in df[main_column.TRADE_TYPE].unique():
            pct = shock_pct.get(trade_type, 0.03)  # Default 3%
            
            # Get trades of this type
            type_trades = df[df[main_column.TRADE_TYPE] == trade_type][main_column.TRADE].unique()
            n_to_shock = max(1, int(len(type_trades) * pct))
            
            shocked_trades = np.random.choice(type_trades, size=n_to_shock, replace=False)
            
            iqr = self._get_iqr(trade_type, cds_column.CREDIT_DELTA_SINGLE)
            sign = self._random_sign()
            shock = sign * severity * iqr
            
            for trade_id in shocked_trades:
                mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == shock_date)
                mask &= eligible_mask
                
                if mask.sum() > 0:
                    df.loc[mask, cds_column.CREDIT_DELTA_SINGLE] += shock
                    df.loc[mask, main_column.RECORD_TYPE] = "CD_TradeTypeWide_Shock"
        
        return df
    
    # ========================================================================
    # Scenario 5: CD_PointShock
    # ========================================================================
    def inject_cd_point_shock(self, dataset: pd.DataFrame, trade_count: int = 10) -> pd.DataFrame:
        """
        Inject isolated 1-day shocks in Credit Delta for random trades.
        
        Formula: Δ' = Δ + s·k·IQR_seg(Δ)
        
        Real-world: One-off bad market data snapshot, transient calc glitch, data feed artifact.
        
        Args:
            dataset: Input dataset
            trade_count: Number of trades to shock
            
        Returns:
            Dataset with point shocks
        """
        df = dataset.copy()
        
        if not self._iqr_stats:
            train_data = self._get_train_data(df)
            self._compute_iqr_stats(train_data, [cds_column.CREDIT_DELTA_SINGLE])
        
        selected_trades = self._select_random_trades(df, trade_count)
        eligible_mask = self._eligible_mask(df)
        severity = self.SEVERITY_MEDIUM
        
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
                iqr = self._get_iqr(trade_type, cds_column.CREDIT_DELTA_SINGLE)
                sign = self._random_sign()
                
                df.loc[mask, cds_column.CREDIT_DELTA_SINGLE] += sign * severity * iqr
                df.loc[mask, main_column.RECORD_TYPE] = "CD_PointShock"
        
        return df
    
    # ========================================================================
    # Scenario 6: CD_SignFlip
    # ========================================================================
    def inject_cd_sign_flip(self, dataset: pd.DataFrame, trade_count: int = 5) -> pd.DataFrame:
        """
        Inject sign flip in Credit Delta for random trades (1 day).
        
        Formula: Δ' = -Δ
        
        Real-world: Sign convention inversion, wrong direction, mapping inversion bug.
        
        Args:
            dataset: Input dataset
            trade_count: Number of trades to inject
            
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
            
            if mask.sum() > 0:
                df.loc[mask, cds_column.CREDIT_DELTA_SINGLE] *= -1
                df.loc[mask, main_column.RECORD_TYPE] = "CD_SignFlip"
        
        return df
    
    # ========================================================================
    # Scenario 7: CD_ScaleError
    # ========================================================================
    def inject_cd_scale_error(self, dataset: pd.DataFrame, trade_count: int = 5) -> pd.DataFrame:
        """
        Inject scale/unit error in Credit Delta (1 day).
        
        Trade-type specific scaling:
        - Index: 1000× or /1000 (basis point vs percentage confusion)
        - Others: 100× or /100 (currency/notional scaling error)
        
        Real-world: Unit/currency/notional scaling error, decimal placement bug.
        
        Args:
            dataset: Input dataset
            trade_count: Number of trades to inject
            
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
            
            if mask.sum() > 0:
                trade_type = df.loc[mask, main_column.TRADE_TYPE].iloc[0]
                
                # Index uses 1000×, others use 100×
                scale_factor = 1000.0 if trade_type == "Index" else 100.0
                factor = scale_factor if np.random.random() > 0.5 else 1.0 / scale_factor
                
                df.loc[mask, cds_column.CREDIT_DELTA_SINGLE] *= factor
                df.loc[mask, main_column.RECORD_TYPE] = "CD_ScaleError"
        
        return df
    
    # ========================================================================
    # Scenario 8: CD_SuddenZero
    # ========================================================================
    def inject_cd_sudden_zero(self, dataset: pd.DataFrame, trade_count: int = 5) -> pd.DataFrame:
        """
        Inject sudden zero values in Credit Delta (1 day).
        
        Formula: Δ' = 0
        
        Real-world: Valuation error, dead code path, null/missing value, convention inversion.
        
        Args:
            dataset: Input dataset
            trade_count: Number of trades to inject
            
        Returns:
            Dataset with sudden zeros
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
            
            if mask.sum() > 0:
                df.loc[mask, cds_column.CREDIT_DELTA_SINGLE] = 0.0
                df.loc[mask, main_column.RECORD_TYPE] = "CD_SuddenZero"
        
        return df
