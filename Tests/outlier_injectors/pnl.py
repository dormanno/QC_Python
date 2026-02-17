"""
PnL-specific outlier injector implementing 10 data quality scenarios.

Implements various data quality issue scenarios based on real-world problems
in PnL reporting systems.
"""

import pandas as pd
import numpy as np
from typing import Optional
from column_names import pnl_column, main_column
from .base import OutlierInjector


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
    def inject_pv_eod_spike_point(self, dataset: pd.DataFrame, severity: float = None,
                                  trade_count: int = None) -> pd.DataFrame:
        """
        Inject single-day spikes in End_PV for 1-3% of trades.
        
        Formula: End_PV' = End_PV + s*k*MAD_seg(End_PV)
        
        Real-world: One-off bad market data snapshot, transient calc glitch.
        
        Args:
            dataset: Input dataset
            severity: Severity multiplier (k)
            trade_count: Number of trades to inject
            
        Returns:
            Dataset with injected spikes
        """
        if severity is None:
            severity = self.SEVERITY_MEDIUM
            
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
    def inject_pv_eod_step_change(self, dataset: pd.DataFrame, severity: float = None,
                                  trade_count: int = None) -> pd.DataFrame:
        """
        Inject persistent step change in End_PV for 0.5-2% of trades, 10+ days.
        
        Formula: For t≥t₀: End_PV' = End_PV + s*k*MAD_seg(End_PV)
        
        Real-world: Persistent mapping/config/model change affecting subset of trades.
        
        Args:
            dataset: Input dataset
            severity: Severity multiplier (k)
            trade_count: Number of trades to inject
            
        Returns:
            Dataset with injected step changes
        """
        if severity is None:
            severity = self.SEVERITY_MEDIUM
            
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
            trade_count: Number of trades to inject
            
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
            df.loc[mask, pnl_column.END] *= -1
            df.loc[mask, main_column.RECORD_TYPE] = "Injected_PV_SignFlip"
        
        return df
    
    # ========================================================================
    # Scenario 6: Slice_CreditSingle_Spike_Point
    # ========================================================================
    def inject_slice_credit_single_spike(self, dataset: pd.DataFrame, severity: float = None,
                                         trade_count: int = None) -> pd.DataFrame:
        """
        Inject single-day spike in Credit_Single_PnL for 1-3% of trades.
        
        Formula: Credit_Single_PnL' = Credit_Single_PnL + s*k*MAD_seg(Credit_Single_PnL)
        
        Real-world: Component PnL spike due to factor mispricing, wrong bucket.
        
        Args:
            dataset: Input dataset
            severity: Severity multiplier (k)
            trade_count: Number of trades to inject
            
        Returns:
            Dataset with injected spikes
        """
        if severity is None:
            severity = self.SEVERITY_MEDIUM
            
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
            trade_count: Number of trades to inject
            
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
    def inject_slice_reallocation_bug(self, dataset: pd.DataFrame, severity: float = None,
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
            trade_count: Number of trades to inject
            
        Returns:
            Dataset with reallocation bugs
        """
        if severity is None:
            severity = self.SEVERITY_MEDIUM
            
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
    def inject_total_pnl_identity_break(self, dataset: pd.DataFrame, severity: float = None,
                                        trade_count: int = None) -> pd.DataFrame:
        """
        Break reconciliation by changing Credit_Single_PnL without updating Total_PnL.
        
        Formula: Credit_Single_PnL' = Credit_Single_PnL + Δ, but keep Total_PnL unchanged
        
        Real-world: Broken reconciliation/aggregation, stale totals, downstream join failure.
        
        Args:
            dataset: Input dataset
            severity: Severity multiplier (k)
            trade_count: Number of trades to inject
            
        Returns:
            Dataset with identity breaks
        """
        if severity is None:
            severity = self.SEVERITY_MEDIUM
            
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
    def inject_cross_family_inconsistency(self, dataset: pd.DataFrame, severity: float = None,
                                          trade_count: int = None) -> pd.DataFrame:
        """
        Create inconsistency between End_PV and Total_PnL.
        
        Formula: End_PV' = End_PV + s*k*MAD_seg(End_PV), Total_PnL' = Total_PnL + ε (small noise)
        
        Real-world: PV feed/date alignment issue - PV changes but PnL pipeline doesn't match.
        
        Args:
            dataset: Input dataset
            severity: Severity multiplier (k)
            trade_count: Number of trades to inject
            
        Returns:
            Dataset with cross-family inconsistencies
        """
        if severity is None:
            severity = self.SEVERITY_MEDIUM
            
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
