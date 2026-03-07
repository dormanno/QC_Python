"""
PnL Slices outlier injector implementing 7 data quality scenarios.

Implements 7 data quality issue scenarios for PnL Slices datasets (10 slice features).
All features are injected in parallel with identical outliers.
Based on PV injection patterns but with relative formulas for shock scenarios.

Relative formula: value' = value · (1 + s·k·MAD/|median|)
where MAD/|median| is the relative dispersion (coefficient of variation analog).
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from column_names import main_column
from .base import OutlierInjector
from .pnl_slices_config import PnLSlicesInjectorConfig, ScenarioNames


class PnLSlicesOutlierInjector(OutlierInjector):
    """
    PnL Slices outlier injector implementing 7 data quality scenarios.
    
    Injects identical outliers into all 10 PnL slice features in parallel.
    Uses relative (multiplicative) formulas for shock-type scenarios and
    additive formulas for drift/stale scenarios.
    
    ScaleError scenario is intentionally omitted.
    """
    
    def __init__(self, config: PnLSlicesInjectorConfig,
                 severity: float = OutlierInjector.SEVERITY_MEDIUM,
                 random_seed: int = 42):
        """
        Initialize the PnL Slices injector with configuration.
        
        Args:
            config: Configuration object containing all numerical parameters
            severity: Default severity multiplier (k) applied across all injection scenarios
            random_seed: Random seed for reproducibility
        """
        super().__init__(severity=severity, random_seed=random_seed)
        self.config = config
        self.slice_features = list(config.slice_columns)
        # Will store median stats for relative formulas
        self._median_stats: Dict[Tuple[str, str], float] = {}
    
    def inject(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject outliers into the dataset.
        
        Applies all 7 scenarios sequentially to all 10 slice features,
        each respecting eligible OOS rows.
        
        Args:
            dataset: The dataset to inject outliers into
            
        Returns:
            The modified dataset with outliers injected
        """
        df = dataset.copy()
        df = self._ensure_record_type(df)

        # Determine which scenarios have at least one non-zero trade type entry in config
        active_scenarios = {
            scenario
            for (scenario, _), count in self.config.trade_type_counts.items()
            if count > 0
        }

        # Apply configured scenarios sequentially
        scenario_methods = [
            (ScenarioNames.DRIFT,                self.inject_drift),
            (ScenarioNames.STALE_VALUE,           self.inject_stale_value),
            (ScenarioNames.CLUSTER_SHOCK_3D,      self.inject_cluster_shock_3d),
            (ScenarioNames.TRADE_TYPE_WIDE_SHOCK, self.inject_trade_type_wide_shock),
            (ScenarioNames.POINT_SHOCK,           self.inject_point_shock),
            (ScenarioNames.SIGN_FLIP,             self.inject_sign_flip),
            (ScenarioNames.SUDDEN_ZERO,           self.inject_sudden_zero),
        ]

        for scenario_name, method in scenario_methods:
            if scenario_name in active_scenarios:
                df = method(df)

        # Restore any source-protection labels back to OOS — these rows were never modified
        source_mask = df[main_column.RECORD_TYPE] == ScenarioNames.STALE_VALUE_SOURCE
        df.loc[source_mask, main_column.RECORD_TYPE] = "OOS"

        return df

    # ── statistics helpers ──────────────────────────────────────────────────

    def _ensure_stats(self, df: pd.DataFrame):
        """Compute MAD and median stats from train data if not already computed."""
        if not self._mad_stats:
            train_data = self._get_train_data(df)
            self._compute_mad_stats(train_data, self.slice_features)
            self._compute_median_stats(train_data, self.slice_features)

    def _compute_median_stats(self, train_data: pd.DataFrame, features: List[str]):
        """
        Compute median for each feature × TradeType.
        
        Args:
            train_data: Training data (RecordType == "Train")
            features: List of feature column names
        """
        self._median_stats.clear()
        for trade_type in train_data[main_column.TRADE_TYPE].unique():
            type_data = train_data[train_data[main_column.TRADE_TYPE] == trade_type]
            for feature in features:
                if feature in type_data.columns:
                    values = type_data[feature].dropna()
                    if len(values) > 0:
                        self._median_stats[(trade_type, feature)] = values.median()

    def _get_median(self, trade_type: str, feature: str) -> float:
        """Get median for a specific TradeType × feature."""
        return self._median_stats.get((trade_type, feature), 1.0)

    def _get_relative_scale(self, trade_type: str, feature: str) -> float:
        """
        Get relative dispersion scale = MAD / |median|.
        
        Falls back to 1.0 when median is near zero to avoid division issues.
        """
        mad = self._get_mad(trade_type, feature)
        median_abs = abs(self._get_median(trade_type, feature))
        if median_abs < 1e-12:
            return 1.0
        return mad / median_abs
    
    # ========================================================================
    # Scenario 1: Drift (additive, same as PV)
    # ========================================================================
    def inject_drift(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject linear drift in all slice features over last T consecutive days.
        
        Per trade type spec: select 1 trade, apply drift over T days = T records.
        Formula: value'(i) = value(i) + (i/(T-1))·k·MAD_tt for i=0..T-1
        Applied identically to all 10 slice features.
        
        Real-world: Gradual model degradation, stale curve, wrong rates assumption.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with drift injected
        """
        df = dataset.copy()
        self._ensure_stats(df)
        
        drift_days_by_type = self.config.drift_days_by_type
        eligible_mask = self._eligible_mask(df)
        
        for trade_type in df[main_column.TRADE_TYPE].unique():
            T = drift_days_by_type.get(trade_type, 0)
            if T == 0:
                continue
            
            type_trades = df[(df[main_column.TRADE_TYPE] == trade_type) & eligible_mask][main_column.TRADE].unique()
            if len(type_trades) == 0:
                continue
            
            selected_trade = np.random.choice(type_trades)
            
            trade_data = df[df[main_column.TRADE] == selected_trade].sort_values(main_column.DATE)
            eligible_trade = trade_data[eligible_mask[trade_data.index]]
            
            if len(eligible_trade) < T:
                continue
            
            all_eligible_dates = sorted(eligible_trade[main_column.DATE].unique())
            last_dates = all_eligible_dates[-T:]
            
            # Use first slice feature's MAD as reference scale
            mad = self._get_mad(trade_type, self.slice_features[0])
            sign = self._random_sign()
            
            for i, date in enumerate(last_dates):
                mask = (df[main_column.TRADE] == selected_trade) & (df[main_column.DATE] == date)
                mask &= eligible_mask
                
                if mask.sum() > 0:
                    scale_factor = i / (T - 1) if T > 1 else 0
                    drift = sign * scale_factor * self.severity * mad
                    
                    for feature in self.slice_features:
                        df.loc[mask, feature] += drift
                    df.loc[mask, main_column.RECORD_TYPE] = ScenarioNames.DRIFT
        
        return df
    
    # ========================================================================
    # Scenario 2: StaleValue (same logic as PV)
    # ========================================================================
    def inject_stale_value(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject stale/stuck values over first N consecutive days per trade type.
        
        Per trade type spec: select 1 trade, copy value(t-1) for N consecutive days.
        Formula: value'(t) = value(t-1) (copy previous day's value)
        Applied identically to all 10 slice features.
        
        Real-world: Valuations not updating, feed stuck, source frozen.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with stale values
        """
        df = dataset.copy()
        eligible_mask = self._eligible_mask(df)
        stale_days = self.config.stale_days
        
        for trade_type in df[main_column.TRADE_TYPE].unique():
            type_trades = df[(df[main_column.TRADE_TYPE] == trade_type) & eligible_mask][main_column.TRADE].unique()
            if len(type_trades) == 0:
                continue
            
            selected_trade = np.random.choice(type_trades)
            
            trade_data = df[df[main_column.TRADE] == selected_trade].sort_values(main_column.DATE)
            eligible_trade = trade_data[eligible_mask[trade_data.index]]
            
            if len(eligible_trade) < stale_days + 1:
                continue
            
            all_eligible_dates = sorted(eligible_trade[main_column.DATE].unique())
            target_dates = all_eligible_dates[1:stale_days + 1]
            
            # Use first eligible date as source value; protect it from subsequent scenarios
            source_mask = (df[main_column.TRADE] == selected_trade) & \
                          (df[main_column.DATE] == all_eligible_dates[0])
            
            stale_values = {}
            for feature in self.slice_features:
                stale_values[feature] = df.loc[source_mask, feature].iloc[0]
            df.loc[source_mask, main_column.RECORD_TYPE] = ScenarioNames.STALE_VALUE_SOURCE
            
            for date in target_dates:
                mask = (df[main_column.TRADE] == selected_trade) & (df[main_column.DATE] == date)
                mask &= eligible_mask
                
                if mask.sum() > 0:
                    for feature in self.slice_features:
                        df.loc[mask, feature] = stale_values[feature]
                    df.loc[mask, main_column.RECORD_TYPE] = ScenarioNames.STALE_VALUE
        
        return df
    
    # ========================================================================
    # Scenario 3: ClusterShock_3d (relative formula)
    # ========================================================================
    def inject_cluster_shock_3d(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject 3-day cluster shock to all slice features using relative formula.
        
        Per trade type spec: select 1 trade, apply shock over 3 consecutive days.
        Relative formula: value'(t) = value(t) · (1 + s·k·MAD/|median|)
        Applied identically to all 10 slice features.
        
        Real-world: Market event/shock spanning multiple days.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with cluster shocks
        """
        df = dataset.copy()
        self._ensure_stats(df)
        
        eligible_mask = self._eligible_mask(df)
        shock_days = 3
        
        for trade_type in df[main_column.TRADE_TYPE].unique():
            type_trades = df[(df[main_column.TRADE_TYPE] == trade_type) & eligible_mask][main_column.TRADE].unique()
            if len(type_trades) == 0:
                continue
            
            selected_trade = np.random.choice(type_trades)
            
            trade_data = df[df[main_column.TRADE] == selected_trade].sort_values(main_column.DATE)
            eligible_trade = trade_data[eligible_mask[trade_data.index]]
            
            if len(eligible_trade) < shock_days:
                continue
            
            all_eligible_dates = sorted(eligible_trade[main_column.DATE].unique())
            if len(all_eligible_dates) < shock_days:
                continue
            
            start_idx = np.random.randint(0, len(all_eligible_dates) - shock_days + 1)
            shock_dates = all_eligible_dates[start_idx:start_idx + shock_days]
            
            sign = self._random_sign()
            
            for date in shock_dates:
                mask = (df[main_column.TRADE] == selected_trade) & (df[main_column.DATE] == date)
                mask &= eligible_mask
                
                if mask.sum() > 0:
                    for feature in self.slice_features:
                        rel_scale = self._get_relative_scale(trade_type, feature)
                        multiplier = 1.0 + sign * self.severity * rel_scale
                        df.loc[mask, feature] *= multiplier
                    df.loc[mask, main_column.RECORD_TYPE] = ScenarioNames.CLUSTER_SHOCK_3D
        
        return df
    
    # ========================================================================
    # Scenario 4: TradeTypeWide_Shock (relative formula)
    # ========================================================================
    def inject_trade_type_wide_shock(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject systemic shock on first OOS date across % of each trade type.
        
        Uses relative formula: value'(t) = value(t) · (1 + s·k·MAD/|median|)
        Applied identically to all 10 slice features.
        
        Real-world: Market-wide repricing event affecting risk category.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with trade-type-wide shocks
        """
        df = dataset.copy()
        self._ensure_stats(df)
        
        eligible_mask = self._eligible_mask(df)
        trade_type_counts = self.config.trade_type_counts
        
        # Find first OOS date
        oos_mask = df[main_column.RECORD_TYPE] != "Train"
        oos_dates = sorted(df[oos_mask][main_column.DATE].unique())
        if len(oos_dates) == 0:
            return df
        
        first_oos_date = oos_dates[0]
        
        for trade_type in df[main_column.TRADE_TYPE].unique():
            pct = trade_type_counts.get((ScenarioNames.TRADE_TYPE_WIDE_SHOCK, trade_type), 0)
            if pct == 0:
                continue
            
            first_date_mask = (df[main_column.DATE] == first_oos_date) & \
                            (df[main_column.TRADE_TYPE] == trade_type) & \
                            oos_mask
            
            trades_on_first_date = df[first_date_mask][main_column.TRADE].unique()
            if len(trades_on_first_date) == 0:
                continue
            
            target_count = max(1, int(len(trades_on_first_date) * pct))
            shuffled_trades = np.random.permutation(trades_on_first_date)
            
            sign = self._random_sign()
            
            injected_count = 0
            for trade_id in shuffled_trades:
                if injected_count >= target_count:
                    break
                
                mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == first_oos_date)
                mask &= eligible_mask
                
                if mask.sum() > 0:
                    for feature in self.slice_features:
                        rel_scale = self._get_relative_scale(trade_type, feature)
                        multiplier = 1.0 + sign * self.severity * rel_scale
                        df.loc[mask, feature] *= multiplier
                    df.loc[mask, main_column.RECORD_TYPE] = ScenarioNames.TRADE_TYPE_WIDE_SHOCK
                    injected_count += 1
                else:
                    # Try any eligible date for this trade
                    trade_mask = (df[main_column.TRADE] == trade_id) & eligible_mask
                    if trade_mask.sum() > 0:
                        trade_rows = df[trade_mask].iloc[:1]
                        idx = trade_rows.index[0]
                        for feature in self.slice_features:
                            rel_scale = self._get_relative_scale(trade_type, feature)
                            multiplier = 1.0 + sign * self.severity * rel_scale
                            df.loc[idx, feature] *= multiplier
                        df.loc[idx, main_column.RECORD_TYPE] = ScenarioNames.TRADE_TYPE_WIDE_SHOCK
                        injected_count += 1
        
        return df
    
    # ========================================================================
    # Scenario 5: PointShock (relative formula)
    # ========================================================================
    def inject_point_shock(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject isolated 1-day shocks for specified trades per type.
        
        Relative formula: value'(t) = value(t) · (1 + s·k·MAD/|median|)
        Applied identically to all 10 slice features.
        
        Real-world: One-off bad market data snapshot, transient calc glitch.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with point shocks
        """
        df = dataset.copy()
        self._ensure_stats(df)
        
        eligible_mask = self._eligible_mask(df)
        trade_type_counts = self.config.trade_type_counts
        
        for trade_type in df[main_column.TRADE_TYPE].unique():
            trade_count = int(trade_type_counts.get((ScenarioNames.POINT_SHOCK, trade_type), 0))
            if trade_count == 0:
                continue
            
            type_trades = df[(df[main_column.TRADE_TYPE] == trade_type) & eligible_mask][main_column.TRADE].unique()
            if len(type_trades) == 0:
                continue
            
            n_to_inject = min(trade_count, len(type_trades))
            selected_trades = np.random.choice(type_trades, size=n_to_inject, replace=False)
            
            sign = self._random_sign()
            
            for trade_id in selected_trades:
                trade_data = df[(df[main_column.TRADE] == trade_id) & eligible_mask]
                if len(trade_data) == 0:
                    continue
                
                random_date = np.random.choice(trade_data[main_column.DATE].unique())
                
                mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == random_date)
                mask &= eligible_mask
                
                if mask.sum() > 0:
                    for feature in self.slice_features:
                        rel_scale = self._get_relative_scale(trade_type, feature)
                        multiplier = 1.0 + sign * self.severity * rel_scale
                        df.loc[mask, feature] *= multiplier
                    df.loc[mask, main_column.RECORD_TYPE] = ScenarioNames.POINT_SHOCK
        
        return df
    
    # ========================================================================
    # Scenario 6: SignFlip (same logic as PV)
    # ========================================================================
    def inject_sign_flip(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject sign flip for specified trades per type on 1 random day.
        
        Formula: value' = -value
        Applied identically to all 10 slice features.
        
        Real-world: Sign convention inversion, wrong direction, mapping inversion bug.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with sign flips
        """
        df = dataset.copy()
        eligible_mask = self._eligible_mask(df)
        trade_type_counts = self.config.trade_type_counts
        
        for trade_type in df[main_column.TRADE_TYPE].unique():
            trade_count = int(trade_type_counts.get((ScenarioNames.SIGN_FLIP, trade_type), 0))
            if trade_count == 0:
                continue
            
            type_trades = df[(df[main_column.TRADE_TYPE] == trade_type) & eligible_mask][main_column.TRADE].unique()
            if len(type_trades) == 0:
                continue
            
            n_to_inject = min(trade_count, len(type_trades))
            selected_trades = np.random.choice(type_trades, size=n_to_inject, replace=False)
            
            for trade_id in selected_trades:
                trade_data = df[(df[main_column.TRADE] == trade_id) & eligible_mask]
                if len(trade_data) == 0:
                    continue
                
                random_date = np.random.choice(trade_data[main_column.DATE].unique())
                
                mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == random_date)
                mask &= eligible_mask
                
                if mask.sum() > 0:
                    for feature in self.slice_features:
                        df.loc[mask, feature] *= -1
                    df.loc[mask, main_column.RECORD_TYPE] = ScenarioNames.SIGN_FLIP
        
        return df
    
    # ========================================================================
    # Scenario 7: SuddenZero (same logic as PV)
    # ========================================================================
    def inject_sudden_zero(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Inject sudden zero values for specified trades per type on 1 random day.
        
        Formula: value' = 0
        Applied identically to all 10 slice features.
        
        Real-world: Valuation error, dead code path, null/missing value.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with sudden zeros
        """
        df = dataset.copy()
        eligible_mask = self._eligible_mask(df)
        trade_type_counts = self.config.trade_type_counts
        
        for trade_type in df[main_column.TRADE_TYPE].unique():
            trade_count = int(trade_type_counts.get((ScenarioNames.SUDDEN_ZERO, trade_type), 0))
            if trade_count == 0:
                continue
            
            type_trades = df[(df[main_column.TRADE_TYPE] == trade_type) & eligible_mask][main_column.TRADE].unique()
            if len(type_trades) == 0:
                continue
            
            n_to_inject = min(trade_count, len(type_trades))
            selected_trades = np.random.choice(type_trades, size=n_to_inject, replace=False)
            
            for trade_id in selected_trades:
                trade_data = df[(df[main_column.TRADE] == trade_id) & eligible_mask]
                if len(trade_data) == 0:
                    continue
                
                random_date = np.random.choice(trade_data[main_column.DATE].unique())
                
                mask = (df[main_column.TRADE] == trade_id) & (df[main_column.DATE] == random_date)
                mask &= eligible_mask
                
                if mask.sum() > 0:
                    for feature in self.slice_features:
                        df.loc[mask, feature] = 0.0
                    df.loc[mask, main_column.RECORD_TYPE] = ScenarioNames.SUDDEN_ZERO
        
        return df
