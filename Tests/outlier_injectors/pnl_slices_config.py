"""
Configuration classes for PnL Slices outlier injectors.

Provides configuration presets for PnL Slices datasets (10 slice features)
and allows customization of injection parameters.
"""

from typing import Dict, Tuple
from dataclasses import dataclass
from column_names import PnLSlicesColumnSet


@dataclass(frozen=True)
class ScenarioNames:
    """Injection scenario type names."""
    DRIFT = "Drift"
    STALE_VALUE = "StaleValue"
    STALE_VALUE_SOURCE = "StaleValue_Source"
    CLUSTER_SHOCK_3D = "ClusterShock_3d"
    TRADE_TYPE_WIDE_SHOCK = "TradeTypeWide_Shock"
    POINT_SHOCK = "PointShock"
    SIGN_FLIP = "SignFlip"
    SUDDEN_ZERO = "SuddenZero"


class PnLSlicesInjectorConfig:
    """
    Configuration for PnL Slices outlier injector.
    
    Contains all numerical parameters needed to customize injection scenarios
    for PnL Slices datasets (10 slice features).
    """
    
    def __init__(
        self,
        slice_columns: list,
        trade_type_counts: Dict[Tuple[str, str], float],
        drift_days_by_type: Dict[str, int],
        stale_days: int,
    ):
        """
        Initialize configuration.
        
        Args:
            slice_columns: List of slice feature column names
            trade_type_counts: Number of trades to inject per scenario/trade type combination
                             Key: (scenario_name, trade_type)
                             Value: number of trades or percentage (float)
            drift_days_by_type: Number of days for Drift scenario by trade type
                              Key: trade type
                              Value: number of days
            stale_days: Number of days for StaleValue scenario (same for all trade types)
        """
        self.slice_columns = slice_columns
        self.trade_type_counts = trade_type_counts
        self.drift_days_by_type = drift_days_by_type
        self.stale_days = stale_days
    
    @staticmethod
    def pnl_slices_preset() -> "PnLSlicesInjectorConfig":
        """
        Preset configuration for PnL Slices dataset.
        
        Returns:
            Configuration with PnL Slices-specific parameters
        """
        col = PnLSlicesColumnSet()
        return PnLSlicesInjectorConfig(
            slice_columns=col.SLICE_COLUMNS,
            trade_type_counts={
                (ScenarioNames.DRIFT, "Basis"): 3,
                (ScenarioNames.DRIFT, "Basket"): 1,
                (ScenarioNames.DRIFT, "Index"): 2,
                (ScenarioNames.DRIFT, "SingleName"): 3,
                (ScenarioNames.DRIFT, "Tranche"): 1,
                (ScenarioNames.STALE_VALUE, "Basis"): 3,
                (ScenarioNames.STALE_VALUE, "Basket"): 1,
                (ScenarioNames.STALE_VALUE, "Index"): 2,
                (ScenarioNames.STALE_VALUE, "SingleName"): 3,
                (ScenarioNames.STALE_VALUE, "Tranche"): 1,
                (ScenarioNames.CLUSTER_SHOCK_3D, "Basis"): 3,
                (ScenarioNames.CLUSTER_SHOCK_3D, "Basket"): 1,
                (ScenarioNames.CLUSTER_SHOCK_3D, "Index"): 2,
                (ScenarioNames.CLUSTER_SHOCK_3D, "SingleName"): 3,
                (ScenarioNames.CLUSTER_SHOCK_3D, "Tranche"): 1,
                (ScenarioNames.TRADE_TYPE_WIDE_SHOCK, "Basis"): 0.5,
                (ScenarioNames.TRADE_TYPE_WIDE_SHOCK, "Basket"): 1.0,
                (ScenarioNames.TRADE_TYPE_WIDE_SHOCK, "Index"): 0.5,
                (ScenarioNames.TRADE_TYPE_WIDE_SHOCK, "SingleName"): 0.5,
                (ScenarioNames.TRADE_TYPE_WIDE_SHOCK, "Tranche"): 1.0,
                (ScenarioNames.POINT_SHOCK, "Basis"): 8,
                (ScenarioNames.POINT_SHOCK, "Basket"): 3,
                (ScenarioNames.POINT_SHOCK, "Index"): 6,
                (ScenarioNames.POINT_SHOCK, "SingleName"): 8,
                (ScenarioNames.POINT_SHOCK, "Tranche"): 3,
                (ScenarioNames.SIGN_FLIP, "Basis"): 5,
                (ScenarioNames.SIGN_FLIP, "Basket"): 2,
                (ScenarioNames.SIGN_FLIP, "Index"): 3,
                (ScenarioNames.SIGN_FLIP, "SingleName"): 5,
                (ScenarioNames.SIGN_FLIP, "Tranche"): 2,
                (ScenarioNames.SUDDEN_ZERO, "Basis"): 6,
                (ScenarioNames.SUDDEN_ZERO, "Basket"): 2,
                (ScenarioNames.SUDDEN_ZERO, "Index"): 4,
                (ScenarioNames.SUDDEN_ZERO, "SingleName"): 6,
                (ScenarioNames.SUDDEN_ZERO, "Tranche"): 2,
            },
            drift_days_by_type={
                "Basis": 15,
                "Basket": 10,
                "Index": 15,
                "SingleName": 15,
                "Tranche": 10,
            },
            stale_days=5,
        )
