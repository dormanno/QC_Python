"""
Configuration classes for PV (Present Value) outlier injectors.

Provides configuration presets for PV datasets and allows customization
of injection parameters.
"""

from typing import Dict, Tuple
from dataclasses import dataclass
from column_names import PVColumnSet


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
    SCALE_ERROR = "ScaleError"
    SUDDEN_ZERO = "SuddenZero"


class PVInjectorConfig:
    """
    Configuration for PV outlier injector.
    
    Contains all numerical parameters needed to customize injection scenarios
    for PV datasets (Start_PV and End_PV features).
    """
    
    def __init__(
        self,
        start_pv_column: str,
        end_pv_column: str,
        trade_type_counts: Dict[Tuple[str, str], float],
        drift_days_by_type: Dict[str, int],
        stale_days: int,
        scale_factors_by_type: Dict[str, float],
    ):
        """
        Initialize configuration.
        
        Args:
            start_pv_column: Name of the Start_PV feature column
            end_pv_column: Name of the End_PV feature column
            trade_type_counts: Number of trades to inject per scenario/trade type combination
                             Key: (scenario_name, trade_type)
                             Value: number of trades or percentage (float)
            drift_days_by_type: Number of days for PV_Drift scenario by trade type
                              Key: trade type
                              Value: number of days
            stale_days: Number of days for PV_StaleValue scenario (same for all trade types)
            scale_factors_by_type: Scale multipliers for PV_ScaleError by trade type
                                  Key: trade type
                                  Value: scale factor (e.g., 100.0 or 1000.0)
        """
        self.start_pv_column = start_pv_column
        self.end_pv_column = end_pv_column
        self.trade_type_counts = trade_type_counts
        self.drift_days_by_type = drift_days_by_type
        self.stale_days = stale_days
        self.scale_factors_by_type = scale_factors_by_type
    
    @staticmethod
    def pv_preset() -> "PVInjectorConfig":
        """
        Preset configuration for PV dataset.
        
        Returns:
            Configuration with PV-specific parameters based on CDS preset
        """
        return PVInjectorConfig(
            start_pv_column=PVColumnSet.START_PV,
            end_pv_column=PVColumnSet.END_PV,
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
                (ScenarioNames.SCALE_ERROR, "Basis"): 3,
                (ScenarioNames.SCALE_ERROR, "Basket"): 1,
                (ScenarioNames.SCALE_ERROR, "Index"): 2,
                (ScenarioNames.SCALE_ERROR, "SingleName"): 3,
                (ScenarioNames.SCALE_ERROR, "Tranche"): 1,
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
            scale_factors_by_type={
                "Basis": 100.0,
                "Basket": 100.0,
                "Index": 100.0,
                "SingleName": 100.0,
                "Tranche": 100.0,
            },
        )
