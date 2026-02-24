"""
Configuration classes for Credit Delta outlier injectors.

Provides configuration presets for different Credit Delta datasets (CDS, CDI, etc.)
and allows customization of injection parameters.
"""

from typing import Dict, Tuple
from dataclasses import dataclass
from column_names import CreditDeltaSingleColumnSet, CreditDeltaIndexColumnSet


@dataclass(frozen=True)
class ScenarioNames:
    """Injection scenario type names."""
    DRIFT = "Drift"
    STALE_VALUE = "StaleValue"
    CLUSTER_SHOCK_3D = "ClusterShock_3d"
    TRADE_TYPE_WIDE_SHOCK = "TradeTypeWide_Shock"
    POINT_SHOCK = "PointShock"
    SIGN_FLIP = "SignFlip"
    SCALE_ERROR = "ScaleError"
    SUDDEN_ZERO = "SuddenZero"


class CreditDeltaInjectorConfig:
    """
    Configuration for Credit Delta outlier injector.
    
    Contains all numerical parameters needed to customize injection scenarios
    for different datasets or testing requirements.
    """
    
    def __init__(
        self,
        feature_column: str,
        trade_type_counts: Dict[Tuple[str, str], float],
        drift_days_by_type: Dict[str, int],
        stale_days: int,
        scale_factors_by_type: Dict[str, float],
    ):
        """
        Initialize configuration.
        
        Args:
            feature_column: Name of the feature column to inject outliers into
                          (e.g., 'CreditDeltaSingle', 'CreditDeltaIndex')
            # trade_type_days: T (number of days) for each trade type from scenarios spec
            #                Key: trade type (e.g., 'Basis', 'Index')
            #                Value: number of days
            trade_type_counts: Number of trades to inject per scenario/trade type combination
                             Key: (scenario_name, trade_type)
                             Value: number of trades or percentage (float)
            drift_days_by_type: Number of days for CD_Drift scenario by trade type
                              Key: trade type
                              Value: number of days
            stale_days: Number of days for CD_StaleValue scenario (same for all trade types)
            scale_factors_by_type: Scale multipliers for CD_ScaleError by trade type
                                  Key: trade type
                                  Value: scale factor (e.g., 100.0 or 1000.0)
        """
        self.feature_column = feature_column
        self.trade_type_counts = trade_type_counts
        self.drift_days_by_type = drift_days_by_type
        self.stale_days = stale_days
        self.scale_factors_by_type = scale_factors_by_type
    
    @staticmethod
    def cds_preset() -> "CreditDeltaInjectorConfig":
        """
        Preset configuration for Credit Delta Single (CDS) dataset.
        
        Returns:
            Configuration with CDS-specific parameters
        """
        return CreditDeltaInjectorConfig(
            feature_column=CreditDeltaSingleColumnSet.CREDIT_DELTA_SINGLE,
            trade_type_counts={
                (ScenarioNames.DRIFT, "Basis"): 1,
                (ScenarioNames.DRIFT, "Basket"): 1,
                (ScenarioNames.DRIFT, "Index"): 1,
                (ScenarioNames.DRIFT, "SingleName"): 1,
                (ScenarioNames.DRIFT, "Tranche"): 0,
                (ScenarioNames.STALE_VALUE, "Basis"): 1,
                (ScenarioNames.STALE_VALUE, "Basket"): 1,
                (ScenarioNames.STALE_VALUE, "Index"): 1,
                (ScenarioNames.STALE_VALUE, "SingleName"): 1,
                (ScenarioNames.STALE_VALUE, "Tranche"): 1,
                (ScenarioNames.CLUSTER_SHOCK_3D, "Basis"): 1,
                (ScenarioNames.CLUSTER_SHOCK_3D, "Basket"): 1,
                (ScenarioNames.CLUSTER_SHOCK_3D, "Index"): 1,
                (ScenarioNames.CLUSTER_SHOCK_3D, "SingleName"): 1,
                (ScenarioNames.CLUSTER_SHOCK_3D, "Tranche"): 1,
                (ScenarioNames.TRADE_TYPE_WIDE_SHOCK, "Basis"): 0.5,
                (ScenarioNames.TRADE_TYPE_WIDE_SHOCK, "Basket"): 1.0,
                (ScenarioNames.TRADE_TYPE_WIDE_SHOCK, "Index"): 0.5,
                (ScenarioNames.TRADE_TYPE_WIDE_SHOCK, "SingleName"): 0.5,
                (ScenarioNames.TRADE_TYPE_WIDE_SHOCK, "Tranche"): 1.0,
                (ScenarioNames.POINT_SHOCK, "Basis"): 3,
                (ScenarioNames.POINT_SHOCK, "Basket"): 1,
                (ScenarioNames.POINT_SHOCK, "Index"): 2,
                (ScenarioNames.POINT_SHOCK, "SingleName"): 3,
                (ScenarioNames.POINT_SHOCK, "Tranche"): 1,
                (ScenarioNames.SIGN_FLIP, "Basis"): 3,
                (ScenarioNames.SIGN_FLIP, "Basket"): 1,
                (ScenarioNames.SIGN_FLIP, "Index"): 2,
                (ScenarioNames.SIGN_FLIP, "SingleName"): 3,
                (ScenarioNames.SIGN_FLIP, "Tranche"): 1,
                (ScenarioNames.SCALE_ERROR, "Basis"): 3,
                (ScenarioNames.SCALE_ERROR, "Basket"): 1,
                (ScenarioNames.SCALE_ERROR, "Index"): 2,
                (ScenarioNames.SCALE_ERROR, "SingleName"): 3,
                (ScenarioNames.SCALE_ERROR, "Tranche"): 1,
                (ScenarioNames.SUDDEN_ZERO, "Basis"): 3,
                (ScenarioNames.SUDDEN_ZERO, "Basket"): 1,
                (ScenarioNames.SUDDEN_ZERO, "Index"): 2,
                (ScenarioNames.SUDDEN_ZERO, "SingleName"): 3,
                (ScenarioNames.SUDDEN_ZERO, "Tranche"): 1,
            },
            drift_days_by_type={
                "Basis": 15,
                "Basket": 10,
                "Index": 15,
                "SingleName": 15,
                "Tranche": 0,
            },
            stale_days=5,
            scale_factors_by_type={
                "Basis": 100.0,
                "Basket": 100.0,
                "Index": 1000.0,
                "SingleName": 100.0,
                "Tranche": 100.0,
            },
        )
    
    @staticmethod
    def credit_delta_index_preset() -> "CreditDeltaInjectorConfig":
        """
        Preset configuration for Credit Delta Index (CDI) dataset.
        
        Not yet implemented - placeholder for future CDI configuration.
        
        Returns:
            Configuration with CDI-specific parameters            
        
        """
        return CreditDeltaInjectorConfig(
            feature_column=CreditDeltaIndexColumnSet.CREDIT_DELTA_INDEX,
            trade_type_counts={
                (ScenarioNames.DRIFT, "Basis"): 10,
                (ScenarioNames.DRIFT, "Index"): 5,
                (ScenarioNames.STALE_VALUE, "Basis"): 10,
                (ScenarioNames.STALE_VALUE, "Index"): 5,
                (ScenarioNames.POINT_SHOCK, "Basis"): 20,
                (ScenarioNames.POINT_SHOCK, "Index"): 10,
                (ScenarioNames.SIGN_FLIP, "Basis"): 10,
                (ScenarioNames.SIGN_FLIP, "Index"): 5,
                (ScenarioNames.SCALE_ERROR, "Basis"): 20,
                (ScenarioNames.SCALE_ERROR, "Index"): 10,
                (ScenarioNames.SUDDEN_ZERO, "Basis"): 10,
                (ScenarioNames.SUDDEN_ZERO, "Index"): 5,
            },
            
            drift_days_by_type={
                "Basis": 15,
                "Index": 15,
            },
            stale_days=5,
            scale_factors_by_type={
                "Basis": 100.0,
                "Index": 100.0,
            },
        )
