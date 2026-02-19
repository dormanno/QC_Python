"""
Configuration classes for Credit Delta outlier injectors.

Provides configuration presets for different Credit Delta datasets (CDS, CDI, etc.)
and allows customization of injection parameters.
"""

from typing import Dict, Tuple
from column_names import CreditDeltaSingleColumnSet, CreditDeltaIndexColumnSet


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
        """
        self.feature_column = feature_column
        self.trade_type_counts = trade_type_counts
        self.drift_days_by_type = drift_days_by_type
        self.stale_days = stale_days
    
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
                ("CD_Drift", "Basis"): 1,
                ("CD_Drift", "Basket"): 1,
                ("CD_Drift", "Index"): 1,
                ("CD_Drift", "SingleName"): 1,
                ("CD_Drift", "Tranche"): 0,
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
                ("CD_TradeTypeWide_Shock", "Basis"): 0.5,
                ("CD_TradeTypeWide_Shock", "Basket"): 1.0,
                ("CD_TradeTypeWide_Shock", "Index"): 0.5,
                ("CD_TradeTypeWide_Shock", "SingleName"): 0.5,
                ("CD_TradeTypeWide_Shock", "Tranche"): 1.0,
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
            },
            drift_days_by_type={
                "Basis": 15,
                "Basket": 10,
                "Index": 15,
                "SingleName": 15,
                "Tranche": 0,
            },
            stale_days=5,
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
                ("CD_Drift", "Basis"): 10,
                ("CD_Drift", "Index"): 5,
                ("CD_StaleValue", "Basis"): 10,
                ("CD_StaleValue", "Index"): 5,
                ("CD_PointShock", "Basis"): 10,
                ("CD_PointShock", "Index"): 5,
                ("CD_SignFlip", "Basis"): 10,
                ("CD_SignFlip", "Index"): 5,
                ("CD_ScaleError", "Basis"): 10,
                ("CD_ScaleError", "Index"): 5,
                ("CD_SuddenZero", "Basis"): 10,
                ("CD_SuddenZero", "Index"): 5,
            },
            
            drift_days_by_type={
                "Basis": 15,
                "Index": 15,
            },
            stale_days=5,
        )
