"""
Outlier injectors for generating synthetic data quality scenarios.

Provides OutlierInjector base class and implementations for PnL and CDS injection.
"""

from .base import OutlierInjector
from .pnl import PnLOutlierInjector
from .pnl_config import PnLInjectorConfig, PnLScenarioNames
from .credit_delta import CreditDeltaOutlierInjector
from .credit_delta_config import CreditDeltaInjectorConfig

__all__ = [
    'OutlierInjector',
    'PnLOutlierInjector',
    'PnLInjectorConfig',
    'PnLScenarioNames',
    'CreditDeltaOutlierInjector',
    'CreditDeltaInjectorConfig',
]
