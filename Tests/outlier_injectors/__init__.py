"""
Outlier injectors for generating synthetic data quality scenarios.

Provides OutlierInjector base class and implementations for PnL, CDS, PV,
and PnL Slices injection.
"""

from .base import OutlierInjector
from .pnl import PnLOutlierInjector
from .pnl_config import PnLInjectorConfig, PnLScenarioNames
from .credit_delta import CreditDeltaOutlierInjector
from .credit_delta_config import CreditDeltaInjectorConfig
from .pv import PVOutlierInjector
from .pv_config import PVInjectorConfig
from .pnl_slices import PnLSlicesOutlierInjector
from .pnl_slices_config import PnLSlicesInjectorConfig

__all__ = [
    'OutlierInjector',
    'PnLOutlierInjector',
    'PnLInjectorConfig',
    'PnLScenarioNames',
    'CreditDeltaOutlierInjector',
    'CreditDeltaInjectorConfig',
    'PVOutlierInjector',
    'PVInjectorConfig',
    'PnLSlicesOutlierInjector',
    'PnLSlicesInjectorConfig',
]
