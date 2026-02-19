"""
Outlier injectors for generating synthetic data quality scenarios.

Provides OutlierInjector base class and implementations for PnL and CDS injection.
"""

from .base import OutlierInjector
from .pnl import PnLOutlierInjector
from .credit_delta import CreditDeltaOutlierInjector

__all__ = ['OutlierInjector', 'PnLOutlierInjector', 'CreditDeltaOutlierInjector', 'CdsOutlierInjector']
