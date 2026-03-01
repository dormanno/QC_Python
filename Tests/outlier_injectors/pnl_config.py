"""
Configuration classes for PnL outlier injectors.

Provides configuration presets for PnL datasets and allows customization
of injection parameters per scenario.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PnLScenarioNames:
    """PnL injection scenario type names (used as RecordType labels)."""
    PV_SPIKE = "Injected_PV_Spike"
    PV_STEP = "Injected_PV_Step"
    PV_STALE = "Injected_PV_Stale"
    PV_SCALE = "Injected_PV_Scale"
    PV_SIGN_FLIP = "Injected_PV_SignFlip"
    SLICE_SPIKE = "Injected_Slice_Spike"
    SLICE_STALE = "Injected_Slice_Stale"
    REALLOCATION = "Injected_Reallocation"
    IDENTITY_BREAK = "Injected_IdentityBreak"
    CROSS_FAMILY = "Injected_CrossFamily"


class PnLInjectorConfig:
    """
    Configuration for PnL outlier injector.

    Contains per-scenario trade counts and the scale factor used by the
    PV_EoD_ScaleError scenario.  All other behavioural parameters
    (stale-day ranges, donor slices, etc.) remain inside PnLOutlierInjector
    to keep this config minimal.
    """

    def __init__(
        self,
        trade_counts: Dict[str, int],
        scale_factor: float = 100.0,
    ):
        """
        Initialise configuration.

        Args:
            trade_counts: Number of trades to inject per scenario.
                          Keys are PnLScenarioNames constants.
            scale_factor: Multiplicative factor for the PV_EoD_ScaleError
                          scenario (default 100).
        """
        self.trade_counts = trade_counts
        self.scale_factor = scale_factor

    # -- convenience accessor --------------------------------------------------

    def get_trade_count(self, scenario: str) -> int:
        """Return the trade count for *scenario*, defaulting to 0."""
        return self.trade_counts.get(scenario, 0)

    # -- presets ---------------------------------------------------------------

    @staticmethod
    def default_preset() -> "PnLInjectorConfig":
        """
        Default preset mirroring the original hard-coded values in
        ``PnLOutlierInjector.inject()``.

        Returns:
            PnLInjectorConfig with default trade counts.
        """
        names = PnLScenarioNames
        return PnLInjectorConfig(
            trade_counts={
                names.PV_SPIKE: 10,
                names.PV_STEP: 4,
                names.PV_STALE: 4,
                names.PV_SCALE: 5,
                names.PV_SIGN_FLIP: 5,
                names.SLICE_SPIKE: 10,
                names.SLICE_STALE: 4,
                names.REALLOCATION: 10,
                names.IDENTITY_BREAK: 5,
                names.CROSS_FAMILY: 5,
            },
            scale_factor=100.0,
        )
