"""
Preset configurations for QCEngine with different method combinations.

This module defines QCEnginePreset instances that hold all parameters needed
to instantiate QCEngine objects. Engines are built on-demand via build_engine().
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from column_names import pnl_column, cds_column, cdi_column, pv_column, pnl_slices_column, qc_column, QCFeatureFamily
from QC_methods.qc_method_definitions import QCMethodDefinition, QCMethodDefinitions
from Engine.aggregator import ConsensusMode
from Engine.qc_engine import QCEngine
from Engine.score_normalizer import ScoreNormalizer


@dataclass
class QCEnginePreset:
    """Preset holding all parameters needed to build QCEngine instances per feature family.

    Engines are not instantiated at module load time. Instead, call build_engine(family)
    to create a fresh QCEngine instance for a specific feature family on demand.

    Attributes:
        qc_feature_families: List of QCFeatureFamily instances defining feature groups.
        methods_config: Dictionary mapping QCMethodDefinition instances to weights.
        roll_window: Window size for rolling methods.
        filters: List of QCMethodDefinition instances for filter methods (e.g., stale value detection).
    """
    qc_feature_families: List[QCFeatureFamily]
    methods_config: Dict[QCMethodDefinition, float]
    roll_window: int = 20
    consensus: ConsensusMode | str = ConsensusMode.NONE
    filters: List[QCMethodDefinition] = None

    def __post_init__(self):
        """Validate feature family weights and initialize filters."""
        # Initialize filters to empty list if not provided
        if self.filters is None:
            object.__setattr__(self, 'filters', [])

        # Validate that feature family weights sum to 1
        weight_sum = sum(f.weight for f in self.qc_feature_families)
        if abs(weight_sum - 1.0) > 1e-9:
            raise ValueError(
                f"Feature family weights must sum to 1.0. "
                f"Current sum: {weight_sum} from families: "
                f"{[(f.name, f.weight) for f in self.qc_feature_families]}"
            )

    def build_engine(self, family: QCFeatureFamily) -> QCEngine:
        """Instantiate a fresh QCEngine for a specific feature family.

        Args:
            family: The feature family whose features will be used by the engine.

        Returns:
            QCEngine: New engine instance with the family's features and a fresh ScoreNormalizer.
        """
        return QCEngine(
            qc_features=list(family.features),
            methods_config=self.methods_config,
            roll_window=self.roll_window,
            score_normalizer=ScoreNormalizer(),
            consensus=self.consensus,
            filters=self.filters
        )

    @property
    def all_qc_features(self) -> List[str]:
        """Flat ordered union of all feature family features.

        Returns:
            List[str]: All unique feature column names across all families.
        """
        seen = set()
        result = []
        for fam in self.qc_feature_families:
            for f in fam.features:
                if f not in seen:
                    seen.add(f)
                    result.append(f)
        return result

    def get_score_columns(self, include_family_scores: bool = False) -> List[str]:
        """Get list of all score columns that engines built from this preset will generate.

        Args:
            include_family_scores: If True, include family-specific per-method and aggregated scores.

        Returns:
            List[str]: Score column names including individual method scores,
                      aggregated score, and QC flag. If include_family_scores=True,
                      also includes family-prefixed scores.
        """
        result = []
        
        # Add family-specific scores if requested
        if include_family_scores:
            method_score_names = [method_def.score_name for method_def in self.methods_config.keys()]
            for family in self.qc_feature_families:
                # Per-method scores for this family
                for score_name in method_score_names:
                    result.append(f"{family.name}_{score_name}")
                # Aggregated score for this family
                result.append(f"{family.name}_AggScore")
        
        # Add combined scores
        method_score_cols = [method_def.score_name for method_def in self.methods_config.keys()]
        result.extend(method_score_cols)
        result.append(qc_column.AGGREGATED_SCORE)
        result.append(qc_column.QC_FLAG)
        
        return result

    def get_family_score_columns(self, family: QCFeatureFamily) -> List[str]:
        """Get score columns for a specific feature family.

        Returns the family-prefixed per-method scores and the family aggregated
        score column name.

        Args:
            family: The feature family to get score columns for.

        Returns:
            List[str]: Family-specific score column names,
                e.g. ["Valuation_IF_score", ..., "Valuation_AggScore"].
        """
        result = []
        for method_def in self.methods_config.keys():
            result.append(f"{family.name}_{method_def.score_name}")
        result.append(f"{family.name}_AggScore")
        return result

class MultiFamilyQCEnginePreset(QCEnginePreset):
    """Preset that allows different QC method configurations per feature family.

    Inherits from QCEnginePreset. When a family has an entry in
    ``family_methods_config``, that configuration is used instead of the
    base ``methods_config``.

    Attributes:
        family_methods_config: Mapping of family name to its own
            {QCMethodDefinition: weight} dictionary. Families not listed
            here fall back to the base ``methods_config``.
    """

    def __init__(
        self,
        qc_feature_families: List[QCFeatureFamily],
        methods_config: Dict[QCMethodDefinition, float],
        family_methods_config: Dict[str, Dict[QCMethodDefinition, float]],
        roll_window: int = 20,
        consensus: ConsensusMode | str = ConsensusMode.NONE,
        filters: Optional[List[QCMethodDefinition]] = None,
    ):
        self.family_methods_config = family_methods_config
        super().__init__(
            qc_feature_families=qc_feature_families,
            methods_config=methods_config,
            roll_window=roll_window,
            consensus=consensus,
            filters=filters,
        )

    # -- helpers ---------------------------------------------------------

    def _methods_for_family(self, family: QCFeatureFamily) -> Dict[QCMethodDefinition, float]:
        """Return the method config applicable to *family*."""
        return self.family_methods_config.get(family.name, self.methods_config)

    # -- overrides -------------------------------------------------------

    def build_engine(self, family: QCFeatureFamily) -> QCEngine:
        """Build a QCEngine using the family-specific method config."""
        return QCEngine(
            qc_features=list(family.features),
            methods_config=self._methods_for_family(family),
            roll_window=self.roll_window,
            score_normalizer=ScoreNormalizer(),
            consensus=self.consensus,
            filters=self.filters,
        )

    def get_score_columns(self, include_family_scores: bool = False) -> List[str]:
        """Score columns as the union of all per-family methods."""
        result: List[str] = []

        if include_family_scores:
            for family in self.qc_feature_families:
                fam_config = self._methods_for_family(family)
                for method_def in fam_config:
                    result.append(f"{family.name}_{method_def.score_name}")
                result.append(f"{family.name}_AggScore")

        # Combined columns = union of all method score names (order-preserving)
        seen: set = set()
        combined: List[str] = []
        for family in self.qc_feature_families:
            for method_def in self._methods_for_family(family):
                if method_def.score_name not in seen:
                    seen.add(method_def.score_name)
                    combined.append(method_def.score_name)
        result.extend(combined)
        result.append(qc_column.AGGREGATED_SCORE)
        result.append(qc_column.QC_FLAG)
        return result

    def get_family_score_columns(self, family: QCFeatureFamily) -> List[str]:
        """Score columns for a specific family (family-prefixed)."""
        fam_config = self._methods_for_family(family)
        result = [f"{family.name}_{md.score_name}" for md in fam_config]
        result.append(f"{family.name}_AggScore")
        return result


method_config_temporal_multivariate = {
    QCMethodDefinitions.ISOLATION_FOREST: 0.25,
    QCMethodDefinitions.ROLLING: 0.15,
    QCMethodDefinitions.HAMPEL: 0.10,
    QCMethodDefinitions.ROBUST_Z: 0.15,
    QCMethodDefinitions.LOF: 0.20,
    QCMethodDefinitions.ECDF: 0.15
}

preset_temporal_multivariate_pnl = QCEnginePreset(
    qc_feature_families=pnl_column.QC_FEATURE_FAMILIES,
    methods_config=method_config_temporal_multivariate,
    roll_window=20,
    consensus=ConsensusMode.QUALIFIED_MAJORITY,
    filters=[QCMethodDefinitions.STALE_VALUE]
)

methods_config_robust_univariate = {
    QCMethodDefinitions.ROBUST_Z: 0.25,
    # QCMethodDefinitions.ECDF: 0.20,
    QCMethodDefinitions.HAMPEL: 0.25,
    QCMethodDefinitions.ROLLING: 0.25,
    # QCMethodDefinitions.IQR: 0.10,    
    # QCMethodDefinitions.LOF: 0.10,
    QCMethodDefinitions.ISOLATION_FOREST: 0.25
}

preset_robust_univariate_cdi = QCEnginePreset(
    qc_feature_families=cdi_column.QC_FEATURE_FAMILIES,
    methods_config=methods_config_robust_univariate,
    roll_window=20,
    consensus=ConsensusMode.QUALIFIED_MAJORITY,
    filters=[QCMethodDefinitions.STALE_VALUE]
)

methods_reactive_univariate = {
    QCMethodDefinitions.ROBUST_Z: 0.20,    
    QCMethodDefinitions.ECDF: 0.25,
    QCMethodDefinitions.HAMPEL: 0.25,
    QCMethodDefinitions.ROLLING: 0.30
}

preset_reactive_univariate_cds = QCEnginePreset(
    qc_feature_families=cds_column.QC_FEATURE_FAMILIES,
    methods_config=methods_reactive_univariate,
    roll_window=20,
    consensus=ConsensusMode.QUALIFIED_MAJORITY,
    filters=[QCMethodDefinitions.STALE_VALUE]
)

methods_reactive_univariate_pv = {
    QCMethodDefinitions.ROBUST_Z: 0.25,    
    QCMethodDefinitions.ECDF: 0.25,
    QCMethodDefinitions.HAMPEL: 0.25,
    QCMethodDefinitions.ROLLING: 0.25,
    # QCMethodDefinitions.ISOLATION_FOREST: 0.15,
    # QCMethodDefinitions.IQR: 0.10,    
    # QCMethodDefinitions.LOF: 0.15
}

preset_reactive_univariate_pv = QCEnginePreset(
    qc_feature_families=pv_column.QC_FEATURE_FAMILIES,
    methods_config=methods_reactive_univariate_pv,
    roll_window=20,
    consensus=ConsensusMode.QUALIFIED_MAJORITY,
    filters=[QCMethodDefinitions.STALE_VALUE]
)

methods_all_pnl_slices = {
    QCMethodDefinitions.ISOLATION_FOREST: 1/7,
    QCMethodDefinitions.ROBUST_Z: 1/7,
    QCMethodDefinitions.ROLLING: 1/7,
    QCMethodDefinitions.IQR: 1/7,
    QCMethodDefinitions.LOF: 1/7,
    QCMethodDefinitions.ECDF: 1/7,
    QCMethodDefinitions.HAMPEL: 1/7,
}

preset_all_methods_pnl_slices = QCEnginePreset(
    qc_feature_families=pnl_slices_column.QC_FEATURE_FAMILIES,
    methods_config=methods_all_pnl_slices,
    roll_window=20,
    consensus=ConsensusMode.QUALIFIED_MAJORITY,
    filters=[QCMethodDefinitions.STALE_VALUE]
)

# ---------- PnL Slices per-family presets (MultiFamilyQCEnginePreset) ----------
# SmallSlices and LargeSlices get their own method mix.
# Edit the dicts below to tune each family independently.

methods_pnl_slices_small = {
    # QCMethodDefinitions.ISOLATION_FOREST: 1/5,
    # QCMethodDefinitions.ROBUST_Z: 1/7,
    QCMethodDefinitions.ROLLING: 1/3,
    QCMethodDefinitions.IQR: 1/3,
    # QCMethodDefinitions.LOF: 1/7,
    QCMethodDefinitions.ECDF: 1/3,
    # QCMethodDefinitions.HAMPEL: 1/5,
}

methods_pnl_slices_large = {
    # QCMethodDefinitions.ISOLATION_FOREST: 1/7,
    # QCMethodDefinitions.ROBUST_Z: 1/4,
    QCMethodDefinitions.ROLLING: 1/3,
    QCMethodDefinitions.IQR: 1/3,
    # QCMethodDefinitions.LOF: 1/7,
    QCMethodDefinitions.ECDF: 1/3,
    # QCMethodDefinitions.HAMPEL: 1/5,
}

preset_per_family_pnl_slices = MultiFamilyQCEnginePreset(
    qc_feature_families=pnl_slices_column.QC_FEATURE_FAMILIES,
    methods_config=methods_all_pnl_slices,              # fallback (unused when all families are listed)
    family_methods_config={
        "SmallSlices": methods_pnl_slices_small,
        "LargeSlices": methods_pnl_slices_large,
    },
    roll_window=20,
    consensus=ConsensusMode.QUALIFIED_MAJORITY,
    filters=[QCMethodDefinitions.STALE_VALUE],
)

# ============================================================================
# Available QCMethod instances from QCMethods class:
# - QCMethods.ISOLATION_FOREST: Isolation Forest anomaly detection
# - QCMethods.ROBUST_Z: Robust Z-Score using median and MAD
# - QCMethods.IQR: Interquartile Range outlier detection
# - QCMethods.ROLLING: Rolling window Z-Score
# - QCMethods.LOF: Local Outlier Factor
# - QCMethods.ECDF: Empirical Cumulative Distribution Function
# - QCMethods.HAMPEL: Hampel Filter for time series
# ============================================================================
