"""
Preset configurations for QCEngine with different method combinations.

This module defines QCEnginePreset instances that hold all parameters needed
to instantiate QCEngine objects. Engines are built on-demand via build_engine().
"""

from dataclasses import dataclass
from typing import List, Dict

from column_names import pnl_column, cds_column, cdi_column, qc_column, QCFeatureFamily
from QC_methods.qc_method_definitions import QCMethodDefinition, QCMethodDefinitions
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
    """
    qc_feature_families: List[QCFeatureFamily]
    methods_config: Dict[QCMethodDefinition, float]
    roll_window: int = 20

    def __post_init__(self):
        """Validate that feature family weights sum to 1."""
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
            score_normalizer=ScoreNormalizer()
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

    def get_score_columns(self) -> List[str]:
        """Get list of all score columns that engines built from this preset will generate.

        Returns:
            List[str]: Score column names including individual method scores,
                      aggregated score, and QC flag.
        """
        method_score_cols = [method_def.score_name for method_def in self.methods_config.keys()]
        return method_score_cols + [qc_column.AGGREGATED_SCORE, qc_column.QC_FLAG]

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
    roll_window=20
)

methods_config_robust_univariate = {
    QCMethodDefinitions.ROBUST_Z: 0.35,
    QCMethodDefinitions.IQR: 0.20,
    QCMethodDefinitions.ECDF: 0.25,
    QCMethodDefinitions.HAMPEL: 0.20
}

preset_robust_univariate_cdi = QCEnginePreset(
    qc_feature_families=cdi_column.QC_FEATURE_FAMILIES,
    methods_config=methods_config_robust_univariate,
    roll_window=20
)

methods_reactive_univariate = {
    QCMethodDefinitions.ROBUST_Z: 0.30,    
    QCMethodDefinitions.ECDF: 0.25,
    QCMethodDefinitions.HAMPEL: 0.25,
    QCMethodDefinitions.ROLLING: 0.20
}

preset_reactive_univariate_cds = QCEnginePreset(
    qc_feature_families=cds_column.QC_FEATURE_FAMILIES,
    methods_config=methods_reactive_univariate,
    roll_window=20
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
