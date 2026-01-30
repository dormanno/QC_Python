"""
Example configurations for QCEngine with different method combinations.

This module demonstrates how to instantiate QCEngine with various
combinations of QC methods using QCMethod dataclass instances.
"""

from column_names import pnl_column, cds_column, cdi_column
from QC_methods.qc_method_definitions import QCMethodDefinitions
from Engine.qc_engine import QCEngine

# ============================================================================
# Example 1: Use ALL methods
# ============================================================================
methods_config_all = {
    QCMethodDefinitions.ISOLATION_FOREST: 0.2,
    QCMethodDefinitions.ROBUST_Z: 0.1,
    QCMethodDefinitions.ROLLING: 0.1,
    QCMethodDefinitions.IQR: 0.1,
    QCMethodDefinitions.LOF: 0.2,
    QCMethodDefinitions.ECDF: 0.2,
    QCMethodDefinitions.HAMPEL: 0.1
}

engine_all_methods_pnl = QCEngine(
    qc_features=pnl_column.QC_FEATURES,
    methods_config=methods_config_all,
    roll_window=20
)

# ============================================================================
# Example 2: Use only STATISTICAL methods (no ML-based methods)
# ============================================================================
methods_config_statistical = {
    QCMethodDefinitions.ROBUST_Z: 0.25,
    QCMethodDefinitions.ROLLING: 0.25,
    QCMethodDefinitions.IQR: 0.25,
    QCMethodDefinitions.HAMPEL: 0.25
}

engine_statistical_pnl = QCEngine(
    qc_features=pnl_column.QC_FEATURES,
    methods_config=methods_config_statistical,
    roll_window=20
)

# ============================================================================
# Example 3: Use only ML-BASED methods
# ============================================================================
methods_config_ml = {
    QCMethodDefinitions.ISOLATION_FOREST: 0.34,
    QCMethodDefinitions.LOF: 0.33,
    QCMethodDefinitions.ECDF: 0.33
}

engine_ml_pnl = QCEngine(
    qc_features=pnl_column.QC_FEATURES,
    methods_config=methods_config_ml,
    roll_window=20
)

# ============================================================================
# Example 4: Use FAST methods only (minimal computation)
# ============================================================================
methods_config_fast = {
    QCMethodDefinitions.ROBUST_Z: 0.4,
    QCMethodDefinitions.IQR: 0.3,
    QCMethodDefinitions.ECDF: 0.3
}

engine_fast_pnl = QCEngine(
    qc_features=pnl_column.QC_FEATURES,
    methods_config=methods_config_fast,
    roll_window=20
)

# ============================================================================
# Example 5: Use TEMPORAL methods (consider historical data)
# ============================================================================
methods_config_temporal = {
    QCMethodDefinitions.ISOLATION_FOREST: 0.3,
    QCMethodDefinitions.ROLLING: 0.35,
    QCMethodDefinitions.HAMPEL: 0.35
}

engine_temporal_pnl = QCEngine(
    qc_features=pnl_column.QC_FEATURES,
    methods_config=methods_config_temporal,
    roll_window=20
)

# ============================================================================
# Example 6: Single method for testing/debugging
# ============================================================================
methods_config_IF = {
    QCMethodDefinitions.ISOLATION_FOREST: 1.0
}

engine_IF_pnl = QCEngine(
    qc_features=pnl_column.QC_FEATURES,
    methods_config=methods_config_IF,
    roll_window=20
)

# ============================================================================
# Example 7: Custom balanced configuration
# ============================================================================
methods_config_balanced = {
    QCMethodDefinitions.ISOLATION_FOREST: 0.3,
    QCMethodDefinitions.ROBUST_Z: 0.2,
    QCMethodDefinitions.ECDF: 0.25,
    QCMethodDefinitions.ROLLING: 0.25
}

engine_balanced_pnl = QCEngine(
    qc_features=pnl_column.QC_FEATURES,
    methods_config=methods_config_balanced,
    roll_window=20
)

method_config_temporal_multivariate = {
    QCMethodDefinitions.ISOLATION_FOREST: 0.25,
    QCMethodDefinitions.ROLLING: 0.15,
    QCMethodDefinitions.HAMPEL: 0.10,
    QCMethodDefinitions.ROBUST_Z: 0.15,
    QCMethodDefinitions.LOF: 0.20,
    QCMethodDefinitions.ECDF: 0.15
}

engine_temporal_multivariate_pnl = QCEngine(
    qc_features=pnl_column.QC_FEATURES,
    methods_config=method_config_temporal_multivariate,
    roll_window=20
)

methods_config_robust_univariate = {
    QCMethodDefinitions.ROBUST_Z: 0.35,
    QCMethodDefinitions.IQR: 0.20,
    QCMethodDefinitions.ECDF: 0.25,
    QCMethodDefinitions.HAMPEL: 0.20
}

engine_robust_univariate_cdi = QCEngine(
    qc_features=cdi_column.QC_FEATURES,
    methods_config=methods_config_robust_univariate,
    roll_window=20
)

methods_reactive_univariate = {
    QCMethodDefinitions.ROBUST_Z: 0.30,    
    QCMethodDefinitions.ECDF: 0.25,
    QCMethodDefinitions.HAMPEL: 0.25,
    QCMethodDefinitions.ROLLING: 0.20
}

engine_reactive_univariate_cds = QCEngine(
    qc_features=cds_column.QC_FEATURES,
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
