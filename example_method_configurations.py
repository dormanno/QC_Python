"""
Example configurations for QCEngine with different method combinations.

This module demonstrates how to instantiate QCEngine with various
combinations of QC methods using QCMethod dataclass instances.
"""

from column_names import pnl_column
from qc_method_definitions import QCMethods
from Engine.qc_engine import QCEngine

# Define QC features (common to all examples)
qc_features = [
    pnl_column.START, 
    *pnl_column.SLICE_COLUMNS, 
    pnl_column.TOTAL, 
    pnl_column.EXPLAINED, 
    pnl_column.UNEXPLAINED
]

# ============================================================================
# Example 1: Use ALL methods
# ============================================================================
methods_config_all = {
    QCMethods.ISOLATION_FOREST: 0.2,
    QCMethods.ROBUST_Z: 0.1,
    QCMethods.ROLLING: 0.1,
    QCMethods.IQR: 0.1,
    QCMethods.LOF: 0.2,
    QCMethods.ECDF: 0.2,
    QCMethods.HAMPEL: 0.1
}

engine_all_methods = QCEngine(
    qc_features=qc_features,
    methods_config=methods_config_all,
    roll_window=20
)

# ============================================================================
# Example 2: Use only STATISTICAL methods (no ML-based methods)
# ============================================================================
methods_config_statistical = {
    QCMethods.ROBUST_Z: 0.25,
    QCMethods.ROLLING: 0.25,
    QCMethods.IQR: 0.25,
    QCMethods.HAMPEL: 0.25
}

engine_statistical = QCEngine(
    qc_features=qc_features,
    methods_config=methods_config_statistical,
    roll_window=20
)

# ============================================================================
# Example 3: Use only ML-BASED methods
# ============================================================================
methods_config_ml = {
    QCMethods.ISOLATION_FOREST: 0.34,
    QCMethods.LOF: 0.33,
    QCMethods.ECDF: 0.33
}

engine_ml = QCEngine(
    qc_features=qc_features,
    methods_config=methods_config_ml,
    roll_window=20
)

# ============================================================================
# Example 4: Use FAST methods only (minimal computation)
# ============================================================================
methods_config_fast = {
    QCMethods.ROBUST_Z: 0.4,
    QCMethods.IQR: 0.3,
    QCMethods.ECDF: 0.3
}

engine_fast = QCEngine(
    qc_features=qc_features,
    methods_config=methods_config_fast,
    roll_window=20
)

# ============================================================================
# Example 5: Use TEMPORAL methods (consider historical data)
# ============================================================================
methods_config_temporal = {
    QCMethods.ISOLATION_FOREST: 0.3,
    QCMethods.ROLLING: 0.35,
    QCMethods.HAMPEL: 0.35
}

engine_temporal = QCEngine(
    qc_features=qc_features,
    methods_config=methods_config_temporal,
    roll_window=20
)

# ============================================================================
# Example 6: Single method for testing/debugging
# ============================================================================
methods_config_single = {
    QCMethods.ISOLATION_FOREST: 1.0
}

engine_single = QCEngine(
    qc_features=qc_features,
    methods_config=methods_config_single,
    roll_window=20
)

# ============================================================================
# Example 7: Custom balanced configuration
# ============================================================================
methods_config_balanced = {
    QCMethods.ISOLATION_FOREST: 0.3,
    QCMethods.ROBUST_Z: 0.2,
    QCMethods.ECDF: 0.25,
    QCMethods.ROLLING: 0.25
}

engine_balanced = QCEngine(
    qc_features=qc_features,
    methods_config=methods_config_balanced,
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
