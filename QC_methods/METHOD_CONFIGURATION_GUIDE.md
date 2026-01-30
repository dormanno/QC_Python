# QCEngine Method Configuration Enhancement

## Overview
The QCEngine class supports configurable combinations of QC methods through a single `methods_config` parameter that uses frozen dataclass instances (`QCMethod`) for type-safe method specification.

## What Changed

### 1. Created QCMethod Dataclass (qc_method_definitions.py)
A frozen, immutable dataclass defines each QC method:

```python
@dataclass(frozen=True)
class QCMethod:
    name: str           # Internal identifier (e.g., 'isolation_forest')
    score_name: str     # Column name for score output (e.g., qc_column.IF_SCORE)
```

### 2. QCMethods Class
Provides pre-defined instances of all available methods:

```python
class QCMethods:
    ISOLATION_FOREST = QCMethod(name='isolation_forest', score_name=qc_column.IF_SCORE)
    ROBUST_Z = QCMethod(name='robust_z', score_name=qc_column.ROBUST_Z_SCORE)
    IQR = QCMethod(name='iqr', score_name=qc_column.IQR_SCORE)
    ROLLING = QCMethod(name='rolling', score_name=qc_column.ROLLING_SCORE)
    LOF = QCMethod(name='lof', score_name=qc_column.LOF_SCORE)
    ECDF = QCMethod(name='ecdf', score_name=qc_column.ECDF_SCORE)
    HAMPEL = QCMethod(name='hampel', score_name=qc_column.HAMPEL_SCORE)
```

### 3. QCEngine Constructor
Uses QCMethod instances as dictionary keys:

```python
def __init__(self,
             qc_features: List[str],
             methods_config: dict[QCMethod, float],
             roll_window: int = 20):
```

**Parameters:**
- `methods_config` (dict[QCMethod, float]): Dictionary mapping QCMethod instances to weights
  - Keys: QCMethod instances (e.g., `QCMethods.ISOLATION_FOREST`)
  - Values: Weights (floats) for aggregation
  - Only methods included in this dict will be enabled

## Usage Examples

### Example 1: All Methods
```python
from qc_method_definitions import QCMethods

methods_config = {
    QCMethods.ISOLATION_FOREST: 0.2,
    QCMethods.ROBUST_Z: 0.1,
    QCMethods.ROLLING: 0.1,
    QCMethods.IQR: 0.1,
    QCMethods.LOF: 0.2,
    QCMethods.ECDF: 0.2,
    QCMethods.HAMPEL: 0.1
}

engine = QCEngine(
    qc_features=qc_features,
    methods_config=methods_config,
    roll_window=20
)
```

### Example 2: Statistical Methods Only
```python
methods_config = {
    QCMethods.ROBUST_Z: 0.25,
    QCMethods.ROLLING: 0.25,
    QCMethods.IQR: 0.25,
    QCMethods.HAMPEL: 0.25
}

engine = QCEngine(
    qc_features=qc_features,
    methods_config=methods_config,
    roll_window=20
)
```

### Example 3: ML-Based Methods Only
```python
methods_config = {
    QCMethods.ISOLATION_FOREST: 0.34,
    QCMethods.LOF: 0.33,
    QCMethods.ECDF: 0.33
}

engine = QCEngine(
    qc_features=qc_features,
    methods_config=methods_config,
    roll_window=20
)
```

### Example 4: Single Method for Testing
```python
methods_config = {
    QCMethods.ISOLATION_FOREST: 1.0
}

engine = QCEngine(
    qc_features=qc_features,
    methods_config=methods_config,
    roll_window=20
)
```

## Available Methods

| QCMethod Instance | Description | Type |
|------------------|-------------|------|
| `QCMethods.ISOLATION_FOREST` | Isolation Forest anomaly detection | ML |
| `QCMethods.ROBUST_Z` | Robust Z-Score using median and MAD | Statistical |
| `QCMethods.IQR` | Interquartile Range outlier detection | Statistical |
| `QCMethods.ROLLING` | Rolling window Z-Score | Statistical + Temporal |
| `QCMethods.LOF` | Local Outlier Factor | ML |
| `QCMethods.ECDF` | Empirical Cumulative Distribution Function | Statistical |
| `QCMethods.HAMPEL` | Hampel Filter for time series | Statistical + Temporal |

## Benefits

1. **Type Safety**: IDE autocomplete and type checking for method names
2. **Immutability**: Frozen dataclass prevents accidental modification
3. **Single Source of Truth**: Method names and score columns defined in one place
4. **Cleaner API**: No string literals scattered throughout code
5. **Refactoring Safety**: Renaming a method updates all references automatically
6. **Self-Documenting**: Clear relationship between method and its score column

## Comparison: Old vs New API

### Old API (String-based)
```python
# Prone to typos, no IDE support
methods_config = {
    'isolation_forest': 0.5,  # String literal - no autocomplete
    'ecdf': 0.5
}

engine = QCEngine(
    qc_features=qc_features,
    methods_config=methods_config,
    roll_window=20
)
```

### New API (Dataclass-based)
```python
# Type-safe, IDE autocomplete, refactor-safe
methods_config = {
    QCMethods.ISOLATION_FOREST: 0.5,  # IDE autocomplete works
    QCMethods.ECDF: 0.5
}

engine = QCEngine(
    qc_features=qc_features,
    methods_config=methods_config,
    roll_window=20
)
```

## Files Modified/Created

1. **qc_method_definitions.py**: New file with QCMethod dataclass and QCMethods class
2. **Engine/qc_engine.py**: Updated to accept QCMethod instances
3. **run_qc.py**: Updated to use QCMethods instances
4. **example_method_configurations.py**: Updated all examples
5. **Tests/test_qc_orchestrator.py**: Updated configuration
6. **Tests/test_qc_engine_method_configuration.py**: Updated all tests

## Testing

Run the tests with:
```bash
pytest Tests/test_qc_engine_method_configuration.py -v
```

## Common Use Cases

### Fast QC (Minimal Computation)
```python
methods_config = {
    QCMethods.ROBUST_Z: 0.4,
    QCMethods.IQR: 0.3,
    QCMethods.ECDF: 0.3
}
```

### Temporal Analysis (Historical Context)
```python
methods_config = {
    QCMethods.ISOLATION_FOREST: 0.3,
    QCMethods.ROLLING: 0.35,
    QCMethods.HAMPEL: 0.35
}
```

### Comprehensive QC (All Methods)
```python
methods_config = {
    QCMethods.ISOLATION_FOREST: 0.2,
    QCMethods.ROBUST_Z: 0.1,
    QCMethods.ROLLING: 0.1,
    QCMethods.IQR: 0.1,
    QCMethods.LOF: 0.2,
    QCMethods.ECDF: 0.2,
    QCMethods.HAMPEL: 0.1
}
```

### Production Optimized (Balanced Speed/Accuracy)
```python
methods_config = {
    QCMethods.ISOLATION_FOREST: 0.3,
    QCMethods.ROBUST_Z: 0.35,
    QCMethods.ECDF: 0.35
}
```
