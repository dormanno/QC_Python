"""Simple test to verify family scores output structure."""
import pandas as pd
import numpy as np
from Engine import qc_engine_presets
from column_names import pnl_column, main_column
from Engine.feature_normalizer import FeatureNormalizer
from QC_Orchestrator import QCOrchestrator

# Create minimal synthetic data
np.random.seed(42)
n_rows = 100
dates = pd.date_range('2024-01-01', periods=20)
trade_ids = range(1, n_rows + 1)

data = {
    main_column.RECORD_TYPE: ['Train'] * 60 + ['OOS'] * 40,
    main_column.TRADE: trade_ids,
    main_column.BOOK: ['Book1'] * n_rows,
    main_column.TRADE_TYPE: ['Type1'] * n_rows,
    main_column.DATE: np.repeat(dates, n_rows // len(dates)),
}

# Add PnL input features
for col in pnl_column.INPUT_FEATURES:
    data[col] = np.random.randn(n_rows) * 100

df = pd.DataFrame(data)

# Engineer features (simplified - normally done by Input handler)
df[pnl_column.TOTAL] = sum(df[col] for col in pnl_column.SLICE_COLUMNS)
df[pnl_column.EXPLAINED] = df[pnl_column.TOTAL] * 0.9
df[pnl_column.UNEXPLAINED] = df[pnl_column.TOTAL] * 0.1
df[pnl_column.TOTAL_JUMP] = df[pnl_column.TOTAL].diff().fillna(0)
df[pnl_column.UNEXPLAINED_JUMP] = df[pnl_column.UNEXPLAINED].diff().fillna(0)

print("Test data shape:", df.shape)
print("Columns:", list(df.columns))
print()

# Test WITHOUT family scores
print("=" * 60)
print("Testing WITHOUT keep_family_scores")
print("=" * 60)
normalizer1 = FeatureNormalizer(features=pnl_column.QC_FEATURES)
orchestrator1 = QCOrchestrator(
    normalizer=normalizer1,
    engine_preset=qc_engine_presets.preset_temporal_multivariate_pnl,
    split_identifier=main_column.TRADE_TYPE,
    keep_family_scores=False
)

try:
    oos_scores1 = orchestrator1.run(df)
    print("SUCCESS!")
    print(f"OOS scores shape: {oos_scores1.shape}")
    print(f"OOS score columns ({len(oos_scores1.columns)}):")
    for col in oos_scores1.columns:
        print(f"  - {col}")
    print()
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test WITH family scores
print()
print("=" * 60)
print("Testing WITH keep_family_scores")
print("=" * 60)
normalizer2 = FeatureNormalizer(features=pnl_column.QC_FEATURES)
orchestrator2 = QCOrchestrator(
    normalizer=normalizer2,
    engine_preset=qc_engine_presets.preset_temporal_multivariate_pnl,
    split_identifier=main_column.TRADE_TYPE,
    keep_family_scores=True
)

try:
    oos_scores2 = orchestrator2.run(df)
    print("SUCCESS!")
    print(f"OOS scores shape: {oos_scores2.shape}")
    print(f"OOS score columns ({len(oos_scores2.columns)}):")
    for col in oos_scores2.columns:
        print(f"  - {col}")
    print()
    
    # Check for family-specific columns
    family_cols = [col for col in oos_scores2.columns if any(
        f.name in col for f in qc_engine_presets.preset_temporal_multivariate_pnl.qc_feature_families
    ) and ('_AggScore' in col or any(m.score_name in col for m in 
         qc_engine_presets.preset_temporal_multivariate_pnl.methods_config.keys()))]
    print(f"\nFamily-specific columns found: {len(family_cols)}")
    for col in family_cols:
        print(f"  - {col}")
    
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("Comparison")
print("=" * 60)
if 'oos_scores1' in locals() and 'oos_scores2' in locals():
    print(f"Columns without family scores: {oos_scores1.shape[1]}")
    print(f"Columns with family scores: {oos_scores2.shape[1]}")
    print(f"Additional columns: {oos_scores2.shape[1] - oos_scores1.shape[1]}")
