"""Quick verification script to check family scores functionality."""
import pandas as pd
from Engine import qc_engine_presets
from column_names import pnl_column

# Check the preset configuration
preset = qc_engine_presets.preset_temporal_multivariate_pnl
print("PnL QC Feature Families:")
for family in preset.qc_feature_families:
    print(f"  - {family.name} (weight={family.weight})")
    print(f"    Features: {family.features}")
print()

# Check methods config
print("Methods Configuration:")
for method, weight in preset.methods_config.items():
    print(f"  - {method.name}: {weight} (score: {method.score_name})")
print()

# Calculate expected columns for family scores
print("Expected additional columns with keep_family_scores=True:")
family_method_cols = []
family_agg_cols = []

for family in preset.qc_feature_families:
    for method in preset.methods_config.keys():
        col_name = f"{family.name}_{method.score_name}"
        family_method_cols.append(col_name)
        print(f"  - {col_name}")
    
    agg_col = f"{family.name}_AggScore"
    family_agg_cols.append(agg_col)
    print(f"  - {agg_col}")

print()
print(f"Total additional columns: {len(family_method_cols) + len(family_agg_cols)}")
print(f"  - Per-method scores: {len(family_method_cols)}")
print(f"  - Aggregated scores: {len(family_agg_cols)}")
