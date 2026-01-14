import numpy as np
import pandas as pd
from typing import List

import InputOutput as IO
import ColumnNames as Column

from QC_methods import IsolationForestQC, RobustZQC, IQRQC, RollingZQC
from Aggregator import ScoreAggregator

# Create input and output handlers
input_handler = IO.Input()
output_handler = IO.Output()

# ----------------------------
# Config
# ----------------------------
# TRAIN_DAYS is now computed dynamically in run_qc_orchestrator

QC_FEATURES = [
    Column.START, *Column.PNL_SLICES, Column.TOTAL, Column.EXPLAINED, Column.UNEXPLAINED
]
ROLL_WINDOW = 20

# ----------------------------
# Normalization across trades
# ----------------------------

def compute_per_trade_denominators(train_df: pd.DataFrame, level_feats: List[str]) -> pd.DataFrame:
    den = (train_df.groupby(Column.TRADE)[level_feats]
           .apply(lambda g: g.abs().median())
           .reset_index())
    den.columns = [Column.TRADE] + [f"{c}__den" for c in level_feats]
    return den

def apply_denominators(df: pd.DataFrame, den: pd.DataFrame, level_feats: List[str]) -> pd.DataFrame:
    if den is None:
        return df
    g = df.merge(den, on=Column.TRADE, how="left")
    for c in level_feats:
        dcol = f"{c}__den"
        g[c] = g[c] / g[dcol].replace(0, np.nan)
    return g.drop(columns=[f"{c}__den" for c in level_feats])

# ----------------------------
# Orchestrate
# ----------------------------

def run_qc_orchestrator(input_path: str, qc_features: List[str]) -> str:
    """
    Main QC orchestrator function that processes the input CSV and returns the output path.
    
    Args:
        input_path (str): Path to the input CSV file.
        qc_features (List[str]): List of feature column names to use for QC scoring.
    
    Returns:
        str: Path to the output file with QC scores.
    """
    fullDataSet = input_handler.read_input(input_path)
    fullDataSet = fullDataSet.sort_values(Column.DATE)

    # 2) Split train / OOS by date
    dates = fullDataSet[Column.DATE].drop_duplicates().sort_values().to_list()
    TRAIN_DAYS = int(2 / 3 * len(dates))
    if len(dates) <= TRAIN_DAYS:
        raise ValueError("Not enough data: need > TRAIN_DAYS unique dates for walk-forward.")

    cutoff = dates[TRAIN_DAYS - 1]
    rawTrainDataSet = fullDataSet.loc[fullDataSet[Column.DATE] <= cutoff].copy()
    oos_dates = dates[TRAIN_DAYS:]

    # 3) Stabilize scale (optional but recommended)
    denominators = compute_per_trade_denominators(rawTrainDataSet, qc_features)
    trainDataSet = apply_denominators(rawTrainDataSet, denominators, qc_features)

    # 4) Instantiate QC methods and fit on TRAIN only
    ifqc = IsolationForestQC(
        base_feats=qc_features,
        identity_column=Column.TRADE,
        temporal_column=Column.DATE,
        mode="time_series",
        per_trade_normalize=False,  # we normalize in the orchestrator using TRAIN denominators
        use_robust_scaler=True,
        n_estimators=200,
        max_samples=256,
        contamination=0.01,
    )
    ifqc.fit(trainDataSet)

    robustZScoreEngine = RobustZQC(features=qc_features, identity_column=Column.TRADE)
    robustZScoreEngine.fit(trainDataSet)

    interquartileRangeEngine = IQRQC(features=qc_features, identity_column=Column.TRADE)
    interquartileRangeEngine.fit(trainDataSet)

    rollingMeanEngine = RollingZQC(features=qc_features, identity_column=Column.TRADE, temporal_column=Column.DATE, window=ROLL_WINDOW)
    rollingMeanEngine.fit(trainDataSet)  # warm-up buffers

    aggregator = ScoreAggregator(w_if=0.4, w_rz=0.3, w_roll=0.2, w_iqr=0.1)

    # 5) Iterate OOS day-by-day
    results = []
    for d in oos_dates:
        day_raw = fullDataSet.loc[fullDataSet[Column.DATE] == d].copy()
        day = apply_denominators(day_raw, denominators, qc_features)  # use TRAIN denominators

        # Compute individual scores (each returns a Series aligned to day.index)
        ifScore = ifqc.score_day(day)
        rzScore = robustZScoreEngine.score_day(day)
        iqrScore = interquartileRangeEngine.score_day(day)
        rollScore = rollingMeanEngine.score_day(day)

        # Aggregate
        day_scores = pd.concat([ifScore, rzScore, rollScore, iqrScore], axis=1)
        aggregatedScore = aggregator.combine(day_scores)
        qc_flag = aggregator.map_to_flag(aggregatedScore).to_frame()

        out = pd.concat([
            day[[Column.TRADE]].reset_index(drop=True),
            pd.Series([d] * len(day), name=Column.DATE),
            day_scores.reset_index(drop=True),
            aggregatedScore.reset_index(drop=True),
            qc_flag.reset_index(drop=True),
        ], axis=1)

        results.append(out)

        # update rolling state AFTER scoring to avoid look-ahead
        rollingMeanEngine.update_state(day)

    oos_scores = pd.concat(results, ignore_index=True)

    # 7) Full export
    out_path = output_handler.export_full_dataset(
        full_data_set=fullDataSet,
        oos_scores=oos_scores,
        input_path=input_path,
        score_cols=Column.DEFAULT_SCORES,
        # (
        #     "IF_score", "RobustZ_score", "Rolling_score", "IQR_score",
        #     "QC_Aggregated", "QC_Flag"
        # ),
        suffix="_with_scores"
    )
    return out_path

if __name__ == "__main__":
    # 1) Ask user for input path
    path = input("Enter full path to PnL_Input.csv: ").strip()
    out_path = run_qc_orchestrator(path, QC_FEATURES)
    print(f"\n=== Full export written ===\n{out_path}")


