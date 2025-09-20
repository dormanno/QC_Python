import numpy as np
import pandas as pd
from typing import List

import InputOutput as IO
import ColumnNames as Column

from QC_methods import IsolationForestQC, RobustZQC, IQRQC, RollingZQC
from Aggregator import ScoreAggregator

# ----------------------------
# Config
# ----------------------------
TRAIN_DAYS = 60

ROBUST_Z_FEATURES = [
    Column.START, *Column.PNL_SLICES, Column.TOTAL, Column.EXPLAINED, Column.UNEXPLAINED
]
IQR_FEATURES = ROBUST_Z_FEATURES
ROLL_FEATURES = ROBUST_Z_FEATURES
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

if __name__ == "__main__":
    # 1) Ask user for input path
    path = input("Enter full path to PnL_Input.csv: ").strip()
    fullDataSet = IO.read_input(path)
    fullDataSet = fullDataSet.sort_values(Column.DATE)

    print("\n=== 1. prepared data frame first and last rows  ===")

    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(fullDataSet.head(5).to_string(index=False))
        print("...")
        print(fullDataSet.tail(5).to_string(index=False))

    # 2) Split train / OOS by date
    print("\n=== 2. Segregating training and out-of-sample data sets  ===")
    dates = fullDataSet[Column.DATE].drop_duplicates().sort_values().to_list()
    if len(dates) <= TRAIN_DAYS:
        raise ValueError("Not enough data: need > TRAIN_DAYS unique dates for walk-forward.")

    cutoff = dates[TRAIN_DAYS - 1]
    rawTrainDataSet = fullDataSet.loc[fullDataSet[Column.DATE] <= cutoff].copy()
    oos_dates = dates[TRAIN_DAYS:]

    # 3) Stabilize scale (optional but recommended)
    level_feats = [Column.START, Column.END, *Column.PNL_SLICES, Column.TOTAL, Column.EXPLAINED, Column.UNEXPLAINED]
    denominators = compute_per_trade_denominators(rawTrainDataSet, level_feats)
    trainDataSet = apply_denominators(rawTrainDataSet, denominators, level_feats)


    print(f"First {len(trainDataSet)} rows selected as training set rest will be used as out-of-sample")

    print("\n=== 3. Initializing QC engines...  ===")

    # 4) Instantiate QC methods and fit on TRAIN only
    ifqc = IsolationForestQC(
        mode="time_series",
        per_trade_normalize=False,  # we normalize in the orchestrator using TRAIN denominators
        use_robust_scaler=True,
        n_estimators=200,
        max_samples=256,
        contamination=0.01,
    )
    ifqc.fit(trainDataSet)
    print("Isolation Forest QC initialized")

    robustZScoreEngine = RobustZQC(features=ROBUST_Z_FEATURES)
    robustZScoreEngine.fit(trainDataSet)
    print("Robust Z-Score QC initialized")

    interquartileRangeEngine = IQRQC(features=IQR_FEATURES)
    interquartileRangeEngine.fit(trainDataSet)
    print("Interquartile Range QC initialized")

    rollingMeanEngine = RollingZQC(features=ROLL_FEATURES, window=ROLL_WINDOW)
    rollingMeanEngine.fit(trainDataSet)  # warm-up buffers
    print("Rolling mean QC initialized")

    print("All QC engines initialized")

    aggregator = ScoreAggregator(w_if=0.4, w_rz=0.3, w_roll=0.2, w_iqr=0.1)

    # 5) Iterate OOS day-by-day
    print("\n=== 4. Iterating through OOS day-by-day calculating scores...  ===")
    results = []
    for d in oos_dates:
        day_raw = fullDataSet.loc[fullDataSet[Column.DATE] == d].copy()
        day = apply_denominators(day_raw, denominators, level_feats)  # use TRAIN denominators

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

    print(f"Scores for all {len(oos_dates)} dates calculated")
    oos_scores = pd.concat(results, ignore_index=True)

    # print("\n=== 5a. Flag counts (GREEN/AMBER/RED) ===")
    # print(oos_scores["QC_Flag"].value_counts(sort=False))

    # 6) Example report
    print("\n=== 5. OOS 10 worst per-trade (by QC_Aggregated) ===")
    oos_scores = oos_scores.sort_values([Column.AGGREGATED_SCORE], ascending=[False])

    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(oos_scores.head(10).to_string(index=False))

    # ---- FULL EXPORT ----
    out_path = IO.export_full_dataset(
        full_data_set=fullDataSet,
        oos_scores=oos_scores,
        input_path=path,
        score_cols=Column.DEFAULT_SCORES,
        # (
        #     "IF_score", "RobustZ_score", "Rolling_score", "IQR_score",
        #     "QC_Aggregated", "QC_Flag"
        # ),
        suffix="_with_scores"
    )
    print(f"\n=== Full export written ===\n{out_path}")


