import numpy as np
import pandas as pd
from typing import List

import ReadInput
import ColumnNames as COL_NAME

from QC_methods import IsolationForestQC, RobustZQC, IQRQC, RollingZQC
from Aggregator import ScoreAggregator

# ----------------------------
# Config
# ----------------------------
TRAIN_DAYS = 60



ROBUST_Z_FEATURES = [
    COL_NAME.START, *COL_NAME.PNL_SLICES, COL_NAME.TOTAL, "SumSlices", COL_NAME.UNEXPLAINED
]
IQR_FEATURES = ROBUST_Z_FEATURES
ROLL_FEATURES = ROBUST_Z_FEATURES
ROLL_WINDOW = 20

# ----------------------------
# Normalization across trades
# ----------------------------

def compute_per_trade_denominators(train_df: pd.DataFrame, level_feats: List[str]) -> pd.DataFrame:
    den = (train_df.groupby(COL_NAME.TRADE)[level_feats]
           .apply(lambda g: g.abs().median())
           .reset_index())
    den.columns = [COL_NAME.TRADE] + [f"{c}__den" for c in level_feats]
    return den

def apply_denominators(df: pd.DataFrame, den: pd.DataFrame, level_feats: List[str]) -> pd.DataFrame:
    if den is None:
        return df
    g = df.merge(den, on=COL_NAME.TRADE, how="left")
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
    df = ReadInput.read_calcs_csv(path)
    df = ReadInput.engineer_features(df)
    df = df.sort_values(COL_NAME.DATE)

    # 2) Split train / OOS by date
    dates = df[COL_NAME.DATE].drop_duplicates().sort_values().to_list()
    if len(dates) <= TRAIN_DAYS:
        raise ValueError("Not enough data: need > TRAIN_DAYS unique dates for walk-forward.")

    cutoff = dates[TRAIN_DAYS - 1]
    train_raw = df.loc[df[COL_NAME.DATE] <= cutoff].copy()
    oos_dates = dates[TRAIN_DAYS:]

    # 3) Stabilize scale (optional but recommended)
    level_feats = [COL_NAME.START, COL_NAME.END, *COL_NAME.PNL_SLICES, COL_NAME.TOTAL, "SumSlices", COL_NAME.UNEXPLAINED]
    denoms = compute_per_trade_denominators(train_raw, level_feats)
    train = apply_denominators(train_raw, denoms, level_feats)

    # 4) Instantiate QC methods and fit on TRAIN only
    ifqc = IsolationForestQC(
        mode="time_series",
        per_trade_normalize=False,  # we normalize in the orchestrator using TRAIN denominators
        use_robust_scaler=True,
        n_estimators=200,
        max_samples=256,
        contamination=0.01,
    )
    ifqc.fit(train)

    rzqc = RobustZQC(features=ROBUST_Z_FEATURES)
    rzqc.fit(train)

    iqrqc = IQRQC(features=IQR_FEATURES)
    iqrqc.fit(train)

    rollqc = RollingZQC(features=ROLL_FEATURES, window=ROLL_WINDOW)
    rollqc.fit(train)  # warm-up buffers

    aggregator = ScoreAggregator(w_if=0.4, w_rz=0.3, w_roll=0.2, w_iqr=0.1)

    # 5) Iterate OOS day-by-day
    results = []
    for d in oos_dates:
        day_raw = df.loc[df[COL_NAME.DATE] == d].copy()
        day = apply_denominators(day_raw, denoms, level_feats)  # use TRAIN denominators

        # Compute individual scores (each returns a Series aligned to day.index)
        s_if = ifqc.score_day(day)
        s_rz = rzqc.score_day(day)
        s_iqr = iqrqc.score_day(day)
        s_roll = rollqc.score_day(day)

        # Aggregate
        day_scores = pd.concat([s_if, s_rz, s_roll, s_iqr], axis=1)
        s_agg = aggregator.combine(day_scores)

        out = pd.concat([
            day[[COL_NAME.TRADE]].reset_index(drop=True),
            pd.Series([d] * len(day), name=COL_NAME.DATE),
            day_scores.reset_index(drop=True),
            s_agg.reset_index(drop=True),
        ], axis=1)

        results.append(out)

        # update rolling state AFTER scoring to avoid look-ahead
        rollqc.update_state(day)

    oos_scores = pd.concat(results, ignore_index=True)

    # 6) Example report
    print("\n=== OOS daily worst per-trade (by QC_Aggregated) ===")
    print(oos_scores.sort_values([COL_NAME.DATE, "QC_Aggregated"], ascending=[True, False]).head(40))
