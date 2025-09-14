# IF_Test.py (refactored)
# ------------------------------------------------------------
# Purpose: separate TS and CS Isolation Forest checks into functions,
# leaving __main__ for orchestration.
# ------------------------------------------------------------

from typing import Optional, List
import pandas as pd

import ColumnNames as COL_NAME
import IsolationForest as IF
import ReadInput


# ----------------------------
# Time-series (per-trade) QC
# ----------------------------
def run_time_series_iforest(
    df: pd.DataFrame,
    n_estimators: int = 200,
    max_samples: int = 256,
    contamination: Optional[float] = 0.01,
    per_trade_normalize: bool = True,
    use_robust_scaler: bool = True,
    top_quantile: float = 0.99,
) -> pd.DataFrame:
    """
    Fits iForest on a (rolling) history and scores all rows in df.
    Returns a DataFrame with identifiers and anomaly scores; also marks
    top anomalies by the given quantile.

    Notes:
    - This is the recommended mode for small cross-sections (e.g., 4 trades).
    """
    ts_input = IF.build_iforest_matrix(
        df,
        mode="time_series",
        per_trade_normalize=per_trade_normalize,
        use_robust_scaler=use_robust_scaler,
    )
    ts_clf = IF.fit_isolation_forest(
        ts_input.X,
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
    )
    ts_scores = IF.score_isolation_forest(ts_clf, ts_input.X)
    ts_result = pd.concat([ts_input.ids.reset_index(drop=True), ts_scores], axis=1)

    # Flag top anomalies by intensity (monotone 0..1, higher=worse)
    q = ts_result["anomaly_intensity"].quantile(top_quantile)
    ts_result["flag_top"] = ts_result["anomaly_intensity"] >= q
    return ts_result


# ---------------------------------
# Cross-sectional (per-day) QC
# ---------------------------------
def run_cross_sectional_iforest(
    df: pd.DataFrame,
    n_estimators: int = 200,
    max_samples: int = 256,
    contamination: Optional[float] = 0.02,
    per_trade_normalize: bool = False,   # IMPORTANT: off by default for CS
    use_robust_scaler: bool = True,
    min_obs_per_day: int = 10,
    top_k_per_day: int = 2,
    dates: Optional[List[pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    For each Date, fit iForest on that day's cross-section (trades) and
    return the top-k most anomalous trades per day.

    Caveat:
    - With very small cross-sections (e.g., only 4 trades), this mode is
      statistically weak; prefer the time-series function above.
    """
    flags = []
    grouped = df.groupby(COL_NAME.DATE)

    # Optionally restrict to specific dates
    items = grouped if dates is None else [(d, grouped.get_group(d)) for d in dates if d in grouped.groups]

    for d, g in items:
        # Skip tiny cross-sections
        if len(g) < min_obs_per_day:
            continue

        cs_input = IF.build_iforest_matrix(
            g,
            mode="cross_sectional",
            per_trade_normalize=per_trade_normalize,  # keep False unless you pass history-based denominators
            use_robust_scaler=use_robust_scaler,
        )
        cs_clf = IF.fit_isolation_forest(
            cs_input.X,
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
        )
        cs_scores = IF.score_isolation_forest(cs_clf, cs_input.X)
        cs_res = pd.concat([cs_input.ids.reset_index(drop=True), cs_scores], axis=1)

        # Keep the worst offenders for that day
        worst = cs_res.nlargest(top_k_per_day, "anomaly_intensity")
        flags.append(worst)

    return pd.concat(flags, ignore_index=True) if flags else pd.DataFrame(
        columns=[COL_NAME.TRADE, COL_NAME.DATE, "decision_function", "sklearn_label", "anomaly_intensity"]
    )


# ----------------------------
# High-level orchestration
# ----------------------------
if __name__ == "__main__":
    # 1) Path to input CSV (adjust as needed)
    path = "C://Users//dorma//OneDrive - uek.krakow.pl//ComputationalFinance//UBS//PnL_Input.csv"

    # 2) Read + engineer features
    df = ReadInput.read_calcs_csv(path)
    df = ReadInput.engineer_features(df)

    # 3) Time-series QC (recommended when cross-section is small)
    ts_result = run_time_series_iforest(
        df,
        n_estimators=200,
        max_samples=256,
        contamination=0.01,
        per_trade_normalize=True,
        use_robust_scaler=True,
        top_quantile=0.99,
    )

    # # 4) Cross-sectional QC (skip if daily cross-section is tiny)
    # cs_flags = run_cross_sectional_iforest(
    #     df,
    #     n_estimators=200,
    #     max_samples=256,
    #     contamination=0.02,
    #     per_trade_normalize=False,   # keep False to preserve CS signal
    #     use_robust_scaler=True,
    #     min_obs_per_day=10,
    #     top_k_per_day=2,
    # )

    # 5) Output / routing
    print("\n=== Time-series results (top 1% marked) ===")
    print(ts_result.loc[ts_result["flag_top"]].sort_values("anomaly_intensity", ascending=False).head(20))

    # print("\n=== Cross-sectional daily worst cases ===")
    # print(cs_flags.head(20))
