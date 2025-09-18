import ReadInput
import IsolationForestRunner as IF_Runner

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
    ts_result = IF_Runner.run_time_series_iforest(
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
