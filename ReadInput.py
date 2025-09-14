from typing import Optional

import pandas as pd
import ColumnNames as COL_NAME
import os

# -----------------------
# 1) IO + schema checks
# -----------------------

EXPECTED_COLS = [
    COL_NAME.TRADE, COL_NAME.DATE,
    COL_NAME.START,
    *COL_NAME.PNL_SLICES,
    COL_NAME.END
]

def read_calcs_csv(path: str, date_format: Optional[str] = "%Y%m%d") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

    """
    Read the CSV and ensure canonical column names.
    Adjust the mapping below if your headers differ slightly.
    """
    df = pd.read_csv(path)
    # # Try to auto-map common header variants to canonical names:
    # colmap = {
    #     "Trade": "TradeName",
    #     "trade": "TradeName",
    #     "trade_name": "TradeName",
    #     "date": "Date",
    #     "PV Start of a day": "PV_Start",
    #     "PV_Start_of_day": "PV_Start",
    #     "PV start": "PV_Start",
    #     "PV End of a day": "PV_End",
    #     "PV_End_of_day": "PV_End",
    #     "PV end": "PV_End",
    #     "PNL1": "PNL_Slice1",
    #     "PNL2": "PNL_Slice2",
    #     "PNL3": "PNL_Slice3",
    #     "PNL4": "PNL_Slice4",
    #     "Slice1": "PNL_Slice1",
    #     "Slice2": "PNL_Slice2",
    #     "Slice3": "PNL_Slice3",
    #     "Slice4": "PNL_Slice4",
    # }
    # # Apply mapping where applicable
    # df = df.rename(columns={c: colmap.get(c, c) for c in df.columns})

    # Validate presence
    missing = [col for col in EXPECTED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nHave: {list(df.columns)}")

    # Parse date
    if date_format:
        df[COL_NAME.DATE] = pd.to_datetime(df[COL_NAME.DATE], format=date_format, errors="coerce")
    else:
        df[COL_NAME.DATE] = pd.to_datetime(df[COL_NAME.DATE], errors="coerce")

    # Basic sanitization
    df = df.dropna(subset=[COL_NAME.TRADE, COL_NAME.DATE]).copy()
    df[COL_NAME.TRADE] = df[COL_NAME.TRADE].astype(str).str.strip()

    # Ensure numeric types
    num_cols = [COL_NAME.START, *COL_NAME.PNL_SLICES, COL_NAME.END]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# ---------------------------------------
# 2) Feature engineering (QC-oriented)
# ---------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds ΔPV, SumSlices, Residual = ΔPV - SumSlices, and safe ratios.
    Keeps original columns.
    """
    df = df.copy()
    df[COL_NAME.TOTAL] = df[COL_NAME.END] - df[COL_NAME.START]
    df["SumSlices"] = df[COL_NAME.PNL_SLICES].sum(axis=1)
    df[COL_NAME.UNEXPLAINED] = df[COL_NAME.TOTAL] - df["SumSlices"]  # mismatch diagnostic

    # Scale-invariant ratios (guard against division by zero/near-zero)
    eps = 1e-8
    df[COL_NAME.TOTAL_JUMP] = df[COL_NAME.TOTAL] / (df[COL_NAME.START].abs() + eps)
    df[COL_NAME.UNEXPLAINED_JUMP] = df[COL_NAME.UNEXPLAINED] / (df[COL_NAME.START].abs() + eps)

    return df