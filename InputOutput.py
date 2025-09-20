from typing import Optional, Sequence

import pandas as pd
import ColumnNames as ColumnName
import os

# -----------------------
# 1) Schema declaration
# -----------------------

EXPECTED_COLS = [
    ColumnName.TRADE, ColumnName.DATE,
    ColumnName.START,
    *ColumnName.PNL_SLICES,
    ColumnName.END
]

def _assert_expected_columns(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

# ----------------------------------------
# 2) Input: read + basic preparation
# ----------------------------------------

def read_input(path: str,
               *,
               parse_dates: bool = True,
               date_format: Optional[str] = None) -> pd.DataFrame:
    """
    Read the raw PnL dataset from CSV and perform minimal normalization:
      - parse Date column (optional, default True),
      - enforce schema,
      - engineer standard features (ΔPV, residuals, ratios),
      - sort and reset index.

    Parameters
    ----------
    path : str
        Filesystem path to CSV.
    parse_dates : bool
        If True, parse COL_NAME.DATE with pandas.to_datetime.
    date_format : Optional[str]
        Optional explicit str-time-compatible format.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], format="%Y%m%d")
    _assert_expected_columns(df)

    if parse_dates:
        # If a format is provided, use it; otherwise let pandas infer
        df[ColumnName.DATE] = pd.to_datetime(df[ColumnName.DATE], format=date_format, errors="raise")

    df = engineer_features(df)
    df = df.sort_values([ColumnName.DATE, ColumnName.TRADE]).reset_index(drop=True)
    return df

def read_calcs_csv(path: str, date_format: Optional[str] = "%Y%m%d") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

    """
    Read the CSV and ensure canonical column names.
    Adjust the mapping below if your headers differ slightly.
    """
    df = pd.read_csv(path)

    # Validate presence
    missing = [col for col in EXPECTED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nHave: {list(df.columns)}")

    # Parse date
    if date_format:
        df[ColumnName.DATE] = pd.to_datetime(df[ColumnName.DATE], format=date_format, errors="coerce")
    else:
        df[ColumnName.DATE] = pd.to_datetime(df[ColumnName.DATE], errors="coerce")

    # Basic sanitization
    df = df.dropna(subset=[ColumnName.TRADE, ColumnName.DATE]).copy()
    df[ColumnName.TRADE] = df[ColumnName.TRADE].astype(str).str.strip()

    # Ensure numeric types
    num_cols = [ColumnName.START, *ColumnName.PNL_SLICES, ColumnName.END]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# ----------------------------------------
# 3) Feature engineering (unchanged API)
# ----------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds ΔPV, SumSlices, Residual = ΔPV - SumSlices, and scale-invariant ratios.
    Keeps original columns.
    """
    df = df.copy()
    df[ColumnName.TOTAL] = df[ColumnName.END] - df[ColumnName.START]
    df[ColumnName.EXPLAINED] = df[ColumnName.PNL_SLICES].sum(axis=1)
    df[ColumnName.UNEXPLAINED] = df[ColumnName.TOTAL] - df[ColumnName.EXPLAINED]  # mismatch diagnostic

    eps = 1e-8
    df[ColumnName.TOTAL_JUMP] = df[ColumnName.TOTAL] / (df[ColumnName.START].abs() + eps)
    df[ColumnName.UNEXPLAINED_JUMP] = df[ColumnName.UNEXPLAINED] / (df[ColumnName.START].abs() + eps)
    return df

# ----------------------------------------
# 4) Output helpers (attach + save)
# ----------------------------------------

# _DEFAULT_SCORE_COLS: Sequence[str] = (
#     "IF_score", "RobustZ_score", "Rolling_score", "IQR_score",
#     "QC_Aggregated", "QC_Flag"
# )

def attach_scores(full_data_set: pd.DataFrame,
                  oos_scores: pd.DataFrame,
                  *,
                  score_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Left-join per-trade/day QC scores back to the full dataset.
    OOS rows get populated; in-sample rows will have NaNs in score columns.
    """
    cols = tuple(score_cols) if score_cols is not None else ColumnName.DEFAULT_SCORES
    join_frame = oos_scores[[ColumnName.TRADE, ColumnName.DATE, *cols]].copy()

    merged = full_data_set.merge(
        join_frame,
        on=[ColumnName.TRADE, ColumnName.DATE],
        how="left",
        validate="one_to_one"  # raises if duplicates would create fan-out
    )
    return merged

def derive_output_path(input_path: str,
                       *,
                       suffix: str = "_with_scores",
                       ext: str = ".csv") -> str:
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(os.path.dirname(input_path), f"{base}{suffix}{ext}")

def save_csv(df: pd.DataFrame, out_path: str) -> str:
    # Optional: convert Categorical flag to plain string for friendliness
    # if "QC_Flag" in df.columns and pd.api.types.is_categorical_dtype(df["QC_Flag"]):
    #     df = df.copy()
    #     df["QC_Flag"] = df["QC_Flag"].astype(str)

    df.to_csv(out_path, index=False, date_format="%Y-%m-%d")
    return out_path

def export_full_dataset(full_data_set: pd.DataFrame,
                        oos_scores: pd.DataFrame,
                        input_path: str,
                        *,
                        score_cols: Optional[Sequence[str]] = None,
                        suffix: str = "_with_scores") -> str:
    """
    Convenience wrapper: attach scores and write CSV next to the input file.
    Returns the output path.
    """
    merged = attach_scores(full_data_set, oos_scores, score_cols=score_cols)
    out_path = derive_output_path(input_path, suffix=suffix, ext=".csv")

    # if 'Date' in full_data_set.columns:
    #     full_data_set['Date'] = pd.to_datetime(full_data_set['Date'].astype(str), format="%Y%m%d").dt.date

    return save_csv(merged, out_path)