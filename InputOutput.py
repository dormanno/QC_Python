from typing import Optional, Sequence

import pandas as pd
import ColumnNames as ColumnName
import os

class Input:
    """Abstract base class for reading and processing input data.
    
    Provides common functionality for reading CSV files and basic data preparation.
    Subclasses should override EXPECTED_COLS, NUMERIC_COLS, and input_post_process() 
    for specific data types.
    """
    
    EXPECTED_COLS = []  # To be overridden by subclasses
    NUMERIC_COLS = []   # To be overridden by subclasses
    DATE_FORMAT = "%Y%m%d"  # New

    def _assert_expected_columns(self, df: pd.DataFrame) -> None:
        """Validate that all required columns are present in the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to validate.
        
        Raises:
            ValueError: If any required columns are missing.
        """
        missing = [c for c in self.EXPECTED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def read_input(self, path: str,
                *,
                parse_dates: bool = True,
                date_format: Optional[str] = None) -> pd.DataFrame:
        """
        Read dataset from CSV and perform normalization:
        - parse Date column (optional, default True),
        - enforce schema,
        - ensure numeric types,
        - engineer features (subclass-specific),
        - sort and reset index.

        Parameters
        ----------
        path : str
            Filesystem path to CSV.
        parse_dates : bool
            If True, parse date column with pandas.to_datetime.
        date_format : Optional[str]
            Optional explicit str-time-compatible format.

        Returns
        -------
        pd.DataFrame
            Processed DataFrame with engineered features.
        """
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'], format="%Y%m%d")
        self._assert_expected_columns(df)

        if parse_dates and ColumnName.DATE in df.columns:
            df[ColumnName.DATE] = pd.to_datetime(df[ColumnName.DATE], format=self.DATE_FORMAT)

        # Basic sanitization
        df = df.dropna(subset=[ColumnName.TRADE, ColumnName.DATE, ColumnName.TRADE_TYPE]).copy()
        df[ColumnName.TRADE] = df[ColumnName.TRADE].astype(str).str.strip()

        # Ensure numeric types
        for c in self.NUMERIC_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Post-process with feature engineering (subclass-specific)
        df = self.input_post_process(df)
        
        # Sort and reset
        df = df.sort_values([ColumnName.DATE, ColumnName.TRADE]).reset_index(drop=True)
        return df

    def input_post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process input data with feature engineering.
        
        To be implemented by subclasses.
        
        Args:
            df (pd.DataFrame): Input DataFrame with raw data.
        
        Returns:
            pd.DataFrame: DataFrame with engineered features.
        """
        return df  # Default: return unchanged


class PnlInput(Input):
    """Handles reading and engineering PnL-specific input data.
    
    Extends Input class with PnL-specific column expectations and feature engineering.
    """
    
    EXPECTED_COLS = [
        ColumnName.TRADE, 
        ColumnName.DATE,
        ColumnName.TRADE_TYPE,
        ColumnName.START,
        *ColumnName.PNL_SLICES,
        ColumnName.END
    ]
    NUMERIC_COLS = ColumnName.PNL_INPUT_FEATURES

    def input_post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process PnL data with feature engineering.
        
        Adds ΔPV, SumSlices, Residual = ΔPV - SumSlices, and scale-invariant ratios.
        
        Args:
            df (pd.DataFrame): Input DataFrame with raw PnL data.
        
        Returns:
            pd.DataFrame: DataFrame with additional engineered PnL feature columns.
        """
        return self.engineer_PnL_features(df)

    def engineer_PnL_features(self, df: pd.DataFrame) -> pd.DataFrame:
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


class Output:
    def attach_scores(self, full_data_set: pd.DataFrame,
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

    def derive_output_path(self, input_path: str,
                           *,
                           suffix: str = "_with_scores",
                           ext: str = ".csv") -> str:
        base = os.path.splitext(os.path.basename(input_path))[0]
        return os.path.join(os.path.dirname(input_path), f"{base}{suffix}{ext}")

    def save_csv(self, df: pd.DataFrame, out_path: str) -> str:
        # Optional: convert Categorical flag to plain string for friendliness
        # if "QC_Flag" in df.columns and pd.api.types.is_categorical_dtype(df["QC_Flag"]):
        #     df = df.copy()
        #     df["QC_Flag"] = df["QC_Flag"].astype(str)

        df.to_csv(out_path, index=False, date_format="%Y-%m-%d")
        return out_path

    def export_full_dataset(self, full_data_set: pd.DataFrame,
                            oos_scores: pd.DataFrame,
                            input_path: str,
                            *,
                            score_cols: Optional[Sequence[str]] = None,
                            suffix: str = "_with_scores") -> str:
        """
        Convenience wrapper: attach scores and write CSV next to the input file.
        Returns the output path.
        """
        merged = self.attach_scores(full_data_set, oos_scores, score_cols=score_cols)
        out_path = self.derive_output_path(input_path, suffix=suffix, ext=".csv")

        return self.save_csv(merged, out_path)
