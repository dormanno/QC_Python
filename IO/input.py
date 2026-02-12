from typing import Optional
import logging

import pandas as pd
from column_names import main_column, pnl_column, cds_column, cdi_column

logger = logging.getLogger(__name__)


class Input:
    """Abstract base class for reading and processing input data.
    
    Provides common functionality for reading CSV files and basic data preparation.
    Subclasses should override EXPECTED_COLS, NUMERIC_COLS, and input_post_process() 
    for specific data types.
    """
    
    EXPECTED_COLS = []  # To be overridden by subclasses
    NUMERIC_COLS = []   # To be overridden by subclasses
    DATE_FORMAT = "%Y%m%d"

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

        if main_column.RECORD_TYPE in df.columns:
            record_col = df[main_column.RECORD_TYPE]
            if record_col.isna().any() or record_col.astype(str).str.strip().eq("").any():
                raise ValueError("RecordType must be defined for all records when column is present")

        if parse_dates and main_column.DATE in df.columns:
            df[main_column.DATE] = pd.to_datetime(df[main_column.DATE], format=self.DATE_FORMAT)

        # Basic sanitization
        df = df.dropna(subset=[main_column.TRADE, main_column.DATE, main_column.TRADE_TYPE]).copy()
        df[main_column.TRADE] = df[main_column.TRADE].astype(str).str.strip()

        # Ensure numeric types
        for c in self.NUMERIC_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Post-process with feature engineering (subclass-specific)
        df = self.input_post_process(df)
        
        # Sort and reset
        df = df.sort_values([main_column.DATE, main_column.TRADE]).reset_index(drop=True)
        return df

    def read_and_validate(self, path: str, split_identifier: Optional[str] = None) -> pd.DataFrame:
        """Read, sort by date, and validate input data.
        
        Performs read_input, sorts by DATE, and optionally validates split_identifier column.
        
        Args:
            path (str): Path to input CSV file.
            split_identifier (Optional[str]): Column name to validate as split identifier.
                If provided, validates that the column exists and has no missing values.
        
        Returns:
            pd.DataFrame: Processed, sorted, and validated DataFrame.
        
        Raises:
            ValueError: If data validation fails.
        """
        # Read and process input
        df = self.read_input(path)
        
        # Ensure sorted by date
        df = df.sort_values(main_column.DATE).reset_index(drop=True)
        logger.info(f"Loaded {len(df)} rows with {df[main_column.DATE].nunique()} unique dates")
        
        # Validate split_identifier column if defined
        if split_identifier is not None:
            if split_identifier not in df.columns:
                raise ValueError(
                    f"split_identifier column '{split_identifier}' not found in dataset. "
                    f"Available columns: {list(df.columns)}"
                )
            missing_count = df[split_identifier].isna().sum()
            if missing_count > 0:
                raise ValueError(
                    f"split_identifier column '{split_identifier}' has {missing_count} "
                    f"missing values. Every row must have a value."
                )
        
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


class CreditDeltaSingleInput(Input):
    """Handles reading and engineering Credit Delta Single-specific input data.
    
    Extends Input class with Credit Delta Single-specific column expectations and feature engineering.
    """
    
    EXPECTED_COLS = [
        main_column.TRADE, 
        main_column.DATE,
        main_column.TRADE_TYPE,
        cds_column.CREDIT_DELTA_SINGLE
    ]
    NUMERIC_COLS = cds_column.INPUT_FEATURES

    def input_post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process Credit Delta Single data with feature engineering.
        
        Currently no additional features; placeholder for future extensions.
        
        Args:
            df (pd.DataFrame): Input DataFrame with raw Credit Delta Single data.
        
        Returns:
            pd.DataFrame: DataFrame with engineered Credit Delta Single feature columns.
        """
        return df  # No additional processing for now

    
class CreditDeltaIndexInput(Input):
    """Handles reading and engineering Credit Delta Index-specific input data.
    
    Extends Input class with Credit Delta Index-specific column expectations and feature engineering.
    """
    
    EXPECTED_COLS = [
        main_column.TRADE, 
        main_column.DATE,
        main_column.TRADE_TYPE,
        cdi_column.CREDIT_DELTA_INDEX
    ]
    NUMERIC_COLS = cdi_column.INPUT_FEATURES

    def input_post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process Credit Delta Index data with feature engineering.
        
        Currently no additional features; placeholder for future extensions.
        
        Args:
            df (pd.DataFrame): Input DataFrame with raw Credit Delta Index data.
        
        Returns:
            pd.DataFrame: DataFrame with engineered Credit Delta Index feature columns.
        """
        return df  # No additional processing for now


class PnLInput(Input):
    """Handles reading and engineering PnL-specific input data.
    
    Extends Input class with PnL-specific column expectations and feature engineering.
    """
    
    EXPECTED_COLS = [
        main_column.TRADE, 
        main_column.DATE,
        main_column.TRADE_TYPE,
        pnl_column.START,
        *pnl_column.SLICE_COLUMNS,
        pnl_column.END
    ]
    NUMERIC_COLS = pnl_column.INPUT_FEATURES

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
        df[pnl_column.TOTAL] = df[pnl_column.END] - df[pnl_column.START]
        df[pnl_column.EXPLAINED] = df[pnl_column.SLICE_COLUMNS].sum(axis=1)
        df[pnl_column.UNEXPLAINED] = df[pnl_column.TOTAL] - df[pnl_column.EXPLAINED]  # mismatch diagnostic

        eps = 1e-8
        df[pnl_column.TOTAL_JUMP] = df[pnl_column.TOTAL] / (df[pnl_column.START].abs() + eps)
        df[pnl_column.UNEXPLAINED_JUMP] = df[pnl_column.UNEXPLAINED] / (df[pnl_column.START].abs() + eps)
        return df
