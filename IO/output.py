import os
from typing import Optional, Sequence

import pandas as pd
from column_names import main_column, qc_column


class Output:
    """Handles writing QC results to CSV files."""
    
    def attach_scores(self, full_data_set: pd.DataFrame,
                      oos_scores: pd.DataFrame,
                      *,
                      score_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """
        Left-join per-trade/day QC scores back to the full dataset.
        OOS rows get populated; in-sample rows will have NaNs in score columns.
        """
        cols = tuple(score_cols) if score_cols is not None else qc_column.SCORE_COLUMNS
        join_frame = oos_scores[[main_column.TRADE, main_column.DATE, *cols]].copy()

        merged = full_data_set.merge(
            join_frame,
            on=[main_column.TRADE, main_column.DATE],
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
