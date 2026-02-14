import unittest
import os
import tempfile
import shutil
import pandas as pd
from IO.input import PnLInput
from column_names import main_column
from Tests.outlier_injector import OutlierInjector

ORIGINAL_INPUT_DIRECTORY = r"C:\Users\dorma\Documents\UEK_Backup\Test"


class TestOutlierInjector(unittest.TestCase):

    def test_outlier_injections(self):
        """Validate that outlier injections are applied correctly to OOS data."""
        original_input_file = "PnL_Input_Train-OOS.csv"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_path = os.path.join(temp_dir, original_input_file)
            shutil.copy2(os.path.join(ORIGINAL_INPUT_DIRECTORY, original_input_file), temp_input_path)

            input_handler = PnLInput()
            original_df = input_handler.read_and_validate(
                temp_input_path,
                split_identifier=main_column.TRADE_TYPE
            )

            injector = OutlierInjector()
            injected_df = injector.inject(original_df)

            # 1) Shape should be identical
            self.assertEqual(
                original_df.shape,
                injected_df.shape,
                "Injected dataset shape differs from original"
            )

            # 2) Original should contain only Train and OOS labels
            if main_column.RECORD_TYPE in original_df.columns:
                allowed_labels = {"Train", "OOS"}
                original_labels = set(original_df[main_column.RECORD_TYPE].astype(str).unique())
                self.assertTrue(
                    original_labels.issubset(allowed_labels),
                    f"Original dataset has unexpected RecordType labels: {sorted(original_labels - allowed_labels)}"
                )

            # 3) All injection scenarios should be present
            expected_scenarios = {
                "Injected_PV_Spike",
                "Injected_PV_Step",
                "Injected_PV_Stale",
                "Injected_PV_Scale",
                "Injected_PV_SignFlip",
                "Injected_Slice_Spike",
                "Injected_Slice_Stale",
                "Injected_Reallocation",
                "Injected_IdentityBreak",
                "Injected_CrossFamily",
            }

            injected_labels = set(
                injected_df[main_column.RECORD_TYPE]
                .astype(str)
                .unique()
            )

            missing = expected_scenarios - injected_labels
            self.assertFalse(
                missing,
                f"Missing injection scenarios: {sorted(missing)}"
            )

            # 4) No Train rows should be changed
            if main_column.RECORD_TYPE in original_df.columns:
                train_mask = original_df[main_column.RECORD_TYPE] == "Train"
                self.assertTrue(
                    train_mask.any(),
                    "No Train rows found in dataset to validate"
                )

                pd.testing.assert_frame_equal(
                    original_df.loc[train_mask].reset_index(drop=True),
                    injected_df.loc[train_mask].reset_index(drop=True),
                    check_dtype=False
                )

            # 5) All injected rows must be OOS
            injected_mask = injected_df[main_column.RECORD_TYPE].astype(str).str.startswith("Injected_")
            self.assertTrue(
                injected_mask.any(),
                "No injected rows found in dataset"
            )
            self.assertTrue(
                (original_df.loc[injected_mask, main_column.RECORD_TYPE] != "Train").all(),
                "Some Train rows were injected"
            )

    def test_outlier_injections_export_enriched(self):
        """Export enriched file with original values and *_injected columns for review."""
        original_input_file = "PnL_Input_Train-OOS.csv"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_path = os.path.join(temp_dir, original_input_file)
            shutil.copy2(os.path.join(ORIGINAL_INPUT_DIRECTORY, original_input_file), temp_input_path)

            input_handler = PnLInput()
            original_df = input_handler.read_and_validate(
                temp_input_path,
                split_identifier=main_column.TRADE_TYPE
            )

            injector = OutlierInjector()
            injected_df = injector.inject(original_df)

            # Build enriched dataset: updated RecordType + original values + *_injected columns
            enriched_df = original_df.copy()
            if main_column.RECORD_TYPE in injected_df.columns:
                enriched_df[main_column.RECORD_TYPE] = injected_df[main_column.RECORD_TYPE]

            injected_columns = []
            for col in original_df.columns:
                if col == main_column.RECORD_TYPE:
                    continue
                injected_col = f"{col}_injected"
                injected_columns.append(injected_col)
                changed_mask = injected_df[col] != original_df[col]
                enriched_df[injected_col] = injected_df[col].where(changed_mask, pd.NA)

            # Basic validations
            self.assertEqual(
                enriched_df.shape[0],
                original_df.shape[0],
                "Enriched dataset row count differs from original"
            )
            self.assertTrue(
                all(col in enriched_df.columns for col in injected_columns),
                "Missing *_injected columns in enriched dataset"
            )

            # Ensure at least one injected value is present
            any_injected = enriched_df[injected_columns].notna().any(axis=None)
            self.assertTrue(any_injected, "No injected values found in *_injected columns")

            # Export for review
            output_name = os.path.splitext(original_input_file)[0] + "_injected.csv"
            output_path = os.path.join(temp_dir, output_name)
            enriched_df.to_csv(output_path, index=False)
            self.assertTrue(os.path.exists(output_path), f"Output file not created: {output_path}")

            # Copy output to original directory for review
            shutil.copy2(output_path, os.path.join(ORIGINAL_INPUT_DIRECTORY, output_name))


if __name__ == '__main__':
    unittest.main()
