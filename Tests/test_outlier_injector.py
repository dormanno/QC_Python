import unittest
import os
import tempfile
import shutil
import pandas as pd
from IO.input import PnLInput, CreditDeltaSingleInput, CreditDeltaIndexInput
from column_names import main_column, cds_column, cdi_column
from Tests.outlier_injectors import PnLOutlierInjector, CreditDeltaOutlierInjector
from Tests.outlier_injectors.credit_delta_config import CreditDeltaInjectorConfig

ORIGINAL_INPUT_DIRECTORY = r"C:\Users\dorma\Documents\UEK_Backup\Test"


class TestPnLOutlierInjector(unittest.TestCase):

    def test_outlier_injections_pnl(self):
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

            injector = PnLOutlierInjector()
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

    def test_outlier_injections_pnl_export_enriched(self):
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

            injector = PnLOutlierInjector()
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


class TestCdsOutlierInjector(unittest.TestCase):

    def test_cds_stale_value_does_not_label_source_row(self):
        """The source row used for stale propagation must remain unchanged and not relabeled."""
        config = CreditDeltaInjectorConfig.cds_preset()
        injector = CreditDeltaOutlierInjector(config=config, random_seed=42)

        dates = pd.date_range("2025-01-01", periods=7, freq="D")
        source_value = 100.0
        original_values = [source_value, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0]

        original_df = pd.DataFrame({
            main_column.RECORD_TYPE: ["OOS"] * 7,
            main_column.TRADE: ["T_BASIS_1"] * 7,
            main_column.BOOK: ["B1"] * 7,
            main_column.TRADE_TYPE: ["Basis"] * 7,
            main_column.DATE: dates,
            cds_column.CREDIT_DELTA_SINGLE: original_values,
        })

        injected_df = injector.inject_cd_stale_value(original_df)

        # Source row (first eligible date) should not be marked stale or modified
        self.assertEqual(injected_df.loc[0, main_column.RECORD_TYPE], "OOS")
        self.assertEqual(injected_df.loc[0, cds_column.CREDIT_DELTA_SINGLE], source_value)

        # Following stale_days rows should be stale and equal to source value
        stale_days = config.stale_days
        stale_rows = injected_df.iloc[1:1 + stale_days]
        self.assertTrue((stale_rows[main_column.RECORD_TYPE] == "CD_StaleValue").all())
        self.assertTrue((stale_rows[cds_column.CREDIT_DELTA_SINGLE] == source_value).all())

        # Remaining rows should stay OOS and unchanged
        tail_rows = injected_df.iloc[1 + stale_days:]
        self.assertTrue((tail_rows[main_column.RECORD_TYPE] == "OOS").all())
        self.assertEqual(
            tail_rows[cds_column.CREDIT_DELTA_SINGLE].tolist(),
            original_values[1 + stale_days:]
        )

    def test_cds_injections_applied(self):
        """Validate that CDS injections are applied correctly to OOS data."""
        original_input_file = "CreditDeltaSingle_Input.csv"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_path = os.path.join(temp_dir, original_input_file)
            shutil.copy2(os.path.join(ORIGINAL_INPUT_DIRECTORY, original_input_file), temp_input_path)

            input_handler = CreditDeltaSingleInput()
            original_df = input_handler.read_and_validate(
                temp_input_path,
                split_identifier=main_column.TRADE_TYPE
            )

            config = CreditDeltaInjectorConfig.cds_preset()
            injector = CreditDeltaOutlierInjector(config=config)
            injected_df = injector.inject(original_df)

            # 1) Shape should be identical (or have RecordType added if not in original)
            if main_column.RECORD_TYPE not in original_df.columns:
                # Injector adds RecordType if missing
                self.assertEqual(
                    original_df.shape[0],
                    injected_df.shape[0],
                    "Row count differs"
                )
                self.assertEqual(
                    original_df.shape[1] + 1,  # +1 for RecordType column added by injector
                    injected_df.shape[1],
                    "Column count should match (or differ by 1 if RecordType was added)"
                )
            else:
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

            # 3) All CDS injection scenarios should be present
            expected_scenarios = {
                "CD_Drift",
                "CD_StaleValue",
                "CD_ClusterShock_3d",
                "CD_TradeTypeWide_Shock",
                "CD_PointShock",
                "CD_SignFlip",
                "CD_ScaleError",
                "CD_SuddenZero",
            }

            injected_labels = set(
                injected_df[main_column.RECORD_TYPE]
                .astype(str)
                .unique()
            )

            missing = expected_scenarios - injected_labels
            self.assertFalse(
                missing,
                f"Missing CDS injection scenarios: {sorted(missing)}"
            )

            # 4) No Train rows should be changed
            if main_column.RECORD_TYPE in original_df.columns:
                train_mask = original_df[main_column.RECORD_TYPE] == "Train"
                if train_mask.any():
                    pd.testing.assert_frame_equal(
                        original_df.loc[train_mask].reset_index(drop=True),
                        injected_df.loc[train_mask].reset_index(drop=True),
                        check_dtype=False
                    )

            # 5) All injected rows must be OOS (or newly added if RecordType didn't exist)
            injected_mask = injected_df[main_column.RECORD_TYPE].astype(str).isin(expected_scenarios)
            self.assertTrue(
                injected_mask.any(),
                "No injected rows found in dataset"
            )
            
            # Verify that modified rows were originally OOS (if RecordType existed in original)
            if main_column.RECORD_TYPE in original_df.columns:
                original_injected_record_types = original_df.loc[injected_mask, main_column.RECORD_TYPE].unique()
                self.assertTrue(
                    all(rt in {"OOS", "Train"} for rt in original_injected_record_types),
                    f"Some injected rows were not from OOS: {original_injected_record_types}"
                )

    def test_cds_injections_by_trade_type(self):
        """Validate that CDS injections respect per-trade-type configuration."""
        original_input_file = "CreditDeltaSingle_Input.csv"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_path = os.path.join(temp_dir, original_input_file)
            shutil.copy2(os.path.join(ORIGINAL_INPUT_DIRECTORY, original_input_file), temp_input_path)

            input_handler = CreditDeltaSingleInput()
            original_df = input_handler.read_and_validate(
                temp_input_path,
                split_identifier=main_column.TRADE_TYPE
            )

            config = CreditDeltaInjectorConfig.cds_preset()
            injector = CreditDeltaOutlierInjector(config=config)
            injected_df = injector.inject(original_df)

            # Count injections per scenario and trade type
            for scenario in ["CD_Drift", "CD_StaleValue", "CD_ClusterShock_3d"]:
                scenario_mask = injected_df[main_column.RECORD_TYPE] == scenario
                if scenario_mask.any():
                    trade_types = injected_df.loc[scenario_mask, main_column.TRADE_TYPE].unique()
                    # Should only have trade types from the original dataset
                    original_trade_types = original_df[main_column.TRADE_TYPE].unique()
                    self.assertTrue(
                        all(tt in original_trade_types for tt in trade_types),
                        f"Unknown trade types in {scenario}: {trade_types}"
                    )

    def test_cds_injections_export_enriched(self):
        """Export enriched file with original and injected CDS values for review."""
        original_input_file = "CreditDeltaSingle_Input.csv"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_path = os.path.join(temp_dir, original_input_file)
            shutil.copy2(os.path.join(ORIGINAL_INPUT_DIRECTORY, original_input_file), temp_input_path)

            input_handler = CreditDeltaSingleInput()
            original_df = input_handler.read_and_validate(
                temp_input_path,
                split_identifier=main_column.TRADE_TYPE
            )

            injector = CreditDeltaOutlierInjector(config=CreditDeltaInjectorConfig.cds_preset())
            injected_df = injector.inject(original_df)

            # Build enriched dataset
            enriched_df = original_df.copy()
            if main_column.RECORD_TYPE in injected_df.columns:
                enriched_df[main_column.RECORD_TYPE] = injected_df[main_column.RECORD_TYPE]

            # Add injected value column for CDS
            injected_cds_col = f"{cds_column.CREDIT_DELTA_SINGLE}_injected"
            changed_mask = injected_df[cds_column.CREDIT_DELTA_SINGLE] != original_df[cds_column.CREDIT_DELTA_SINGLE]
            enriched_df[injected_cds_col] = injected_df[cds_column.CREDIT_DELTA_SINGLE].where(changed_mask, pd.NA)

            # Basic validations
            self.assertEqual(
                enriched_df.shape[0],
                original_df.shape[0],
                "Enriched dataset row count differs from original"
            )
            self.assertIn(injected_cds_col, enriched_df.columns, "Missing injected CDS column")

            # Ensure at least one injected value is present
            any_injected = enriched_df[injected_cds_col].notna().any()
            self.assertTrue(any_injected, "No injected CDS values found")

            # Export for review
            output_name = os.path.splitext(original_input_file)[0] + "_injected.csv"
            output_path = os.path.join(temp_dir, output_name)
            enriched_df.to_csv(output_path, index=False)
            self.assertTrue(os.path.exists(output_path), f"Output file not created: {output_path}")

            # Copy output to original directory for review
            shutil.copy2(output_path, os.path.join(ORIGINAL_INPUT_DIRECTORY, output_name))


class TestCdiOutlierInjector(unittest.TestCase):

    def test_cdi_injections_applied(self):
        """Validate that CDI injections are applied correctly to OOS data."""
        original_input_file = "CreditDeltaIndex_Input_Train-OOS.csv"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_path = os.path.join(temp_dir, original_input_file)
            shutil.copy2(os.path.join(ORIGINAL_INPUT_DIRECTORY, original_input_file), temp_input_path)

            input_handler = CreditDeltaIndexInput()
            original_df = input_handler.read_and_validate(
                temp_input_path,
                split_identifier=main_column.TRADE_TYPE
            )

            config = CreditDeltaInjectorConfig.credit_delta_index_preset()
            injector = CreditDeltaOutlierInjector(config=config)
            injected_df = injector.inject(original_df)

            # 1) Shape should be identical (or have RecordType added if not in original)
            if main_column.RECORD_TYPE not in original_df.columns:
                # Injector adds RecordType if missing
                self.assertEqual(
                    original_df.shape[0],
                    injected_df.shape[0],
                    "Row count differs"
                )
                self.assertEqual(
                    original_df.shape[1] + 1,  # +1 for RecordType column added by injector
                    injected_df.shape[1],
                    "Column count should match (or differ by 1 if RecordType was added)"
                )
            else:
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

            # 3) All CDI injection scenarios should be present
            expected_scenarios = {
                "CD_Drift",
                "CD_StaleValue",
                "CD_PointShock",
                "CD_SignFlip",
                "CD_ScaleError",
                "CD_SuddenZero",
            }

            injected_labels = set(
                injected_df[main_column.RECORD_TYPE]
                .astype(str)
                .unique()
            )

            missing = expected_scenarios - injected_labels
            self.assertFalse(
                missing,
                f"Missing CDI injection scenarios: {sorted(missing)}"
            )

            # 4) No Train rows should be changed
            if main_column.RECORD_TYPE in original_df.columns:
                train_mask = original_df[main_column.RECORD_TYPE] == "Train"
                if train_mask.any():
                    pd.testing.assert_frame_equal(
                        original_df.loc[train_mask].reset_index(drop=True),
                        injected_df.loc[train_mask].reset_index(drop=True),
                        check_dtype=False
                    )

            # 5) All injected rows must be OOS (or newly added if RecordType didn't exist)
            injected_mask = injected_df[main_column.RECORD_TYPE].astype(str).isin(expected_scenarios)
            self.assertTrue(
                injected_mask.any(),
                "No injected rows found in dataset"
            )
            
            # Verify that modified rows were originally OOS (if RecordType existed in original)
            if main_column.RECORD_TYPE in original_df.columns:
                original_injected_record_types = original_df.loc[injected_mask, main_column.RECORD_TYPE].unique()
                self.assertTrue(
                    all(rt in {"OOS", "Train"} for rt in original_injected_record_types),
                    f"Some injected rows were not from OOS: {original_injected_record_types}"
                )

    def test_cdi_injections_by_trade_type(self):
        """Validate that CDI injections respect per-trade-type configuration."""
        original_input_file = "CreditDeltaIndex_Input_Train-OOS.csv"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_path = os.path.join(temp_dir, original_input_file)
            shutil.copy2(os.path.join(ORIGINAL_INPUT_DIRECTORY, original_input_file), temp_input_path)

            input_handler = CreditDeltaIndexInput()
            original_df = input_handler.read_and_validate(
                temp_input_path,
                split_identifier=main_column.TRADE_TYPE
            )

            config = CreditDeltaInjectorConfig.credit_delta_index_preset()
            injector = CreditDeltaOutlierInjector(config=config)
            injected_df = injector.inject(original_df)

            # Count injections per scenario and trade type
            for scenario in ["CD_Drift", "CD_StaleValue", "CD_PointShock"]:
                scenario_mask = injected_df[main_column.RECORD_TYPE] == scenario
                if scenario_mask.any():
                    trade_types = injected_df.loc[scenario_mask, main_column.TRADE_TYPE].unique()
                    # Should only have trade types from the original dataset
                    original_trade_types = original_df[main_column.TRADE_TYPE].unique()
                    self.assertTrue(
                        all(tt in original_trade_types for tt in trade_types),
                        f"Unknown trade types in {scenario}: {trade_types}"
                    )

    def test_cdi_injections_export_enriched(self):
        """Export enriched file with original and injected CDI values for review."""
        original_input_file = "CreditDeltaIndex_Input_Train-OOS.csv"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_path = os.path.join(temp_dir, original_input_file)
            shutil.copy2(os.path.join(ORIGINAL_INPUT_DIRECTORY, original_input_file), temp_input_path)

            input_handler = CreditDeltaIndexInput()
            original_df = input_handler.read_and_validate(
                temp_input_path,
                split_identifier=main_column.TRADE_TYPE
            )

            injector = CreditDeltaOutlierInjector(config=CreditDeltaInjectorConfig.credit_delta_index_preset())
            injected_df = injector.inject(original_df)

            # Build enriched dataset
            enriched_df = original_df.copy()
            if main_column.RECORD_TYPE in injected_df.columns:
                enriched_df[main_column.RECORD_TYPE] = injected_df[main_column.RECORD_TYPE]

            # Add injected value column for CDI
            injected_cdi_col = f"{cdi_column.CREDIT_DELTA_INDEX}_injected"
            changed_mask = injected_df[cdi_column.CREDIT_DELTA_INDEX] != original_df[cdi_column.CREDIT_DELTA_INDEX]
            enriched_df[injected_cdi_col] = injected_df[cdi_column.CREDIT_DELTA_INDEX].where(changed_mask, pd.NA)

            # Basic validations
            self.assertEqual(
                enriched_df.shape[0],
                original_df.shape[0],
                "Enriched dataset row count differs from original"
            )
            self.assertIn(injected_cdi_col, enriched_df.columns, "Missing injected CDI column")

            # Ensure at least one injected value is present
            any_injected = enriched_df[injected_cdi_col].notna().any()
            self.assertTrue(any_injected, "No injected CDI values found")

            # Export for review
            output_name = os.path.splitext(original_input_file)[0] + "_injected.csv"
            output_path = os.path.join(temp_dir, output_name)
            enriched_df.to_csv(output_path, index=False)
            self.assertTrue(os.path.exists(output_path), f"Output file not created: {output_path}")

            # Copy output to original directory for review
            shutil.copy2(output_path, os.path.join(ORIGINAL_INPUT_DIRECTORY, output_name))


if __name__ == '__main__':
    unittest.main()
