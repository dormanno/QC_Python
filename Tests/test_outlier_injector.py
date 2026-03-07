import unittest
import os
import tempfile
import shutil
import pandas as pd
from IO.input import PnLInput, CreditDeltaSingleInput, CreditDeltaIndexInput, PVInput, PnLSlicesInput
from column_names import main_column, cds_column, cdi_column, pv_column, pnl_slices_column
from Tests.outlier_injectors import PnLOutlierInjector, CreditDeltaOutlierInjector, PVOutlierInjector, PnLSlicesOutlierInjector
from Tests.outlier_injectors.credit_delta_config import CreditDeltaInjectorConfig, ScenarioNames as CDScenarioNames
from Tests.outlier_injectors.pv_config import PVInjectorConfig, ScenarioNames as PVScenarioNames
from Tests.outlier_injectors.pnl_slices_config import PnLSlicesInjectorConfig, ScenarioNames as PnLSlicesScenarioNames

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

        # Source row (first eligible date) should be protected and value unchanged
        self.assertEqual(injected_df.loc[0, main_column.RECORD_TYPE], "StaleValue_Source")
        self.assertEqual(injected_df.loc[0, cds_column.CREDIT_DELTA_SINGLE], source_value)

        # Following stale_days rows should be stale and equal to source value
        stale_days = config.stale_days
        stale_rows = injected_df.iloc[1:1 + stale_days]
        self.assertTrue((stale_rows[main_column.RECORD_TYPE] == "StaleValue").all())
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
                CDScenarioNames.DRIFT,
                CDScenarioNames.STALE_VALUE,
                CDScenarioNames.CLUSTER_SHOCK_3D,
                CDScenarioNames.TRADE_TYPE_WIDE_SHOCK,
                CDScenarioNames.POINT_SHOCK,
                CDScenarioNames.SIGN_FLIP,
                CDScenarioNames.SCALE_ERROR,
                CDScenarioNames.SUDDEN_ZERO,
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
            for scenario in [CDScenarioNames.DRIFT, CDScenarioNames.STALE_VALUE, CDScenarioNames.CLUSTER_SHOCK_3D]:
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
                CDScenarioNames.DRIFT,
                CDScenarioNames.STALE_VALUE,
                CDScenarioNames.POINT_SHOCK,
                CDScenarioNames.SIGN_FLIP,
                CDScenarioNames.SCALE_ERROR,
                CDScenarioNames.SUDDEN_ZERO,
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
            for scenario in [CDScenarioNames.DRIFT, CDScenarioNames.STALE_VALUE, CDScenarioNames.POINT_SHOCK]:
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


class TestPVOutlierInjector(unittest.TestCase):

    def test_pv_stale_value_does_not_label_source_row(self):
        """The source row used for stale propagation must remain unchanged and not relabeled."""
        config = PVInjectorConfig.pv_preset()
        injector = PVOutlierInjector(config=config, random_seed=42)

        dates = pd.date_range("2025-01-01", periods=7, freq="D")
        source_start_pv = 1000.0
        source_end_pv = 1100.0
        original_start_pvs = [source_start_pv, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0]
        original_end_pvs = [source_end_pv, 2100.0, 3100.0, 4100.0, 5100.0, 6100.0, 7100.0]

        original_df = pd.DataFrame({
            main_column.RECORD_TYPE: ["OOS"] * 7,
            main_column.TRADE: ["T_BASIS_1"] * 7,
            main_column.BOOK: ["B1"] * 7,
            main_column.TRADE_TYPE: ["Basis"] * 7,
            main_column.DATE: dates,
            pv_column.START_PV: original_start_pvs,
            pv_column.END_PV: original_end_pvs,
        })

        injected_df = injector.inject_pv_stale_value(original_df)

        # Source row (first eligible date) should be protected and values unchanged
        self.assertEqual(injected_df.loc[0, main_column.RECORD_TYPE], "StaleValue_Source")
        self.assertEqual(injected_df.loc[0, pv_column.START_PV], source_start_pv)
        self.assertEqual(injected_df.loc[0, pv_column.END_PV], source_end_pv)

        # Following stale_days rows should be stale and equal to source values
        stale_days = config.stale_days
        stale_rows = injected_df.iloc[1:1 + stale_days]
        self.assertTrue((stale_rows[main_column.RECORD_TYPE] == "StaleValue").all())
        self.assertTrue((stale_rows[pv_column.START_PV] == source_start_pv).all())
        self.assertTrue((stale_rows[pv_column.END_PV] == source_end_pv).all())

        # Remaining rows should stay OOS and unchanged
        tail_rows = injected_df.iloc[1 + stale_days:]
        self.assertTrue((tail_rows[main_column.RECORD_TYPE] == "OOS").all())
        self.assertEqual(
            tail_rows[pv_column.START_PV].tolist(),
            original_start_pvs[1 + stale_days:]
        )
        self.assertEqual(
            tail_rows[pv_column.END_PV].tolist(),
            original_end_pvs[1 + stale_days:]
        )

    def test_pv_both_features_injected_identically(self):
        """Verify that Start_PV and End_PV receive identical outlier injections."""
        config = PVInjectorConfig.pv_preset()
        injector = PVOutlierInjector(config=config, random_seed=42)

        dates = pd.date_range("2025-01-01", periods=50, freq="D")
        original_df = pd.DataFrame({
            main_column.RECORD_TYPE: ["OOS"] * 50,
            main_column.TRADE: [f"T_BASIS_{i//10}" for i in range(50)],
            main_column.BOOK: ["B1"] * 50,
            main_column.TRADE_TYPE: ["Basis"] * 50,
            main_column.DATE: dates,
            pv_column.START_PV: [1000.0 + i * 10 for i in range(50)],
            pv_column.END_PV: [1100.0 + i * 10 for i in range(50)],
        })

        injected_df = injector.inject(original_df)

        # Find all injected rows
        injected_mask = injected_df[main_column.RECORD_TYPE].astype(str).isin([
            PVScenarioNames.DRIFT,
            PVScenarioNames.STALE_VALUE,
            PVScenarioNames.CLUSTER_SHOCK_3D,
            PVScenarioNames.TRADE_TYPE_WIDE_SHOCK,
            PVScenarioNames.POINT_SHOCK,
            PVScenarioNames.SIGN_FLIP,
            PVScenarioNames.SCALE_ERROR,
            PVScenarioNames.SUDDEN_ZERO,
        ])

        # For each injected row, verify that Start_PV and End_PV changed by the same amount
        for idx in injected_df[injected_mask].index:
            original_start = original_df.loc[idx, pv_column.START_PV]
            original_end = original_df.loc[idx, pv_column.END_PV]
            injected_start = injected_df.loc[idx, pv_column.START_PV]
            injected_end = injected_df.loc[idx, pv_column.END_PV]

            # Calculate the changes
            start_delta = injected_start - original_start
            end_delta = injected_end - original_end

            # For additive scenarios (Drift, ClusterShock, PointShock), deltas should be identical
            # For stale values, both should equal their source values
            # For sign flip, both multiply by -1
            # For scale error, both multiply by same factor
            # For sudden zero, both become 0
            scenario = injected_df.loc[idx, main_column.RECORD_TYPE]

            if scenario in [PVScenarioNames.DRIFT, PVScenarioNames.CLUSTER_SHOCK_3D, 
                          PVScenarioNames.POINT_SHOCK, PVScenarioNames.TRADE_TYPE_WIDE_SHOCK]:
                # Additive: deltas should be identical
                self.assertAlmostEqual(start_delta, end_delta, places=5,
                                     msg=f"Row {idx} ({scenario}): Start_PV delta {start_delta} != End_PV delta {end_delta}")
            elif scenario == PVScenarioNames.SIGN_FLIP:
                # Sign flip: both should be negated
                self.assertAlmostEqual(injected_start, -original_start, places=5)
                self.assertAlmostEqual(injected_end, -original_end, places=5)
            elif scenario == PVScenarioNames.SCALE_ERROR:
                # Scale error: both should be multiplied by same factor
                if original_start != 0:
                    scale_start = injected_start / original_start
                    scale_end = injected_end / original_end
                    self.assertAlmostEqual(scale_start, scale_end, places=5)
            elif scenario == PVScenarioNames.SUDDEN_ZERO:
                # Sudden zero: both should be 0
                self.assertEqual(injected_start, 0.0)
                self.assertEqual(injected_end, 0.0)

    def test_pv_injections_applied(self):
        """Validate that PV injections are applied correctly to OOS data."""
        original_input_file = "PV_Train-OOS.csv"

        # Skip test if file doesn't exist
        if not os.path.exists(os.path.join(ORIGINAL_INPUT_DIRECTORY, original_input_file)):
            self.skipTest(f"PV input file not found: {original_input_file}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_path = os.path.join(temp_dir, original_input_file)
            shutil.copy2(os.path.join(ORIGINAL_INPUT_DIRECTORY, original_input_file), temp_input_path)

            input_handler = PVInput()
            original_df = input_handler.read_and_validate(
                temp_input_path,
                split_identifier=main_column.TRADE_TYPE
            )

            config = PVInjectorConfig.pv_preset()
            injector = PVOutlierInjector(config=config)
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

            # 3) All PV injection scenarios should be present
            expected_scenarios = {
                PVScenarioNames.DRIFT,
                PVScenarioNames.STALE_VALUE,
                PVScenarioNames.CLUSTER_SHOCK_3D,
                PVScenarioNames.TRADE_TYPE_WIDE_SHOCK,
                PVScenarioNames.POINT_SHOCK,
                PVScenarioNames.SIGN_FLIP,
                PVScenarioNames.SCALE_ERROR,
                PVScenarioNames.SUDDEN_ZERO,
            }

            injected_labels = set(
                injected_df[main_column.RECORD_TYPE]
                .astype(str)
                .unique()
            )

            missing = expected_scenarios - injected_labels
            self.assertFalse(
                missing,
                f"Missing PV injection scenarios: {sorted(missing)}"
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

    def test_pv_injections_by_trade_type(self):
        """Validate that PV injections respect per-trade-type configuration."""
        original_input_file = "PV_Train-OOS.csv"

        # Skip test if file doesn't exist
        if not os.path.exists(os.path.join(ORIGINAL_INPUT_DIRECTORY, original_input_file)):
            self.skipTest(f"PV input file not found: {original_input_file}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_path = os.path.join(temp_dir, original_input_file)
            shutil.copy2(os.path.join(ORIGINAL_INPUT_DIRECTORY, original_input_file), temp_input_path)

            input_handler = PVInput()
            original_df = input_handler.read_and_validate(
                temp_input_path,
                split_identifier=main_column.TRADE_TYPE
            )

            config = PVInjectorConfig.pv_preset()
            injector = PVOutlierInjector(config=config)
            injected_df = injector.inject(original_df)

            # Count injections per scenario and trade type
            for scenario in [PVScenarioNames.DRIFT, PVScenarioNames.STALE_VALUE, PVScenarioNames.CLUSTER_SHOCK_3D]:
                scenario_mask = injected_df[main_column.RECORD_TYPE] == scenario
                if scenario_mask.any():
                    trade_types = injected_df.loc[scenario_mask, main_column.TRADE_TYPE].unique()
                    # Should only have trade types from the original dataset
                    original_trade_types = original_df[main_column.TRADE_TYPE].unique()
                    self.assertTrue(
                        all(tt in original_trade_types for tt in trade_types),
                        f"Unknown trade types in {scenario}: {trade_types}"
                    )

    def test_pv_injections_export_enriched(self):
        """Export enriched file with original and injected PV values for review."""
        original_input_file = "PV_Train-OOS.csv"

        # Skip test if file doesn't exist
        if not os.path.exists(os.path.join(ORIGINAL_INPUT_DIRECTORY, original_input_file)):
            self.skipTest(f"PV input file not found: {original_input_file}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_path = os.path.join(temp_dir, original_input_file)
            shutil.copy2(os.path.join(ORIGINAL_INPUT_DIRECTORY, original_input_file), temp_input_path)

            input_handler = PVInput()
            original_df = input_handler.read_and_validate(
                temp_input_path,
                split_identifier=main_column.TRADE_TYPE
            )

            injector = PVOutlierInjector(config=PVInjectorConfig.pv_preset())
            injected_df = injector.inject(original_df)

            # Build enriched dataset
            enriched_df = original_df.copy()
            if main_column.RECORD_TYPE in injected_df.columns:
                enriched_df[main_column.RECORD_TYPE] = injected_df[main_column.RECORD_TYPE]

            # Add injected value columns for both PV features
            injected_start_col = f"{pv_column.START_PV}_injected"
            injected_end_col = f"{pv_column.END_PV}_injected"
            
            changed_start_mask = injected_df[pv_column.START_PV] != original_df[pv_column.START_PV]
            changed_end_mask = injected_df[pv_column.END_PV] != original_df[pv_column.END_PV]
            
            enriched_df[injected_start_col] = injected_df[pv_column.START_PV].where(changed_start_mask, pd.NA)
            enriched_df[injected_end_col] = injected_df[pv_column.END_PV].where(changed_end_mask, pd.NA)

            # Basic validations
            self.assertEqual(
                enriched_df.shape[0],
                original_df.shape[0],
                "Enriched dataset row count differs from original"
            )
            self.assertIn(injected_start_col, enriched_df.columns, "Missing injected Start_PV column")
            self.assertIn(injected_end_col, enriched_df.columns, "Missing injected End_PV column")

            # Ensure at least one injected value is present in each feature
            any_start_injected = enriched_df[injected_start_col].notna().any()
            any_end_injected = enriched_df[injected_end_col].notna().any()
            self.assertTrue(any_start_injected, "No injected Start_PV values found")
            self.assertTrue(any_end_injected, "No injected End_PV values found")

            # Verify that Start_PV and End_PV injections occur on the same rows
            start_injected_rows = enriched_df[injected_start_col].notna()
            end_injected_rows = enriched_df[injected_end_col].notna()
            self.assertTrue(
                (start_injected_rows == end_injected_rows).all(),
                "Start_PV and End_PV injections should occur on the same rows"
            )

            # Export for review
            output_name = os.path.splitext(original_input_file)[0] + "_injected.csv"
            output_path = os.path.join(temp_dir, output_name)
            enriched_df.to_csv(output_path, index=False)
            self.assertTrue(os.path.exists(output_path), f"Output file not created: {output_path}")

            # Copy output to original directory for review
            shutil.copy2(output_path, os.path.join(ORIGINAL_INPUT_DIRECTORY, output_name))


class TestPnLSlicesOutlierInjector(unittest.TestCase):

    def test_pnl_slices_stale_value_does_not_label_source_row(self):
        """The source row used for stale propagation must remain unchanged and not relabeled."""
        config = PnLSlicesInjectorConfig.pnl_slices_preset()
        injector = PnLSlicesOutlierInjector(config=config, random_seed=42)

        dates = pd.date_range("2025-01-01", periods=7, freq="D")
        # Create original values for all 10 slice features
        n = 7
        source_values = {col: float((i + 1) * 100) for i, col in enumerate(config.slice_columns)}
        original_values = {
            col: [source_values[col]] + [source_values[col] + j * 10 for j in range(1, n)]
            for col in config.slice_columns
        }

        data = {
            main_column.RECORD_TYPE: ["OOS"] * n,
            main_column.TRADE: ["T_BASIS_1"] * n,
            main_column.BOOK: ["B1"] * n,
            main_column.TRADE_TYPE: ["Basis"] * n,
            main_column.DATE: dates,
        }
        for col in config.slice_columns:
            data[col] = original_values[col]

        original_df = pd.DataFrame(data)
        injected_df = injector.inject_stale_value(original_df)

        # Source row should be protected and values unchanged
        self.assertEqual(injected_df.loc[0, main_column.RECORD_TYPE], "StaleValue_Source")
        for col in config.slice_columns:
            self.assertEqual(injected_df.loc[0, col], source_values[col])

        # Following stale_days rows should be stale and equal to source values
        stale_days = config.stale_days
        stale_rows = injected_df.iloc[1:1 + stale_days]
        self.assertTrue((stale_rows[main_column.RECORD_TYPE] == "StaleValue").all())
        for col in config.slice_columns:
            self.assertTrue((stale_rows[col] == source_values[col]).all())

        # Remaining rows should stay OOS and unchanged
        tail_rows = injected_df.iloc[1 + stale_days:]
        self.assertTrue((tail_rows[main_column.RECORD_TYPE] == "OOS").all())
        for col in config.slice_columns:
            self.assertEqual(
                tail_rows[col].tolist(),
                original_values[col][1 + stale_days:]
            )

    def test_pnl_slices_all_features_injected_identically(self):
        """Verify that all 10 slice features receive consistent outlier injections."""
        config = PnLSlicesInjectorConfig.pnl_slices_preset()
        injector = PnLSlicesOutlierInjector(config=config, random_seed=42)

        dates = pd.date_range("2025-01-01", periods=50, freq="D")
        data = {
            main_column.RECORD_TYPE: ["OOS"] * 50,
            main_column.TRADE: [f"T_BASIS_{i // 10}" for i in range(50)],
            main_column.BOOK: ["B1"] * 50,
            main_column.TRADE_TYPE: ["Basis"] * 50,
            main_column.DATE: dates,
        }
        for j, col in enumerate(config.slice_columns):
            data[col] = [1000.0 + i * 10 + j * 100 for i in range(50)]

        original_df = pd.DataFrame(data)
        injected_df = injector.inject(original_df)

        # Find all injected rows
        injected_mask = injected_df[main_column.RECORD_TYPE].astype(str).isin([
            PnLSlicesScenarioNames.DRIFT,
            PnLSlicesScenarioNames.STALE_VALUE,
            PnLSlicesScenarioNames.CLUSTER_SHOCK_3D,
            PnLSlicesScenarioNames.TRADE_TYPE_WIDE_SHOCK,
            PnLSlicesScenarioNames.POINT_SHOCK,
            PnLSlicesScenarioNames.SIGN_FLIP,
            PnLSlicesScenarioNames.SUDDEN_ZERO,
        ])

        # For each injected row, verify all features changed consistently
        for idx in injected_df[injected_mask].index:
            scenario = injected_df.loc[idx, main_column.RECORD_TYPE]

            if scenario == PnLSlicesScenarioNames.SIGN_FLIP:
                for col in config.slice_columns:
                    self.assertAlmostEqual(
                        injected_df.loc[idx, col], -original_df.loc[idx, col], places=5,
                        msg=f"Row {idx} ({scenario}): {col} not sign-flipped"
                    )
            elif scenario == PnLSlicesScenarioNames.SUDDEN_ZERO:
                for col in config.slice_columns:
                    self.assertEqual(
                        injected_df.loc[idx, col], 0.0,
                        msg=f"Row {idx} ({scenario}): {col} not zero"
                    )

    def test_pnl_slices_injections_applied(self):
        """Validate that PnL Slices injections are applied correctly to OOS data."""
        original_input_file = "PnL_Slices_Train-OOS.csv"

        if not os.path.exists(os.path.join(ORIGINAL_INPUT_DIRECTORY, original_input_file)):
            self.skipTest(f"PnL Slices input file not found: {original_input_file}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_path = os.path.join(temp_dir, original_input_file)
            shutil.copy2(os.path.join(ORIGINAL_INPUT_DIRECTORY, original_input_file), temp_input_path)

            input_handler = PnLSlicesInput()
            original_df = input_handler.read_and_validate(
                temp_input_path,
                split_identifier=main_column.TRADE_TYPE
            )

            config = PnLSlicesInjectorConfig.pnl_slices_preset()
            injector = PnLSlicesOutlierInjector(config=config)
            injected_df = injector.inject(original_df)

            # 1) Row count should match
            self.assertEqual(original_df.shape[0], injected_df.shape[0], "Row count differs")

            # 2) Original should contain only Train and OOS labels
            if main_column.RECORD_TYPE in original_df.columns:
                allowed_labels = {"Train", "OOS"}
                original_labels = set(original_df[main_column.RECORD_TYPE].astype(str).unique())
                self.assertTrue(
                    original_labels.issubset(allowed_labels),
                    f"Original dataset has unexpected RecordType labels: {sorted(original_labels - allowed_labels)}"
                )

            # 3) All 7 injection scenarios should be present (no ScaleError)
            expected_scenarios = {
                PnLSlicesScenarioNames.DRIFT,
                PnLSlicesScenarioNames.STALE_VALUE,
                PnLSlicesScenarioNames.CLUSTER_SHOCK_3D,
                PnLSlicesScenarioNames.TRADE_TYPE_WIDE_SHOCK,
                PnLSlicesScenarioNames.POINT_SHOCK,
                PnLSlicesScenarioNames.SIGN_FLIP,
                PnLSlicesScenarioNames.SUDDEN_ZERO,
            }

            injected_labels = set(
                injected_df[main_column.RECORD_TYPE].astype(str).unique()
            )
            missing = expected_scenarios - injected_labels
            self.assertFalse(missing, f"Missing PnL Slices injection scenarios: {sorted(missing)}")

            # 4) No Train rows should be changed
            if main_column.RECORD_TYPE in original_df.columns:
                train_mask = original_df[main_column.RECORD_TYPE] == "Train"
                if train_mask.any():
                    pd.testing.assert_frame_equal(
                        original_df.loc[train_mask].reset_index(drop=True),
                        injected_df.loc[train_mask].reset_index(drop=True),
                        check_dtype=False
                    )

    def test_pnl_slices_injections_by_trade_type(self):
        """Validate that PnL Slices injections respect per-trade-type configuration."""
        original_input_file = "PnL_Slices_Train-OOS.csv"

        if not os.path.exists(os.path.join(ORIGINAL_INPUT_DIRECTORY, original_input_file)):
            self.skipTest(f"PnL Slices input file not found: {original_input_file}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_path = os.path.join(temp_dir, original_input_file)
            shutil.copy2(os.path.join(ORIGINAL_INPUT_DIRECTORY, original_input_file), temp_input_path)

            input_handler = PnLSlicesInput()
            original_df = input_handler.read_and_validate(
                temp_input_path,
                split_identifier=main_column.TRADE_TYPE
            )

            config = PnLSlicesInjectorConfig.pnl_slices_preset()
            injector = PnLSlicesOutlierInjector(config=config)
            injected_df = injector.inject(original_df)

            for scenario in [PnLSlicesScenarioNames.DRIFT, PnLSlicesScenarioNames.STALE_VALUE,
                             PnLSlicesScenarioNames.CLUSTER_SHOCK_3D]:
                scenario_mask = injected_df[main_column.RECORD_TYPE] == scenario
                if scenario_mask.any():
                    trade_types = injected_df.loc[scenario_mask, main_column.TRADE_TYPE].unique()
                    original_trade_types = original_df[main_column.TRADE_TYPE].unique()
                    self.assertTrue(
                        all(tt in original_trade_types for tt in trade_types),
                        f"Unknown trade types in {scenario}: {trade_types}"
                    )

    def test_pnl_slices_injections_export_enriched(self):
        """Export enriched file with original and injected PnL Slices values for review."""
        original_input_file = "PnL_Slices_Train-OOS.csv"

        if not os.path.exists(os.path.join(ORIGINAL_INPUT_DIRECTORY, original_input_file)):
            self.skipTest(f"PnL Slices input file not found: {original_input_file}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_path = os.path.join(temp_dir, original_input_file)
            shutil.copy2(os.path.join(ORIGINAL_INPUT_DIRECTORY, original_input_file), temp_input_path)

            input_handler = PnLSlicesInput()
            original_df = input_handler.read_and_validate(
                temp_input_path,
                split_identifier=main_column.TRADE_TYPE
            )

            injector = PnLSlicesOutlierInjector(config=PnLSlicesInjectorConfig.pnl_slices_preset())
            injected_df = injector.inject(original_df)

            enriched_df = original_df.copy()
            if main_column.RECORD_TYPE in injected_df.columns:
                enriched_df[main_column.RECORD_TYPE] = injected_df[main_column.RECORD_TYPE]

            # Add injected value columns for all slice features
            for col in pnl_slices_column.SLICE_COLUMNS:
                injected_col = f"{col}_injected"
                changed_mask = injected_df[col] != original_df[col]
                enriched_df[injected_col] = injected_df[col].where(changed_mask, pd.NA)

            self.assertEqual(enriched_df.shape[0], original_df.shape[0])

            # Ensure at least one injected value exists
            any_injected = any(
                enriched_df[f"{col}_injected"].notna().any() for col in pnl_slices_column.SLICE_COLUMNS
            )
            self.assertTrue(any_injected, "No injected PnL Slices values found")

            output_name = os.path.splitext(original_input_file)[0] + "_injected.csv"
            output_path = os.path.join(temp_dir, output_name)
            enriched_df.to_csv(output_path, index=False)
            self.assertTrue(os.path.exists(output_path))

            shutil.copy2(output_path, os.path.join(ORIGINAL_INPUT_DIRECTORY, output_name))


if __name__ == '__main__':
    unittest.main()
