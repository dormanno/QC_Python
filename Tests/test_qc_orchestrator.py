import unittest
import os
import pandas as pd
import tempfile
import shutil
from Engine import qc_engine_presets
from Tests.outlier_injectors.credit_delta import CreditDeltaOutlierInjector
from Tests.outlier_injectors.credit_delta_config import CreditDeltaInjectorConfig
from qc_orchestrator import QCOrchestrator
from Engine.feature_normalizer import FeatureNormalizer
from IO.input import PnLInput, CreditDeltaSingleInput, CreditDeltaIndexInput
from IO.output import Output
from column_names import pnl_column, cds_column, cdi_column, main_column
from QC_methods.qc_method_definitions import QCMethodDefinitions
from Tests.outlier_injectors import PnLOutlierInjector
from Tests.outlier_injectors.pnl_config import PnLInjectorConfig

ORIGINAL_INPUT_DIRECTORY = r"C:\Users\dorma\Documents\UEK_Backup\Test"

# Define method configuration (QCMethod -> weight)
METHODS_CONFIG = {
    QCMethodDefinitions.ISOLATION_FOREST: 0.2,
    QCMethodDefinitions.ROBUST_Z: 0.1,
    QCMethodDefinitions.ROLLING: 0.1,
    QCMethodDefinitions.IQR: 0.1,
    QCMethodDefinitions.LOF: 0.2,
    QCMethodDefinitions.ECDF: 0.2,
    QCMethodDefinitions.HAMPEL: 0.1
}


ROLL_WINDOW = 20

# Severity levels for outlier injection (multipliers for MAD/IQR scale)
SEVERITY_SMALL = 3
SEVERITY_MEDIUM = 6
SEVERITY_HIGH = 12
SEVERITY_EXTREME = 24

# Severity used in injection tests
INJECTION_SEVERITY = SEVERITY_MEDIUM

class TestQCOrchestrator(unittest.TestCase):
    
    def _run_qc_test(self, original_input_file, input_handler, columnSet, engine_preset, injector=None, inject=False):
        """Helper method to run QC test with specified column configuration and input file.
        
        Args:            
            original_input_file: Input file name
            input_handler: Input handler instance
            columnSet: Column set instance
            engine_preset: QCEnginePreset instance
            injector: Optional OutlierInjector subclass instance to manipulate data before QC
        """          
        
        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Temporary directory created at: {temp_dir}")
            # Copy the input file to temp directory
            temp_input_path = os.path.join(temp_dir, original_input_file)
            shutil.copy2(os.path.join(ORIGINAL_INPUT_DIRECTORY, original_input_file), temp_input_path)

            # Read raw input to get baseline (before engineering)
            raw_input_df = pd.read_csv(temp_input_path)
            input_rows = len(raw_input_df)
            input_cols = len(raw_input_df.columns)

            normalizer = FeatureNormalizer(features=columnSet.QC_FEATURES)

            # Run the orchestrator
            try:
                # Read and validate data
                full_data_set = input_handler.read_and_validate(temp_input_path, split_identifier=main_column.TRADE_TYPE)
                # Inject outliers if requested
                if inject:
                    if injector is None:
                        raise ValueError("inject=True but no injector provided")
                    full_data_set = injector.inject(full_data_set)
                
                
                orchestrator = QCOrchestrator(
                    normalizer=normalizer,
                    engine_preset=engine_preset,
                    split_identifier=main_column.TRADE_TYPE
                )
                oos_scores = orchestrator.run(full_data_set)
                
                # Export results to file
                output_handler = Output()
                score_columns = engine_preset.get_score_columns()
                output_path = output_handler.export_full_dataset(
                    full_data_set=full_data_set,
                    oos_scores=oos_scores,
                    input_path=temp_input_path,
                    score_cols=score_columns,
                    suffix="_with_scores"
                )
                # 1. Run succeeded (no exception)
                self.assertIsInstance(output_path, str)
            except Exception as e:
                self.fail(f"QCOrchestrator.run() raised an exception: {e}")

            # 2. Output file got created
            self.assertTrue(os.path.exists(output_path), f"Output file does not exist: {output_path}")

            # 3. Output file is not empty
            output_df = pd.read_csv(output_path)
            self.assertGreater(len(output_df), 0, "Output file is empty")

            # 4. Output file has same number of rows as input
            self.assertEqual(len(output_df), input_rows, f"Output rows {len(output_df)} != input rows {input_rows}")

            # 5. Output file should have original columns plus engineered and QC score columns
            # Get actual score columns from the preset
            actual_score_cols_count = len(engine_preset.get_score_columns())
            expected_cols = input_cols + columnSet.ENGINEERED_FEATURES.__len__() + actual_score_cols_count
            self.assertEqual(len(output_df.columns), expected_cols, f"Output columns {len(output_df.columns)} != expected {expected_cols}")

            # Copy output back to original directory for review
            try:
                shutil.copy2(output_path, os.path.join(ORIGINAL_INPUT_DIRECTORY, os.path.basename(output_path)))
            except PermissionError:
                print(f"Warning: Could not copy output to {ORIGINAL_INPUT_DIRECTORY}. Permission denied.")

    def test_QC_PnL(self):
        """Test QC for PnL data."""
        self._run_qc_test(
            "PnL_Input_Train-OOS.csv", 
            #"PnL_Input_Injected.csv",
            PnLInput(), 
            pnl_column, 
            qc_engine_presets.preset_temporal_multivariate_pnl)

    def test_QC_PnL_with_injections(self):
        """Test QC for PnL data with outlier injections."""
        config = PnLInjectorConfig.default_preset()
        self._run_qc_test(
            "PnL_Input_Train-OOS.csv",
            PnLInput(),
            pnl_column,
            qc_engine_presets.preset_temporal_multivariate_pnl,
            injector=PnLOutlierInjector(config=config, severity=INJECTION_SEVERITY),
            inject=True)
    
    def test_QC_CreditDeltaSingle(self):
        """Test QC for Credit Delta Single data."""        
        self._run_qc_test(
            "CreditDeltaSingle_Input.csv", 
            CreditDeltaSingleInput(),
            cds_column, 
            qc_engine_presets.preset_reactive_univariate_cds)

    def test_QC_CreditDeltaSingle_with_injections(self):
        """Test QC for Credit Delta Single data with outlier injections."""
        config = CreditDeltaInjectorConfig.cds_preset()
        self._run_qc_test(
            "CreditDeltaSingle_Input.csv",
            CreditDeltaSingleInput(),
            cds_column,
            qc_engine_presets.preset_reactive_univariate_cds,
            injector=CreditDeltaOutlierInjector(config=config, severity=INJECTION_SEVERITY),
            inject=True)

    def test_QC_CreditDeltaIndex(self):
        """Test QC for Credit Delta Index data."""        
        self._run_qc_test(
            "CreditDeltaIndex_Input.csv", 
            CreditDeltaIndexInput(), 
            cdi_column, 
            qc_engine_presets.preset_robust_univariate_cdi)

    def test_QC_CreditDeltaIndex_with_injections(self):
        """Test QC for Credit Delta Index data with outlier injections."""
        config = CreditDeltaInjectorConfig.credit_delta_index_preset()
        self._run_qc_test(
            "CreditDeltaIndex_Input_Train-OOS.csv",
            CreditDeltaIndexInput(),
            cdi_column,
            qc_engine_presets.preset_robust_univariate_cdi,
            injector=CreditDeltaOutlierInjector(config=config, severity=INJECTION_SEVERITY),
            inject=True)

if __name__ == '__main__':
    unittest.main()