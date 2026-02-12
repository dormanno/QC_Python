import unittest
import os
import pandas as pd
import tempfile
import shutil
from Engine import qc_engine_presets
from qc_orchestrator import QCOrchestrator
from Engine.feature_normalizer import FeatureNormalizer
from IO.input import PnLInput, CreditDeltaSingleInput, CreditDeltaIndexInput
from column_names import pnl_column, cds_column, cdi_column, main_column
from QC_methods.qc_method_definitions import QCMethodDefinitions

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

class TestQCOrchestrator(unittest.TestCase):
    
    def _run_qc_test(self, original_input_file, input_handler, columnSet, engine_preset):
        """Helper method to run QC test with specified column configuration and input file.
        
        Args:            
            original_input_file: Input file name
            input_handler: Input handler instance
            columnSet: Column set instance
            engine_preset: QCEnginePreset instance
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
                orchestrator = QCOrchestrator(
                    normalizer=normalizer,
                    engine_preset=engine_preset,
                    input_handler=input_handler,
                    split_identifier=main_column.TRADE_TYPE
                )
                output_path = orchestrator.run(temp_input_path)
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
            shutil.copy2(output_path, os.path.join(ORIGINAL_INPUT_DIRECTORY, os.path.basename(output_path)))

    def test_QC_PnL(self):
        """Test QC for PnL data."""
        self._run_qc_test(
            # "PnL_Input2.csv", 
            "PnL_Input_Injected.csv",
            PnLInput(), 
            pnl_column, 
            qc_engine_presets.preset_temporal_multivariate_pnl)
    
    def test_QC_CreditDeltaSingle(self):
        """Test QC for Credit Delta Single data."""        
        self._run_qc_test(
            "CreditDeltaSingle_Input.csv", 
            CreditDeltaSingleInput(),
            cds_column, 
            qc_engine_presets.preset_reactive_univariate_cds)

    def test_QC_CreditDeltaIndex(self):
        """Test QC for Credit Delta Index data."""        
        self._run_qc_test(
            "CreditDeltaIndex_Input.csv", 
            CreditDeltaIndexInput(), 
            cdi_column, 
            qc_engine_presets.preset_robust_univariate_cdi)
if __name__ == '__main__':
    unittest.main()