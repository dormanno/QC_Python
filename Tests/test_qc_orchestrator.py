import unittest
import os
import pandas as pd
import tempfile
import shutil
from qc_orchestrator import QCOrchestrator
from Engine.qc_engine import QCEngine
from Engine.feature_normalizer import FeatureNormalizer
from input import PnLInput, CreditDeltaSingleInput, CreditDeltaIndexInput
from column_names import pnl_column, cds_column, qc_column, cdi_column

ORIGINAL_INPUT_DIRECTORY = r"C:\Users\dorma\Documents\UEK_Backup\Test"
# Define aggregator weights
WEIGHT_IF = 0.2
WEIGHT_RZ = 0.2
WEIGHT_ROLL = 0.2
WEIGHT_IQR = 0.2
WEIGHT_LOF = 0.2

ROLL_WINDOW = 20

class TestQCOrchestrator(unittest.TestCase):
    def _run_qc_test(self, columnSet, original_input_file, input_handler):
        """Helper method to run QC test with specified column configuration and input file."""
        # Define features for QC
        qc_features = columnSet.QC_FEATURES
        
        # Create feature normalizer
        normalizer = FeatureNormalizer(features=qc_features)
        
        # Create QC Engine
        qc_engine = QCEngine(
            qc_features=qc_features,
            weight_if=WEIGHT_IF,
            weight_rz=WEIGHT_RZ,
            weight_roll=WEIGHT_ROLL,
            weight_iqr=WEIGHT_IQR,
            weight_lof=WEIGHT_LOF,
            roll_window=ROLL_WINDOW
        )
        
        
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

            # Run the orchestrator
            try:
                orchestrator = QCOrchestrator(
                    normalizer=normalizer,
                    qc_engine=qc_engine,
                    input_handler=input_handler
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
            expected_cols = input_cols + columnSet.ENGINEERED_FEATURES.__len__() + qc_column.SCORE_COLUMNS.__len__()
            self.assertEqual(len(output_df.columns), expected_cols, f"Output columns {len(output_df.columns)} != expected {expected_cols}")

            # Copy output back to original directory for review
            shutil.copy2(output_path, os.path.join(ORIGINAL_INPUT_DIRECTORY, os.path.basename(output_path)))

    def test_QC_PnL(self):
        """Test QC for PnL data."""
        self._run_qc_test(pnl_column, "PnL_Input2.csv", PnLInput())
    
    def test_QC_CreditDeltaSingle(self):
        """Test QC for Credit Delta Single data."""        
        self._run_qc_test(cds_column, "CreditDeltaSingle_Input.csv", CreditDeltaSingleInput())

    def test_QC_CreditDeltaIndex(self):
        """Test QC for Credit Delta Index data."""        
        self._run_qc_test(cdi_column, "CreditDeltaIndex_Input.csv", CreditDeltaIndexInput())

if __name__ == '__main__':
    unittest.main()