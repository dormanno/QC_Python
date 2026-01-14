import unittest
import os
import pandas as pd
import tempfile
import shutil
from QC_Orchestrator import QCOrchestrator
import InputOutput as IO
import ColumnNames as Column

class TestQCOrchestrator(unittest.TestCase):
    def test_QC_PnL(self):
        # Define features for QC
        qc_features = Column.PNL_FEATURES
        
        original_input_directory = r"C:\Users\dorma\Documents\UEK_Backup\Test"
        original_input_file = "PnL_Input2.csv"
        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Temporary directory created at: {temp_dir}")
            # Copy the input file to temp directory
            temp_input_path = os.path.join(temp_dir, original_input_file)
            shutil.copy2(os.path.join(original_input_directory, original_input_file), temp_input_path)

            # Read raw input to get baseline (before engineering)
            raw_input_df = pd.read_csv(temp_input_path)
            input_rows = len(raw_input_df)
            input_cols = len(raw_input_df.columns)

            # Run the orchestrator
            try:
                orchestrator = QCOrchestrator(qc_features=qc_features)
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

            # 5. Output file has 11 more columns than input
            expected_cols = input_cols + 11
            self.assertEqual(len(output_df.columns), expected_cols, f"Output columns {len(output_df.columns)} != expected {expected_cols}")

            # Copy output back to original directory for review
            shutil.copy2(output_path, os.path.join(original_input_directory, os.path.basename(output_path)))

if __name__ == '__main__':
    unittest.main()