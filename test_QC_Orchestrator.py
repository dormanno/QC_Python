import unittest
import os
import pandas as pd
import tempfile
import shutil
from QC_Orchestrator import run_qc_orchestrator
import InputOutput as IO

class TestQCOrchestrator(unittest.TestCase):
    def test_main_function(self):
        original_input_path = r"C:\Users\dorma\Documents\UEK_Backup\Test\PnL_Input.csv"

        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Temporary directory created at: {temp_dir}")
            # Copy the input file to temp directory
            temp_input_path = os.path.join(temp_dir, "PnL_Input.csv")
            shutil.copy2(original_input_path, temp_input_path)

            # Read raw input to get baseline (before engineering)
            raw_input_df = pd.read_csv(temp_input_path)
            input_rows = len(raw_input_df)
            input_cols = len(raw_input_df.columns)

            # Run the orchestrator
            try:
                output_path = run_qc_orchestrator(temp_input_path)
                # 1. Run succeeded (no exception)
                self.assertIsInstance(output_path, str)
            except Exception as e:
                self.fail(f"run_qc_orchestrator raised an exception: {e}")

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

if __name__ == '__main__':
    unittest.main()