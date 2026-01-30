"""
Unit tests for individual QC methods using PNL_Input2.csv
Tests each QC method (IQR, Isolation Forest, LOF, Robust Z-Score, Rolling Z-Score) separately
to validate their fit(), score_day(), and update_state() functionality.
"""

import unittest
import os
import pandas as pd
import numpy as np

from QC_methods import IsolationForestQC, RobustZScoreQC, IQRQC, RollingZScoreQC, LOFQC, ECDFQC, HampelFilterQC

from QC_methods.qc_base import StatefulQCMethod
from column_names import main_column, pnl_column, qc_column
from IO.input import PnLInput

# Test data path
ORIGINAL_INPUT_DIRECTORY = r"C:\Users\dorma\Documents\UEK_Backup\Test"
TEST_INPUT_FILE = "PNL_Input2.csv"

# Test configuration - using entire dataset
# Split: first 2/3 for training, last 1/3 for OOS testing
TRAIN_SPLIT = 2/3  # Proportion of data for training
ROLL_WINDOW = 20


class BaseQCMethodTest(unittest.TestCase):
    """Base test class with shared setup for all QC method tests."""
    
    @classmethod
    def setUpClass(cls):
        """Load and prepare test data once for all tests."""
        input_path = os.path.join(ORIGINAL_INPUT_DIRECTORY, TEST_INPUT_FILE)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Test input file not found: {input_path}")
        
        # Read and engineer features
        input_handler = PnLInput()
        cls.raw_df = pd.read_csv(input_path)
        cls.engineered_df = input_handler.input_post_process(cls.raw_df)
        
        # Get QC features
        cls.qc_features = pnl_column.QC_FEATURES
        cls.identity_column = main_column.TRADE
        cls.temporal_column = main_column.DATE
        
        # Sort by date
        cls.engineered_df = cls.engineered_df.sort_values(cls.temporal_column).reset_index(drop=True)
        
        # Get unique dates
        cls.unique_dates = sorted(cls.engineered_df[cls.temporal_column].unique())
        
        # Split into train and test using 2/3 - 1/3 split
        train_size = int(len(cls.unique_dates) * TRAIN_SPLIT)
        cls.train_dates = cls.unique_dates[:train_size]
        cls.test_dates = cls.unique_dates[train_size:]  # All remaining dates for OOS
        
        cls.train_df = cls.engineered_df[cls.engineered_df[cls.temporal_column].isin(cls.train_dates)].copy()
        cls.test_dfs = [
            cls.engineered_df[cls.engineered_df[cls.temporal_column] == date].copy()
            for date in cls.test_dates
        ]
        
        print(f"\nTest data loaded:")
        print(f"  Total rows: {len(cls.engineered_df)}")
        print(f"  Total dates: {len(cls.unique_dates)}")
        print(f"  Train dates: {len(cls.train_dates)} ({len(cls.train_df)} rows)")
        print(f"  Test dates: {len(cls.test_dates)} ({sum(len(df) for df in cls.test_dfs)} rows)")
        print(f"  QC features: {cls.qc_features}")


class TestIndividualQCMethods(BaseQCMethodTest):
    """Test individual QC methods using PNL_Input2.csv"""
    
    def _validate_scores(self, scores: pd.Series, day_df: pd.DataFrame, method_name: str):
        """Helper method to validate score output."""
        # 1. Scores should be a Series
        self.assertIsInstance(scores, pd.Series, f"{method_name}: scores should be a pandas Series")
        
        # 2. Scores should have same length as input
        self.assertEqual(len(scores), len(day_df), 
                        f"{method_name}: scores length {len(scores)} != input length {len(day_df)}")
        
        # 3. Scores should be numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(scores), 
                       f"{method_name}: scores should be numeric")
        
        # 4. Scores should be in [0,1] range
        self.assertTrue((scores >= 0).all() and (scores <= 1).all(), 
                       f"{method_name}: scores should be in [0,1] range. Got min={scores.min()}, max={scores.max()}")
        
        # 5. No NaN values in scores
        self.assertFalse(scores.isna().any(), 
                        f"{method_name}: scores contain NaN values")
    
    def test_iqr_qc_method(self):
        """Test IQR QC method."""
        print("\n=== Testing IQR QC Method ===")
        
        # Initialize method
        method = IQRQC(
            features=self.qc_features,
            identity_column=self.identity_column,
            score_name=qc_column.IQR_SCORE
        )
        
        # Fit on training data
        method.fit(self.train_df)
        
        # Validate fit results
        self.assertIsNotNone(method.q1, "IQR: Q1 should be computed after fit")
        self.assertIsNotNone(method.q3, "IQR: Q3 should be computed after fit")
        self.assertIsInstance(method.q1, pd.DataFrame, "IQR: Q1 should be a DataFrame")
        self.assertIsInstance(method.q3, pd.DataFrame, "IQR: Q3 should be a DataFrame")
        
        print(f"  Fitted on {len(self.train_df)} training samples")
        print(f"  Q1 shape: {method.q1.shape}, Q3 shape: {method.q3.shape}")
        
        # Score all OOS test days
        all_scores = []
        for i, day_df in enumerate(self.test_dfs):
            scores = method.score_day(day_df)
            self._validate_scores(scores, day_df, f"IQR Day {i+1}")
            all_scores.extend(scores.tolist())
        
        print(f"  Scored {len(self.test_dfs)} OOS days, {sum(len(df) for df in self.test_dfs)} total samples")
        print(f"  Overall mean score: {np.mean(all_scores):.4f}, max: {np.max(all_scores):.4f}")
    
    def test_isolation_forest_qc_method(self):
        """Test Isolation Forest QC method."""
        print("\n=== Testing Isolation Forest QC Method ===")
        
        # Initialize method
        method = IsolationForestQC(
            base_feats=self.qc_features,
            identity_column=self.identity_column,
            temporal_column=self.temporal_column,
            score_name=qc_column.IF_SCORE,
            mode="time_series",
            per_trade_normalize=False,
            use_robust_scaler=True,
            n_estimators=200,
            max_samples=256,
            contamination=0.01,
        )
        
        # Fit on training data
        method.fit(self.train_df)
        
        # Validate fit results
        self.assertIsNotNone(method._clf, "IF: model should be fitted")
        
        print(f"  Fitted on {len(self.train_df)} training samples")
        
        # Score all OOS test days
        all_scores = []
        for i, day_df in enumerate(self.test_dfs):
            scores = method.score_day(day_df)
            self._validate_scores(scores, day_df, f"IF Day {i+1}")
            all_scores.extend(scores.tolist())
        
        print(f"  Scored {len(self.test_dfs)} OOS days, {sum(len(df) for df in self.test_dfs)} total samples")
        print(f"  Overall mean score: {np.mean(all_scores):.4f}, max: {np.max(all_scores):.4f}")
    
    def test_robust_z_score_qc_method(self):
        """Test Robust Z-Score QC method."""
        print("\n=== Testing Robust Z-Score QC Method ===")
        
        # Initialize method
        method = RobustZScoreQC(
            features=self.qc_features,
            identity_column=self.identity_column,
            score_name=qc_column.ROBUST_Z_SCORE,
            z_cap=6.0
        )
        
        # Fit on training data
        method.fit(self.train_df)
        
        # Validate fit results
        self.assertIsNotNone(method.median, "RZ: median should be computed after fit")
        self.assertIsNotNone(method.mad, "RZ: MAD should be computed after fit")
        self.assertIsInstance(method.median, pd.DataFrame, "RZ: median should be a DataFrame")
        self.assertIsInstance(method.mad, pd.DataFrame, "RZ: MAD should be a DataFrame")
        
        print(f"  Fitted on {len(self.train_df)} training samples")
        print(f"  Median shape: {method.median.shape}, MAD shape: {method.mad.shape}")
        
        # Score all OOS test days
        all_scores = []
        for i, day_df in enumerate(self.test_dfs):
            scores = method.score_day(day_df)
            self._validate_scores(scores, day_df, f"RZ Day {i+1}")
            all_scores.extend(scores.tolist())
        
        print(f"  Scored {len(self.test_dfs)} OOS days, {sum(len(df) for df in self.test_dfs)} total samples")
        print(f"  Overall mean score: {np.mean(all_scores):.4f}, max: {np.max(all_scores):.4f}")
    
    def test_rolling_z_score_qc_method(self):
        """Test Rolling Z-Score QC method (stateful)."""
        print("\n=== Testing Rolling Z-Score QC Method ===")
        
        # Initialize method
        method = RollingZScoreQC(
            features=self.qc_features,
            identity_column=self.identity_column,
            temporal_column=self.temporal_column,
            score_name=qc_column.ROLLING_SCORE,
            window=ROLL_WINDOW,
            z_cap=6.0
        )
        
        # Verify it's a stateful method
        self.assertIsInstance(method, StatefulQCMethod, "Rolling: should be a StatefulQCMethod")
        
        # Fit on training data (warm up buffers)
        method.fit(self.train_df)
        
        # Validate fit results
        self.assertIsNotNone(method.buffers, "Rolling: buffers should be initialized after fit")
        self.assertGreater(len(method.buffers), 0, "Rolling: buffers should contain trade data after fit")
        
        print(f"  Fitted on {len(self.train_df)} training samples")
        print(f"  Number of trades in buffers: {len(method.buffers)}")
        
        # Score all OOS test days and update state
        all_scores = []
        for i, day_df in enumerate(self.test_dfs):
            scores = method.score_day(day_df)
            self._validate_scores(scores, day_df, f"Rolling Day {i+1}")
            all_scores.extend(scores.tolist())
            
            # Update state for next day
            method.update_state(day_df)
        
        print(f"  Scored {len(self.test_dfs)} OOS days, {sum(len(df) for df in self.test_dfs)} total samples")
        print(f"  Overall mean score: {np.mean(all_scores):.4f}, max: {np.max(all_scores):.4f}")
    
    def test_lof_qc_method(self):
        """Test Local Outlier Factor (LOF) QC method (stateful)."""
        print("\n=== Testing LOF QC Method ===")
        
        # Initialize method
        method = LOFQC(
            features=self.qc_features,
            identity_column=self.identity_column,
            score_name=qc_column.LOF_SCORE,
            n_neighbors=20,
            max_window_size=100,
            contamination=0.1,
            use_robust_scaler=True
        )
        
        # Verify it's a stateful method
        self.assertIsInstance(method, StatefulQCMethod, "LOF: should be a StatefulQCMethod")
        
        # Fit on training data
        method.fit(self.train_df)
        
        # Validate fit results
        self.assertIsNotNone(method._lof, "LOF: model should be initialized after fit")
        
        print(f"  Fitted on {len(self.train_df)} training samples")
        
        # Score all OOS test days and update state
        all_scores = []
        for i, day_df in enumerate(self.test_dfs):
            scores = method.score_day(day_df)
            self._validate_scores(scores, day_df, f"LOF Day {i+1}")
            all_scores.extend(scores.tolist())
            
            # Update state for next day
            method.update_state(day_df)
        
        print(f"  Scored {len(self.test_dfs)} OOS days, {sum(len(df) for df in self.test_dfs)} total samples")
        print(f"  Overall mean score: {np.mean(all_scores):.4f}, max: {np.max(all_scores):.4f}")
    
    def test_ecdf_qc_method(self):
        """Test ECDF QC method (stateful)."""
        print("\n=== Testing ECDF QC Method ===")
        
        # Initialize method
        method = ECDFQC(
            features=self.qc_features,
            score_name=qc_column.ECDF_SCORE,
            window=ROLL_WINDOW,
            min_samples=30
        )
        
        # Verify it's a stateful method
        self.assertIsInstance(method, StatefulQCMethod, "ECDF: should be a StatefulQCMethod")
        
        # Fit on training data (initialize history)
        method.fit(self.train_df)
        
        print(f"  Fitted on {len(self.train_df)} training samples")
        
        # Score all OOS test days and update state
        all_scores = []
        for i, day_df in enumerate(self.test_dfs):
            scores = method.score_day(day_df)
            self._validate_scores(scores, day_df, f"ECDF Day {i+1}")
            all_scores.extend(scores.tolist())
            
            # Update state for next day
            method.update_state(day_df)
        
        print(f"  Scored {len(self.test_dfs)} OOS days, {sum(len(df) for df in self.test_dfs)} total samples")
        print(f"  Overall mean score: {np.mean(all_scores):.4f}, max: {np.max(all_scores):.4f}")
    
    def test_hampel_filter_qc_method(self):
        """Test Hampel Filter QC method (stateful)."""
        print("\n=== Testing Hampel Filter QC Method ===")
        
        # Initialize method
        method = HampelFilterQC(
            features=self.qc_features,
            identity_column=self.identity_column,
            temporal_column=self.temporal_column,
            score_name=qc_column.HAMPEL_SCORE,
            window=ROLL_WINDOW,
            threshold=3.0
        )
        
        # Verify it's a stateful method
        self.assertIsInstance(method, StatefulQCMethod, "Hampel: should be a StatefulQCMethod")
        
        # Fit on training data (warm up buffers)
        method.fit(self.train_df)
        
        # Validate fit results
        self.assertIsInstance(method.buffers, dict, "Hampel: buffers should be a dict")
        print(f"  Fitted on {len(self.train_df)} training samples")
        print(f"  Buffers initialized for {len(method.buffers)} trades")
        
        # Score all OOS test days and update state
        all_scores = []
        for i, day_df in enumerate(self.test_dfs):
            scores = method.score_day(day_df)
            self._validate_scores(scores, day_df, f"Hampel Day {i+1}")
            all_scores.extend(scores.tolist())
            
            # Update state for next day
            method.update_state(day_df)
        
        print(f"  Scored {len(self.test_dfs)} OOS days, {sum(len(df) for df in self.test_dfs)} total samples")
        print(f"  Overall mean score: {np.mean(all_scores):.4f}, max: {np.max(all_scores):.4f}")



class TestQCMethodIntegration(BaseQCMethodTest):
    """Integration tests for QC methods: cross-method validation and edge cases."""
    
    def test_all_methods_produce_different_scores(self):
        """Test that different methods produce different scores (sanity check)."""
        print("\n=== Testing Score Diversity Across Methods ===")
        
        # Initialize all methods
        methods = {
            'IQR': IQRQC(
                features=self.qc_features,
                identity_column=self.identity_column,
                score_name=qc_column.IQR_SCORE
            ),
            'IF': IsolationForestQC(
                base_feats=self.qc_features,
                identity_column=self.identity_column,
                temporal_column=self.temporal_column,
                score_name=qc_column.IF_SCORE,
                mode="time_series",
                per_trade_normalize=False,
                use_robust_scaler=True,
                n_estimators=200,
                max_samples=256,
                contamination=0.01,
            ),
            'RZ': RobustZScoreQC(
                features=self.qc_features,
                identity_column=self.identity_column,
                score_name=qc_column.ROBUST_Z_SCORE,
                z_cap=6.0
            ),
            'Rolling': RollingZScoreQC(
                features=self.qc_features,
                identity_column=self.identity_column,
                temporal_column=self.temporal_column,
                score_name=qc_column.ROLLING_SCORE,
                window=ROLL_WINDOW,
                z_cap=6.0
            ),
            'LOF': LOFQC(
                features=self.qc_features,
                identity_column=self.identity_column,
                score_name=qc_column.LOF_SCORE,
                n_neighbors=20,
                max_window_size=100,
                contamination=0.1,
                use_robust_scaler=True
            ),
            'Hampel': HampelFilterQC(
                features=self.qc_features,
                identity_column=self.identity_column,
                temporal_column=self.temporal_column,
                score_name=qc_column.HAMPEL_SCORE,
                window=ROLL_WINDOW,
                threshold=3.0
            )
        }
        
        # Fit all methods
        for name, method in methods.items():
            method.fit(self.train_df)
            print(f"  {name}: fitted")
        
        # Score first test day with all methods
        test_day = self.test_dfs[0]
        scores = {}
        for name, method in methods.items():
            scores[name] = method.score_day(test_day)
            print(f"  {name}: mean={scores[name].mean():.4f}, std={scores[name].std():.4f}")
        
        # Compare all pairs - they should not be identical
        method_names = list(scores.keys())
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                name1, name2 = method_names[i], method_names[j]
                # Check that scores are not identical
                self.assertFalse(
                    (scores[name1] == scores[name2]).all(),
                    f"{name1} and {name2} produced identical scores"
                )
                print(f"  {name1} vs {name2}: correlation = {scores[name1].corr(scores[name2]):.4f}")
    
    def test_methods_handle_single_day_constraint(self):
        """Test that methods properly enforce single-day constraint."""
        print("\n=== Testing Single-Day Constraint ===")
        
        # Create multi-day DataFrame (should fail)
        multi_day_df = pd.concat([self.test_dfs[0], self.test_dfs[1]])
        
        method = IQRQC(
            features=self.qc_features,
            identity_column=self.identity_column,
            score_name=qc_column.IQR_SCORE
        )
        method.fit(self.train_df)
        
        # Should raise assertion error
        with self.assertRaises(AssertionError) as context:
            method.score_day(multi_day_df)
        
        self.assertIn("exactly one valuation date", str(context.exception))
        print("  Multi-day scoring properly rejected")
    
    def test_methods_handle_empty_dataframe(self):
        """Test that methods properly handle empty DataFrames."""
        print("\n=== Testing Empty DataFrame Handling ===")
        
        # Create empty DataFrame with correct columns
        empty_df = pd.DataFrame(columns=self.engineered_df.columns)
        
        method = IQRQC(
            features=self.qc_features,
            identity_column=self.identity_column,
            score_name=qc_column.IQR_SCORE
        )
        method.fit(self.train_df)
        
        # Should raise assertion error
        with self.assertRaises(AssertionError) as context:
            method.score_day(empty_df)
        
        self.assertIn("must not be empty", str(context.exception))
        print("  Empty DataFrame properly rejected")


class TestQCMethodStatefulBehavior(BaseQCMethodTest):
    """Test stateful behavior specific to Rolling and LOF methods."""
    
    @classmethod
    def setUpClass(cls):
        """Load and prepare test data, using only 3 days for stateful behavior tests."""
        super().setUpClass()  # Call parent setup
        # Override test_dates to use only 3 days for faster stateful tests
        train_size = int(len(cls.unique_dates) * TRAIN_SPLIT)
        cls.test_dates = cls.unique_dates[train_size:train_size + 3]
        cls.test_dfs = [
            cls.engineered_df[cls.engineered_df[cls.temporal_column] == date].copy()
            for date in cls.test_dates
        ]
    
    def test_rolling_state_evolution(self):
        """Test that Rolling method's state evolves correctly."""
        print("\n=== Testing Rolling State Evolution ===")
        
        method = RollingZScoreQC(
            features=self.qc_features,
            identity_column=self.identity_column,
            temporal_column=self.temporal_column,
            score_name=qc_column.ROLLING_SCORE,
            window=5,  # Small window for easier tracking
            z_cap=6.0
        )
        
        method.fit(self.train_df)
        
        # Get a specific trade to track
        sample_trade = self.test_dfs[0][self.identity_column].iloc[0]
        sample_feature = self.qc_features[0]
        
        # Check buffer state before scoring
        buffer_size_before = len(method.buffers[sample_trade][sample_feature]) if sample_trade in method.buffers else 0
        print(f"  Trade {sample_trade}, Feature {sample_feature}:")
        print(f"    Buffer size before: {buffer_size_before}")
        
        # Score and update for multiple days
        for i, day_df in enumerate(self.test_dfs):
            scores = method.score_day(day_df)
            method.update_state(day_df)
            
            if sample_trade in method.buffers:
                buffer_size = len(method.buffers[sample_trade][sample_feature])
                print(f"    After day {i+1}: buffer size = {buffer_size}")
        
        # Buffer should not exceed window size
        final_buffer_size = len(method.buffers[sample_trade][sample_feature])
        self.assertLessEqual(final_buffer_size, 5, "Buffer should not exceed window size")
    
    def test_lof_state_evolution(self):
        """Test that LOF method's state evolves correctly."""
        print("\n=== Testing LOF State Evolution ===")
        
        method = LOFQC(
            features=self.qc_features,
            identity_column=self.identity_column,
            score_name=qc_column.LOF_SCORE,
            n_neighbors=5,
            max_window_size=50,
            contamination=0.1,
            use_robust_scaler=True
        )
        
        method.fit(self.train_df)
        
        # Track window size evolution
        initial_window_size = len(method._history_window) if hasattr(method, '_history_window') else 0
        print(f"  Initial window size: {initial_window_size}")
        
        # Score and update for multiple days
        for i, day_df in enumerate(self.test_dfs):
            scores = method.score_day(day_df)
            method.update_state(day_df)
            window_size = len(method._history_window) if hasattr(method, '_history_window') else 0
            print(f"  After day {i+1}: window size = {window_size}")
    
    def test_hampel_state_evolution(self):
        """Test that Hampel Filter method's state evolves correctly."""
        print("\n=== Testing Hampel Filter State Evolution ===")
        
        method = HampelFilterQC(
            features=self.qc_features,
            identity_column=self.identity_column,
            temporal_column=self.temporal_column,
            score_name=qc_column.HAMPEL_SCORE,
            window=5,  # Small window for testing
            threshold=3.0
        )
        
        method.fit(self.train_df)
        
        # Track buffer state for a sample trade
        sample_trade = self.test_dfs[0][self.identity_column].iloc[0]
        sample_feature = self.qc_features[0]
        
        print(f"  Tracking buffer for trade: {sample_trade}, feature: {sample_feature}")
        
        # Score and update for multiple days
        for i, day_df in enumerate(self.test_dfs):
            scores = method.score_day(day_df)
            method.update_state(day_df)
            
            if sample_trade in method.buffers:
                buffer_size = len(method.buffers[sample_trade][sample_feature])
                print(f"    After day {i+1}: buffer size = {buffer_size}")
        
        # Buffer should not exceed window size
        final_buffer_size = len(method.buffers[sample_trade][sample_feature])
        self.assertLessEqual(final_buffer_size, 5, "Buffer should not exceed window size")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
