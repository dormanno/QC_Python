import unittest

import numpy as np
import pandas as pd

from column_names import qc_column
from Engine.aggregator import ScoreAggregator, ConsensusMode


class TestScoreAggregator(unittest.TestCase):
    def test_combine_weighted_and_nan(self) -> None:
        weights = {
            qc_column.IF_SCORE: 0.5,
            qc_column.ROBUST_Z_SCORE: 0.3,
            qc_column.ROLLING_SCORE: 0.2,
        }
        aggregator = ScoreAggregator(weights=weights)

        df = pd.DataFrame({
            qc_column.IF_SCORE: [0.0, 0.2],
            qc_column.ROBUST_Z_SCORE: [0.5, 0.8],
            qc_column.ROLLING_SCORE: [1.0, np.nan],
        })

        out = aggregator.combine(df)

        expected_row_0 = 0.0 * 0.5 + 0.5 * 0.3 + 1.0 * 0.2
        self.assertAlmostEqual(out.iloc[0], expected_row_0, places=6)

        # ROLLING_SCORE is NaN on row 1 so weights renormalize (0.5/0.8, 0.3/0.8)
        expected_row_1 = 0.2 * (0.5 / 0.8) + 0.8 * (0.3 / 0.8)
        self.assertAlmostEqual(out.iloc[1], expected_row_1, places=6)

    def test_consensus_none_does_not_override(self) -> None:
        weights = {
            qc_column.IF_SCORE: 0.25,
            qc_column.ROBUST_Z_SCORE: 0.25,
            qc_column.ROLLING_SCORE: 0.25,
            qc_column.IQR_SCORE: 0.25,
        }
        aggregator = ScoreAggregator(weights=weights, consensus=ConsensusMode.NONE)

        df = pd.DataFrame({
            qc_column.IF_SCORE: [1.0],
            qc_column.ROBUST_Z_SCORE: [1.0],
            qc_column.ROLLING_SCORE: [0.0],
            qc_column.IQR_SCORE: [0.0],
        })

        out = aggregator.combine(df)
        self.assertAlmostEqual(out.iloc[0], 0.5, places=6)

    def test_consensus_simple_majority(self) -> None:
        weights = {
            qc_column.IF_SCORE: 0.25,
            qc_column.ROBUST_Z_SCORE: 0.25,
            qc_column.ROLLING_SCORE: 0.25,
            qc_column.IQR_SCORE: 0.25,
        }
        aggregator = ScoreAggregator(weights=weights, consensus=ConsensusMode.SIMPLE_MAJORITY)

        df = pd.DataFrame({
            qc_column.IF_SCORE: [1.0, 1.0],
            qc_column.ROBUST_Z_SCORE: [1.0, 0.0],
            qc_column.ROLLING_SCORE: [0.0, 0.0],
            qc_column.IQR_SCORE: [0.0, 0.0],
        })

        out = aggregator.combine(df)
        self.assertAlmostEqual(out.iloc[0], 1.0, places=6)
        self.assertAlmostEqual(out.iloc[1], 0.25, places=6)

    def test_consensus_qualified_majority(self) -> None:
        weights = {
            qc_column.IF_SCORE: 0.25,
            qc_column.ROBUST_Z_SCORE: 0.25,
            qc_column.ROLLING_SCORE: 0.25,
            qc_column.IQR_SCORE: 0.25,
        }
        aggregator = ScoreAggregator(weights=weights, consensus=ConsensusMode.QUALIFIED_MAJORITY)

        df = pd.DataFrame({
            qc_column.IF_SCORE: [1.0, 1.0],
            qc_column.ROBUST_Z_SCORE: [1.0, 1.0],
            qc_column.ROLLING_SCORE: [1.0, 0.0],
            qc_column.IQR_SCORE: [0.0, 0.0],
        })

        out = aggregator.combine(df)
        self.assertAlmostEqual(out.iloc[0], 1.0, places=6)
        self.assertAlmostEqual(out.iloc[1], 0.5, places=6)

    def test_consensus_none_five_methods(self) -> None:
        weights = {
            qc_column.IF_SCORE: 0.2,
            qc_column.ROBUST_Z_SCORE: 0.2,
            qc_column.ROLLING_SCORE: 0.2,
            qc_column.IQR_SCORE: 0.2,
            qc_column.LOF_SCORE: 0.2,
        }
        aggregator = ScoreAggregator(weights=weights, consensus=ConsensusMode.NONE)

        df = pd.DataFrame({
            qc_column.IF_SCORE: [1.0],
            qc_column.ROBUST_Z_SCORE: [1.0],
            qc_column.ROLLING_SCORE: [1.0],
            qc_column.IQR_SCORE: [0.0],
            qc_column.LOF_SCORE: [0.0],
        })

        out = aggregator.combine(df)
        self.assertAlmostEqual(out.iloc[0], 0.6, places=6)

    def test_consensus_simple_majority_five_methods(self) -> None:
        weights = {
            qc_column.IF_SCORE: 0.2,
            qc_column.ROBUST_Z_SCORE: 0.2,
            qc_column.ROLLING_SCORE: 0.2,
            qc_column.IQR_SCORE: 0.2,
            qc_column.LOF_SCORE: 0.2,
        }
        aggregator = ScoreAggregator(weights=weights, consensus=ConsensusMode.SIMPLE_MAJORITY)

        df = pd.DataFrame({
            qc_column.IF_SCORE: [1.0, 1.0],
            qc_column.ROBUST_Z_SCORE: [1.0, 0.0],
            qc_column.ROLLING_SCORE: [1.0, 0.0],
            qc_column.IQR_SCORE: [0.0, 0.0],
            qc_column.LOF_SCORE: [0.0, 0.0],
        })

        out = aggregator.combine(df)
        self.assertAlmostEqual(out.iloc[0], 1.0, places=6)
        self.assertAlmostEqual(out.iloc[1], 0.2, places=6)

    def test_consensus_qualified_majority_five_methods(self) -> None:
        weights = {
            qc_column.IF_SCORE: 0.2,
            qc_column.ROBUST_Z_SCORE: 0.2,
            qc_column.ROLLING_SCORE: 0.2,
            qc_column.IQR_SCORE: 0.2,
            qc_column.LOF_SCORE: 0.2,
        }
        aggregator = ScoreAggregator(weights=weights, consensus=ConsensusMode.QUALIFIED_MAJORITY)

        df = pd.DataFrame({
            qc_column.IF_SCORE: [1.0, 1.0],
            qc_column.ROBUST_Z_SCORE: [1.0, 1.0],
            qc_column.ROLLING_SCORE: [1.0, 1.0],
            qc_column.IQR_SCORE: [1.0, 0.0],
            qc_column.LOF_SCORE: [0.0, 0.0],
        })

        out = aggregator.combine(df)
        self.assertAlmostEqual(out.iloc[0], 1.0, places=6)
        self.assertAlmostEqual(out.iloc[1], 0.6, places=6)


if __name__ == "__main__":
    unittest.main()
