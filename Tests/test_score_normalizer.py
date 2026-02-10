import unittest

import numpy as np
import pandas as pd

from Engine.score_normalizer import ScoreNormalizer
from Engine.qc_engine import QCEngine
from column_names import main_column, qc_column
from QC_methods.qc_method_definitions import QCMethodDefinitions


class TestScoreNormalizer(unittest.TestCase):
    def test_ecdf_quantiles_and_nan(self):
        normalizer = ScoreNormalizer()
        train_df = pd.DataFrame({"A": [10, 20, 30, 40]})
        normalizer.fit(train_df)

        scores_df = pd.DataFrame({"A": [25, 10, 40, 5, 100, np.nan]})
        out = normalizer.transform(scores_df)

        expected = [0.5, 0.25, 1.0, 0.0, 1.0, np.nan]
        for i, val in enumerate(expected):
            if np.isnan(val):
                self.assertTrue(np.isnan(out["A"].iloc[i]))
            else:
                self.assertAlmostEqual(out["A"].iloc[i], val, places=6)


class DummyIFMethod:
    ScoreName = qc_column.IF_SCORE

    def fit(self, train_df: pd.DataFrame) -> None:
        return None

    def score_day(self, day_df: pd.DataFrame) -> pd.Series:
        return day_df["X"].rename(self.ScoreName)


class DummyECDFMethod:
    ScoreName = qc_column.ECDF_SCORE

    def fit(self, train_df: pd.DataFrame) -> None:
        return None

    def score_day(self, day_df: pd.DataFrame) -> pd.Series:
        return pd.Series(0.2, index=day_df.index, name=self.ScoreName)


class TestQCEngineNormalization(unittest.TestCase):
    def test_normalization_applied_before_aggregation(self):
        methods_config = {
            QCMethodDefinitions.ISOLATION_FOREST: 0.5,
            QCMethodDefinitions.ECDF: 0.5,
        }
        engine = QCEngine(
            qc_features=["X"],
            methods_config=methods_config,
            roll_window=2,
            score_normalizer=ScoreNormalizer()
        )

        # Override methods with deterministic dummies
        engine.qc_methods = {
            "if": DummyIFMethod(),
            "ecdf": DummyECDFMethod(),
        }
        engine._skip_normalization_cols = {qc_column.ECDF_SCORE}

        train_df = pd.DataFrame({
            main_column.DATE: ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"],
            main_column.TRADE: [1, 2, 1, 2],
            "X": [0.0, 1.0, 2.0, 3.0],
        })

        engine.fit(train_df)

        day_df = pd.DataFrame({
            main_column.DATE: ["2020-01-03", "2020-01-03"],
            main_column.TRADE: [1, 2],
            "X": [1.5, 3.0],
        })

        day_scores, aggregated, _ = engine.score_day(day_df)

        # IF scores should be normalized via ECDF of [0,1,2,3]
        expected_if = [0.5, 1.0]
        self.assertAlmostEqual(day_scores[qc_column.IF_SCORE].iloc[0], expected_if[0], places=6)
        self.assertAlmostEqual(day_scores[qc_column.IF_SCORE].iloc[1], expected_if[1], places=6)

        # ECDF scores should be left unchanged
        self.assertTrue((day_scores[qc_column.ECDF_SCORE] == 0.2).all())

        # Aggregation should use normalized IF scores
        expected_agg = [0.5 * 0.5 + 0.5 * 0.2, 0.5 * 1.0 + 0.5 * 0.2]
        self.assertAlmostEqual(aggregated.iloc[0], expected_agg[0], places=6)
        self.assertAlmostEqual(aggregated.iloc[1], expected_agg[1], places=6)


if __name__ == "__main__":
    unittest.main()
