import pandas as pd
from typing import Literal
from column_names import qc_column

FLAG = Literal["GREEN", "AMBER", "RED"]

class ScoreAggregator:
    """
    Linear weighted aggregator; weights sum to 1.
    """

    def __init__(self,
                 weight_if: float,
                 weight_rz: float,
                 weight_roll: float,
                 weight_iqr: float,
                 *,
                 amber_lo: float = 0.85,
                 red_lo: float = 0.95
                 ):
        s = weight_if + weight_rz + weight_roll + weight_iqr
        if abs(s - 1.0) > 1e-9:
            raise ValueError("Weights must sum to 1.")
        if not (0.0 <= amber_lo < red_lo <= 1.0):
            raise ValueError("Require 0 ≤ amber_lo < red_lo ≤ 1.")
        self.weight_if = weight_if
        self.weight_rz = weight_rz
        self.weight_roll = weight_roll
        self.weight_iqr = weight_iqr
        self.amber_lo, self.red_lo = amber_lo, red_lo

    def combine(self, df: pd.DataFrame) -> pd.Series:
        # expects columns: IF_score, RobustZ_score, Rolling_score, IQR_score
        return (
            self.weight_if   * df[qc_column.IF_SCORE]
          + self.weight_rz   * df[qc_column.ROBUST_Z_SCORE]
          + self.weight_roll * df[qc_column.ROLLING_SCORE]
          + self.weight_iqr  * df[qc_column.IQR_SCORE]
        ).rename(qc_column.AGGREGATED_SCORE)

    def map_to_flag(self, agg: pd.Series) -> pd.Series:
        """
        Map aggregated anomaly intensity in [0,1] to traffic-light flags:
          [0, amber_lo)  -> GREEN
          [amber_lo, red_lo) -> AMBER
          [red_lo, 1]   -> RED
        Returns a pandas Categorical with ordered categories (GREEN < AMBER < RED).
        """
        cats = pd.cut(
            agg.astype(float),
            bins=[-1e-12, self.amber_lo, self.red_lo, 1.0 + 1e-12],
            labels=["GREEN", "AMBER", "RED"],
            right=False, include_lowest=True
        ).astype("category")
        cats = cats.cat.set_categories(["GREEN", "AMBER", "RED"], ordered=True)
        return cats.rename(qc_column.QC_FLAG)
