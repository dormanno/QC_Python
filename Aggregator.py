import pandas as pd
from typing import Literal
import ColumnNames as Column

FLAG = Literal["GREEN", "AMBER", "RED"]

class ScoreAggregator:
    """
    Linear weighted aggregator; weights sum to 1.
    """

    def __init__(self,
                 w_if: float = 0.4,
                 w_rz: float = 0.3,
                 w_roll: float = 0.2,
                 w_iqr: float = 0.1,
                 *,
                 amber_lo: float = 0.85,
                 red_lo: float = 0.95
                 ):
        s = w_if + w_rz + w_roll + w_iqr
        if abs(s - 1.0) > 1e-9:
            raise ValueError("Weights must sum to 1.")
        if not (0.0 <= amber_lo < red_lo <= 1.0):
            raise ValueError("Require 0 ≤ amber_lo < red_lo ≤ 1.")
        self.w_if, self.w_rz, self.w_roll, self.w_iqr = w_if, w_rz, w_roll, w_iqr
        self.amber_lo, self.red_lo = amber_lo, red_lo

    def combine(self, df: pd.DataFrame) -> pd.Series:
        # expects columns: IF_score, RobustZ_score, Rolling_score, IQR_score
        return (
            self.w_if   * df[Column.IF_SCORE]
          + self.w_rz   * df[Column.ROBUST_Z_SCORE]
          + self.w_roll * df[Column.ROLLING_SCORE]
          + self.w_iqr  * df[Column.IQR_SCORE]
        ).rename(Column.AGGREGATED_SCORE)

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
        return cats.rename(Column.SCORE_FLAG)
