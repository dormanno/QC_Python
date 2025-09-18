import pandas as pd

class ScoreAggregator:
    """
    Linear weighted aggregator; weights sum to 1.
    """

    def __init__(self,
                 w_if: float = 0.4,
                 w_rz: float = 0.3,
                 w_roll: float = 0.2,
                 w_iqr: float = 0.1):
        s = w_if + w_rz + w_roll + w_iqr
        if abs(s - 1.0) > 1e-9:
            raise ValueError("Weights must sum to 1.")
        self.w_if, self.w_rz, self.w_roll, self.w_iqr = w_if, w_rz, w_roll, w_iqr

    def combine(self, df: pd.DataFrame) -> pd.Series:
        # expects columns: IF_score, RobustZ_score, Rolling_score, IQR_score
        return (
            self.w_if   * df["IF_score"]
          + self.w_rz   * df["RobustZ_score"]
          + self.w_roll * df["Rolling_score"]
          + self.w_iqr  * df["IQR_score"]
        ).rename("QC_Aggregated")
