from abc import ABC, abstractmethod
import pandas as pd

class QCMethod(ABC):
    """
    Abstract base for all QC methods.
    Contract:
      - fit(train_df): compute/fit state using TRAIN window only (no leakage).
      - score_day(day_df): return a pd.Series or 1-column DataFrame of scores in [0,1]
                           aligned to day_df.index.
    """

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def score_day(self, day_df: pd.DataFrame) -> pd.Series:
        pass