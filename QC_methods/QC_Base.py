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

    RequiresNormalization: bool = True

    def __init__(self, score_name: str):
        self._score_name = score_name

    @property
    def ScoreName(self) -> str:
        return self._score_name

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def _score_day_impl(self, day_df: pd.DataFrame) -> pd.Series:
        """Actual scoring implementation - to be overridden by subclasses."""
        pass
    
    def score_day(self, day_df: pd.DataFrame) -> pd.Series:
        """Public method with validation - not overridden by subclasses."""
        assert len(day_df) > 0, "day_df must not be empty"
        assert day_df["Date"].nunique() == 1, "day_df must contain exactly one valuation date"
        return self._score_day_impl(day_df)


class StatefulQCMethod(QCMethod):
    """
    Interface for stateful QC methods that maintain state across scoring iterations.
    
    Stateful methods need to update their internal state after scoring each day
    to maintain rolling windows or other temporal dependencies.
    """
    
    @abstractmethod
    def _update_state_impl(self, day_df: pd.DataFrame) -> None:
        """Actual state update implementation - to be overridden by subclasses."""
        pass
    
    def update_state(self, day_df: pd.DataFrame) -> None:
        """Update internal state after scoring a day's data.
        
        This method should be called AFTER score_day() to prevent look-ahead bias.
        
        Args:
            day_df (pd.DataFrame): DataFrame containing the day's data to incorporate into state.
        """
        assert len(day_df) > 0, "day_df must not be empty"
        self._update_state_impl(day_df)