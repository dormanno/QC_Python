import numpy as np
import pandas as pd
from typing import List

import InputOutput as IO
from ColumnNames import main_column, pnl_column, qc_column

from QC_methods import IsolationForestQC, RobustZQC, IQRQC, RollingZQC
from Aggregator import ScoreAggregator


class QCOrchestrator:
    """Orchestrates the walk-forward QC evaluation pipeline.
    
    Manages QC method instantiation, training, scoring, and aggregation.
    """
    
    # Default configuration
    ROLL_WINDOW = 20
    TRAIN_RATIO = 2 / 3  # Use 2/3 of dates for training
    
    def __init__(self, qc_features: List[str], roll_window: int = None):
        """Initialize the orchestrator with QC features and optional overrides.
        
        Args:
            qc_features (List[str]): Required. Features to use for QC scoring.
            roll_window (int, optional): Rolling window size. Defaults to ROLL_WINDOW.
        """
        self.qc_features = qc_features
        self.roll_window = roll_window or self.ROLL_WINDOW
        self.input_handler = IO.PnlInput()
        self.output_handler = IO.Output()

    def compute_per_trade_denominators(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-trade median absolute denominators for normalization.
        
        Args:
            train_df (pd.DataFrame): Training data.
        
        Returns:
            pd.DataFrame: Denominators by trade and feature.
        """
        den = (train_df.groupby(main_column.TRADE)[self.qc_features]
               .apply(lambda g: g.abs().median())
               .reset_index())
        den.columns = [main_column.TRADE] + [f"{c}__den" for c in self.qc_features]
        return den

    def apply_denominators(self, df: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
        """Apply per-trade denominators to normalize features.
        
        Args:
            df (pd.DataFrame): Data to normalize.
            den (pd.DataFrame): Denominators.
        
        Returns:
            pd.DataFrame: Normalized data.
        """
        if den is None:
            return df
        g = df.merge(den, on=main_column.TRADE, how="left")
        for c in self.qc_features:
            dcol = f"{c}__den"
            g[c] = g[c] / g[dcol].replace(0, np.nan)
        return g.drop(columns=[f"{c}__den" for c in self.qc_features])

    def run(self, input_path: str) -> str:
        """Run the complete QC orchestration pipeline.
        
        Performs walk-forward evaluation: trains on first 2/3 of dates,
        then scores remaining dates day-by-day.
        
        Args:
            input_path (str): Path to input CSV file.
        
        Returns:
            str: Path to output file with QC scores.
        
        Raises:
            ValueError: If insufficient data for train/OOS split.
        """
        # 1) Read and sort data
        fullDataSet = self.input_handler.read_input(input_path)
        fullDataSet = fullDataSet.sort_values(main_column.DATE)

        # 2) Split train / OOS by date
        dates = fullDataSet[main_column.DATE].drop_duplicates().sort_values().to_list()
        train_days = int(self.TRAIN_RATIO * len(dates))
        if len(dates) <= train_days:
            raise ValueError("Not enough data: need > TRAIN_DAYS unique dates for walk-forward.")

        cutoff = dates[train_days - 1]
        rawTrainDataSet = fullDataSet.loc[fullDataSet[main_column.DATE] <= cutoff].copy()
        oos_dates = dates[train_days:]

        # 3) Stabilize scale using TRAIN denominators
        denominators = self.compute_per_trade_denominators(rawTrainDataSet)
        trainDataSet = self.apply_denominators(rawTrainDataSet, denominators)

        # 4) Instantiate and fit QC methods on TRAIN
        qc_methods = self._instantiate_qc_methods()
        for method in qc_methods.values():
            method.fit(trainDataSet)

        aggregator = ScoreAggregator(w_if=0.4, w_rz=0.3, w_roll=0.2, w_iqr=0.1)

        # 5) Iterate OOS day-by-day
        results = []
        for d in oos_dates:
            day_raw = fullDataSet.loc[fullDataSet[main_column.DATE] == d].copy()
            day = self.apply_denominators(day_raw, denominators)

            # Score with each method
            scores = {
                qc_methods['ifqc'].ScoreName: qc_methods['ifqc'].score_day(day),
                qc_methods['robustZ'].ScoreName: qc_methods['robustZ'].score_day(day),
                qc_methods['iqr'].ScoreName: qc_methods['iqr'].score_day(day),
                qc_methods['rolling'].ScoreName: qc_methods['rolling'].score_day(day),
            }

            # Aggregate scores
            day_scores = pd.concat(scores.values(), axis=1)
            day_scores.columns = scores.keys()
            aggregatedScore = aggregator.combine(day_scores)
            qc_flag = aggregator.map_to_flag(aggregatedScore).to_frame()

            out = pd.concat([
                day[[main_column.TRADE]].reset_index(drop=True),
                pd.Series([d] * len(day), name=main_column.DATE),
                day_scores.reset_index(drop=True),
                aggregatedScore.reset_index(drop=True),
                qc_flag.reset_index(drop=True),
            ], axis=1)

            results.append(out)

            # Update rolling state AFTER scoring (avoid look-ahead)
            qc_methods['rolling'].update_state(day)

        oos_scores = pd.concat(results, ignore_index=True)

        # 6) Export full dataset with scores
        out_path = self.output_handler.export_full_dataset(
            full_data_set=fullDataSet,
            oos_scores=oos_scores,
            input_path=input_path,
            score_cols=qc_column.SCORE_COLUMNS,
            suffix="_with_scores"
        )
        return out_path

    def _instantiate_qc_methods(self) -> dict:
        """Instantiate all QC scoring methods.
        
        Returns:
            dict: Dictionary of QC method instances keyed by name.
        """
        ifqc = IsolationForestQC(
            base_feats=self.qc_features,
            identity_column=main_column.TRADE,
            temporal_column=main_column.DATE,
            score_name=qc_column.IF_SCORE,
            mode="time_series",
            per_trade_normalize=False,
            use_robust_scaler=True,
            n_estimators=200,
            max_samples=256,
            contamination=0.01,
        )

        robustZ = RobustZQC(
            features=self.qc_features,
            identity_column=main_column.TRADE,
            score_name=qc_column.ROBUST_Z_SCORE,
        )
        iqr = IQRQC(
            features=self.qc_features,
            identity_column=main_column.TRADE,
            score_name=qc_column.IQR_SCORE,
        )
        rolling = RollingZQC(
            features=self.qc_features,
            identity_column=main_column.TRADE,
            temporal_column=main_column.DATE,
            score_name=qc_column.ROLLING_SCORE,
            window=self.roll_window
        )

        return {
            'ifqc': ifqc,
            'robustZ': robustZ,
            'iqr': iqr,
            'rolling': rolling,
        }

if __name__ == "__main__":
    # Interactive entry point
    qc_features = [pnl_column.START, *pnl_column.SLICE_COLUMNS, pnl_column.TOTAL, pnl_column.EXPLAINED, pnl_column.UNEXPLAINED]
    path = input("Enter full path to PnL_Input.csv: ").strip()
    orchestrator = QCOrchestrator(qc_features=qc_features)
    out_path = orchestrator.run(path)
    print(f"\n=== Full export written ===\n{out_path}")


