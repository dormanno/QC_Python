import logging
import pandas as pd
from typing import List, Tuple

from IO.input import Input
from IO.output import Output
from column_names import main_column
from Engine.feature_normalizer import FeatureNormalizer
from Engine.qc_engine import QCEngine

logger = logging.getLogger(__name__)
# Default configuration
TRAIN_RATIO = 2 / 3  # Use 2/3 of dates for training

class QCOrchestrator:
    """Orchestrates the walk-forward QC evaluation pipeline.
    
    Manages QC method instantiation, training, scoring, and aggregation.
    """
    
    
    
    def __init__(self, 
                 normalizer: FeatureNormalizer,
                 qc_engine: QCEngine,
                 input_handler: Input):
        """Initialize the orchestrator with normalizer, engine, and input handler.
        
        Args:
            normalizer (FeatureNormalizer): Configured feature normalizer.
            qc_engine (QCEngine): Configured QC engine with methods and aggregator.
            input_handler (Input): Input handler for reading and processing data.
        """
        self.normalizer = normalizer
        self.qc_engine = qc_engine
        self.input_handler = input_handler
        self.output_handler = Output()

    def _split_train_test_by_date(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """Split data into train/test sets based on date cutoff.
        
        Args:
            df (pd.DataFrame): Full dataset sorted by date.
        
        Returns:
            Tuple containing:
                - Training DataFrame (first TRAIN_RATIO of dates)
                - List of OOS dates (remaining dates)
        
        Raises:
            ValueError: If insufficient dates for train/test split.
        """
        dates = df[main_column.DATE].drop_duplicates().sort_values().to_list()
        train_days = int(TRAIN_RATIO * len(dates))
        
        if len(dates) <= train_days:
            raise ValueError(
                f"Insufficient data: {len(dates)} unique dates, need > {train_days} for train/test split"
            )
        
        cutoff = dates[train_days - 1]
        train_df = df.loc[df[main_column.DATE] <= cutoff].copy()
        oos_dates = dates[train_days:]
        
        logger.info(f"Split data: {len(dates)} total dates, {train_days} train, {len(oos_dates)} OOS")
        return train_df, oos_dates
    
    def _prepare_training_data(self, raw_train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit normalizer and normalize training data.
        
        Args:
            raw_train_df (pd.DataFrame): Raw training data.
        
        Returns:
            pd.DataFrame: Normalized training data.
        """
        normalized_train = self.normalizer.fit_transform(raw_train_df)
        logger.info(f"Prepared training data: {len(normalized_train)} rows")
        return normalized_train

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
        logger.info(f"Starting QC orchestration for: {input_path}")
        
        # 1) Read and sort data
        full_data_set = self.input_handler.read_input(input_path)
        full_data_set = full_data_set.sort_values(main_column.DATE)
        logger.info(f"Loaded {len(full_data_set)} rows with {full_data_set[main_column.DATE].nunique()} unique dates")

        # 2) Split train / OOS by date
        raw_train_data, oos_dates = self._split_train_test_by_date(full_data_set)

        # 3) Fit normalizer and prepare training data
        train_data = self._prepare_training_data(raw_train_data)

        # 4) Fit QC engine on TRAIN
        self.qc_engine.fit(train_data)

        # 5) Iterate OOS day-by-day
        results = []
        for i, d in enumerate(oos_dates, 1):
            day_raw = full_data_set.loc[full_data_set[main_column.DATE] == d].copy()
            day = self.normalizer.transform(day_raw)

            # Score the day using engine
            day_scores, aggregated_score, qc_flag = self.qc_engine.score_day(day)

            out = pd.concat([
                day[[main_column.TRADE]].reset_index(drop=True),
                pd.Series([d] * len(day), name=main_column.DATE),
                day_scores.reset_index(drop=True),
                aggregated_score.reset_index(drop=True),
                qc_flag.reset_index(drop=True),
            ], axis=1)

            results.append(out)

            # Update stateful methods AFTER scoring (avoid look-ahead)
            self.qc_engine.update_stateful_methods(day)
            
            if i % 10 == 0 or i == len(oos_dates):
                logger.info(f"Processed {i}/{len(oos_dates)} OOS dates")

        oos_scores = pd.concat(results, ignore_index=True)
        logger.info(f"Generated scores for {len(oos_scores)} OOS rows")

        # 6) Export full dataset with scores
        # Get actual score columns from the engine's configured methods
        actual_score_cols = self.qc_engine.get_score_columns()
        
        out_path = self.output_handler.export_full_dataset(
            full_data_set=full_data_set,
            oos_scores=oos_scores,
            input_path=input_path,
            score_cols=actual_score_cols,
            suffix="_with_scores"
        )        
        logger.info(f"Output written to: {out_path}")
        return out_path



