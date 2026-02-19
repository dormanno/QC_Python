import logging
import pandas as pd
from typing import List, Tuple, Optional, Dict

from column_names import main_column, qc_column
from Engine.feature_normalizer import FeatureNormalizer
from Engine.qc_engine import QCEngine
from Engine.qc_engine_presets import QCEnginePreset

logger = logging.getLogger(__name__)
# Default configuration
TRAIN_RATIO = 2 / 3  # Use 2/3 of dates for training

class QCOrchestrator:
    """Orchestrates the walk-forward QC evaluation pipeline.
    
    Manages QC method instantiation, training, scoring, and aggregation.
    """
    
    
    
    def __init__(self, 
                 normalizer: FeatureNormalizer,
                 engine_preset: QCEnginePreset,
                 split_identifier: Optional[str] = None):
        """Initialize the orchestrator with normalizer and engine preset.
        
        Args:
            normalizer (FeatureNormalizer): Configured feature normalizer.
            engine_preset (QCEnginePreset): Preset defining QC engine configuration.
                Engines are instantiated on demand from this preset.
            split_identifier (Optional[str]): Column name to split data by before
                training/scoring. If provided, separate QC engines are built and
                fitted per unique value in this column (e.g. TradeType).
        """
        self.normalizer = normalizer
        self.engine_preset = engine_preset
        self.split_identifier = split_identifier

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

    def _split_train_test_by_record_type(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """Split data into train/test sets based on RecordType column.

        Uses rows with RecordType == "Train" for training and all other rows as OOS.

        Args:
            df (pd.DataFrame): Full dataset.

        Returns:
            Tuple containing:
                - Training DataFrame (RecordType == "Train")
                - List of OOS dates (from rows where RecordType != "Train")

        Raises:
            ValueError: If there is no training or OOS data.
        """
        record_col = main_column.RECORD_TYPE
        train_df = df.loc[df[record_col] == "Train"].copy()
        oos_df = df.loc[df[record_col] != "Train"].copy()

        if train_df.empty:
            raise ValueError("No training data found: RecordType == 'Train' returned 0 rows")
        if oos_df.empty:
            raise ValueError("No OOS data found: RecordType != 'Train' returned 0 rows")

        train_dates = set(train_df[main_column.DATE].drop_duplicates())
        oos_df = oos_df.loc[~oos_df[main_column.DATE].isin(train_dates)].copy()

        if oos_df.empty:
            raise ValueError("No OOS data found after removing dates that overlap with Train")

        oos_dates = oos_df[main_column.DATE].drop_duplicates().sort_values().to_list()
        logger.info(
            f"Split data by RecordType: {len(train_df)} train rows, {len(oos_df)} OOS rows, "
            f"{len(oos_dates)} OOS dates"
        )
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

    def _build_and_fit_engines(self, train_data: pd.DataFrame) -> Dict[Optional[str], Dict[str, QCEngine]]:
        """Build and fit QC engines per (split_value, feature_family).
        
        Args:
            train_data (pd.DataFrame): Normalized training data.
        
        Returns:
            Dict mapping split values (or None for unsplit) to dicts of
            family_name -> fitted QCEngine instances.
        """
        families = self.engine_preset.qc_feature_families
        
        if self.split_identifier is None:
            split_values = [None]
            subsets = {None: train_data}
        else:
            split_values = sorted(train_data[self.split_identifier].unique())
            subsets = {sv: train_data[train_data[self.split_identifier] == sv]
                       for sv in split_values}
        
        engines: Dict[Optional[str], Dict[str, QCEngine]] = {}
        for sv in split_values:
            engines[sv] = {}
            for family in families:
                engine = self.engine_preset.build_engine(family)
                try:
                    engine.fit(subsets[sv])
                except ValueError as e:
                    label = f"{self.split_identifier}='{sv}'" if sv is not None else "global"
                    raise ValueError(
                        f"Failed to fit QC engine for {label}, "
                        f"family='{family.name}': {e}"
                    ) from e
                engines[sv][family.name] = engine
                label = f"{self.split_identifier}='{sv}'" if sv is not None else "global"
                logger.info(
                    f"Fitted engine for {label}, family='{family.name}' "
                    f"on {len(subsets[sv])} rows"
                )
        
        total = sum(len(fam_dict) for fam_dict in engines.values())
        logger.info(
            f"Built {total} engines total "
            f"({len(split_values)} split groups \u00d7 {len(families)} families)"
        )
        return engines

    def _score_and_collect(self, family_engines: Dict[str, QCEngine],
                           day_data: pd.DataFrame, date) -> pd.DataFrame:
        """Score a day subset with all family engines and combine via weighted noisy-OR.
        
        For each feature family, scores are computed using the family's dedicated engine.
        Per-method scores are combined across families using weighted average.
        Family-level aggregated scores are combined via weighted noisy-OR:
            combined = 1 - prod((1 - family_agg_i) ^ weight_i)
        
        Args:
            family_engines (Dict[str, QCEngine]): Engines keyed by family name.
            day_data (pd.DataFrame): Normalized data for one day (or day+group subset).
            date: Date value for the current day.
        
        Returns:
            pd.DataFrame: Concatenated trade IDs, date, method scores, aggregated score, and flag.
        """
        families = self.engine_preset.qc_feature_families
        
        family_method_scores = []  # List of (weight, day_scores DataFrame)
        family_agg_scores = []     # List of (weight, aggregated Series)
        
        for family in families:
            engine = family_engines[family.name]
            day_scores, aggregated_score, _ = engine.score_day(day_data)
            family_method_scores.append((family.weight, day_scores))
            family_agg_scores.append((family.weight, aggregated_score))
        
        # Combine per-method scores: weighted average across families
        first_scores = family_method_scores[0][1]
        combined_method_scores = pd.DataFrame(
            0.0, index=first_scores.index, columns=first_scores.columns
        )
        for weight, scores in family_method_scores:
            combined_method_scores += weight * scores

        # Reorder method score columns to respect SCORE_COLUMNS canonical order
        ordered = [c for c in qc_column.SCORE_COLUMNS if c in combined_method_scores.columns]
        remaining = [c for c in combined_method_scores.columns if c not in ordered]
        combined_method_scores = combined_method_scores[ordered + remaining]

        # Combine aggregated scores: weighted noisy-OR = 1 - prod((1 - s_i) ^ w_i)
        survival = pd.Series(1.0, index=day_data.index)
        for weight, agg in family_agg_scores:
            survival *= (1 - agg).clip(lower=0.0) ** weight
        combined_agg = (1 - survival).clip(0.0, 1.0).rename(qc_column.AGGREGATED_SCORE)
        
        # Map to flag using first family engine's aggregator thresholds
        first_engine = family_engines[families[0].name]
        qc_flag = first_engine.aggregator.map_to_flag(combined_agg, simpleMode=True).to_frame()
        
        return pd.concat([
            day_data[[main_column.TRADE]].reset_index(drop=True),
            pd.Series([date] * len(day_data), name=main_column.DATE),
            combined_method_scores.reset_index(drop=True),
            combined_agg.reset_index(drop=True),
            qc_flag.reset_index(drop=True),
        ], axis=1)

    def run(self, full_data_set: pd.DataFrame) -> pd.DataFrame:
        """Run the complete QC orchestration pipeline.
        
        Performs walk-forward evaluation: trains on first 2/3 of dates,
        then scores remaining dates day-by-day. When split_identifier is set,
        separate engines are trained and applied per group.
        
        Args:
            full_data_set (pd.DataFrame): DataFrame with preprocessed, sorted, and
                validated data. Typically obtained from Input.read_and_validate().
        
        Returns:
            pd.DataFrame: Out-of-sample QC scores with columns:
                - TRADE, DATE, method scores, AGGREGATED_SCORE, QC_FLAG
        
        Raises:
            ValueError: If insufficient data for train/OOS split.
        """
        logger.info(f"Starting QC orchestration for {len(full_data_set)} rows")
        
        # Data is already read, sorted, and validated by caller
        logger.info(f"Loaded {len(full_data_set)} rows with {full_data_set[main_column.DATE].nunique()} unique dates")

        # 2) Split train / OOS by RecordType (if present) or by date
        if main_column.RECORD_TYPE in full_data_set.columns:
            raw_train_data, oos_dates = self._split_train_test_by_record_type(full_data_set)
        else:
            raw_train_data, oos_dates = self._split_train_test_by_date(full_data_set)

        # 3) Fit normalizer and prepare training data
        train_data = self._prepare_training_data(raw_train_data)

        # 4) Build and fit QC engine(s) — one per split group, or one global engine
        engines = self._build_and_fit_engines(train_data)

        # 5) Iterate OOS day-by-day
        results = []
        for i, d in enumerate(oos_dates, 1):
            day_raw = full_data_set.loc[full_data_set[main_column.DATE] == d].copy()
            day = self.normalizer.transform(day_raw)

            if self.split_identifier is None:
                # Score entire day with all family engines
                out = self._score_and_collect(engines[None], day, d)
                results.append(out)
                for fe in engines[None].values():
                    fe.update_stateful_methods(day)
            else:
                # Score each split group with its dedicated engine
                group_results = []
                for sv, group_df in day.groupby(self.split_identifier):
                    if sv not in engines:
                        logger.warning(
                            f"Split value '{sv}' not seen in training data, "
                            f"skipping {len(group_df)} rows on date {d}"
                        )
                        continue
                    
                    out = self._score_and_collect(engines[sv], group_df, d)
                    group_results.append(out)
                    for fe in engines[sv].values():
                        fe.update_stateful_methods(group_df)
                
                if group_results:
                    results.append(pd.concat(group_results, ignore_index=True))
            
            if i % 10 == 0 or i == len(oos_dates):
                logger.info(f"Processed {i}/{len(oos_dates)} OOS dates")

        oos_scores = pd.concat(results, ignore_index=True)
        logger.info(f"Generated scores for {len(oos_scores)} OOS rows")
        return oos_scores



