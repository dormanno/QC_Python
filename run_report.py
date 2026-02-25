"""
Run QC pipeline with outlier injection and generate ROC evaluation report.

Usage:
    python run_report.py
"""

import os
import logging

from Engine import qc_engine_presets
from Engine.feature_normalizer import FeatureNormalizer
from IO.input import CreditDeltaSingleInput
from IO.output import Output
from column_names import cds_column, main_column
from qc_orchestrator import QCOrchestrator
from Tests.outlier_injectors.credit_delta import CreditDeltaOutlierInjector
from Tests.outlier_injectors.credit_delta_config import CreditDeltaInjectorConfig
from Reports.roc_evaluation import evaluate_roc

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------- configuration ----------
INPUT_DIRECTORY = r"C:\Users\dorma\Documents\UEK_Backup\Test"
INPUT_FILE = "CreditDeltaSingle_Input.csv"
INJECTION_SEVERITY = 6  # SEVERITY_MEDIUM
REPORT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "Reports")
# ------------------------------------


def run_cds_report():
    """Run full CDS pipeline: load -> inject -> score -> ROC report."""
    input_path = os.path.join(INPUT_DIRECTORY, INPUT_FILE)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # 1. Read & validate
    input_handler = CreditDeltaSingleInput()
    full_data_set = input_handler.read_and_validate(input_path, split_identifier=main_column.TRADE_TYPE)
    logger.info(f"Loaded {len(full_data_set)} rows from {INPUT_FILE}")

    # 2. Inject outliers
    config = CreditDeltaInjectorConfig.cds_preset()
    injector = CreditDeltaOutlierInjector(config=config, severity=INJECTION_SEVERITY)
    full_data_set = injector.inject(full_data_set)

    injected_count = (full_data_set[main_column.RECORD_TYPE] != "OOS").sum() - \
                     (full_data_set[main_column.RECORD_TYPE] == "Train").sum()
    logger.info(f"Injected {injected_count} outlier rows")

    # 3. Normalizer + Orchestrator
    preset = qc_engine_presets.preset_reactive_univariate_cds
    normalizer = FeatureNormalizer(features=cds_column.QC_FEATURES)
    orchestrator = QCOrchestrator(
        normalizer=normalizer,
        engine_preset=preset,
        split_identifier=main_column.TRADE_TYPE,
    )
    oos_scores = orchestrator.run(full_data_set)
    logger.info(f"Scored {len(oos_scores)} OOS rows")

    # 4. Merge scores back to full dataset
    output_handler = Output()
    score_columns = preset.get_score_columns()
    merged = output_handler.attach_scores(full_data_set, oos_scores, score_cols=score_columns)

    # 5. Generate ROC report
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
    roc_png_path = os.path.join(REPORT_OUTPUT_DIR, "roc_curve_cds.png")

    roc_results = evaluate_roc(
        merged_df=merged,
        score_columns=score_columns,
        title="ROC Curves — Credit Delta Single (CDS)",
        output_path=roc_png_path,
    )

    return roc_results


if __name__ == "__main__":
    run_cds_report()
