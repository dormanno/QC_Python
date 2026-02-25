"""
Run QC pipeline with outlier injection and generate ROC evaluation report.

Usage:
    python run_report.py
"""

import os
import logging

from Engine import qc_engine_presets
from Engine.feature_normalizer import FeatureNormalizer
from IO.input import CreditDeltaSingleInput, CreditDeltaIndexInput
from IO.output import Output
from column_names import cds_column, cdi_column, main_column, FeatureColumnSet
from qc_orchestrator import QCOrchestrator
from Tests.outlier_injectors.credit_delta import CreditDeltaOutlierInjector
from Tests.outlier_injectors.credit_delta_config import CreditDeltaInjectorConfig
from Reports.roc_evaluation import evaluate_roc
from Reports.upset_evaluation import evaluate_upset
from Reports.performance_evaluation import evaluate_performance
from Engine.qc_engine_presets import QCEnginePreset

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------- configuration ----------
INPUT_DIRECTORY = r"C:\Users\dorma\Documents\UEK_Backup\Test"
INJECTION_SEVERITY = 6  # SEVERITY_MEDIUM
REPORT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "Reports")
# ------------------------------------


def _run_credit_delta_report(
    input_file: str,
    input_handler,
    column_set: FeatureColumnSet,
    engine_preset: QCEnginePreset,
    injector_config: CreditDeltaInjectorConfig,
    report_title: str,
    output_filename: str,
) -> dict:
    """Generic Credit Delta report pipeline: load -> inject -> score -> ROC.
    
    Args:
        input_file: Name of input CSV file (in INPUT_DIRECTORY).
        input_handler: Input handler instance (e.g., CreditDeltaSingleInput()).
        column_set: Column set instance (e.g., cds_column).
        engine_preset: QCEnginePreset instance.
        injector_config: CreditDeltaInjectorConfig instance.
        report_title: Title for the ROC curve plot.
        output_filename: Output PNG filename (saved in REPORT_OUTPUT_DIR).
    
    Returns:
        Dict mapping score_name -> {fpr, tpr, thresholds, auc}.
    """
    input_path = os.path.join(INPUT_DIRECTORY, input_file)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # 1. Read & validate
    full_data_set = input_handler.read_and_validate(input_path, split_identifier=main_column.TRADE_TYPE)
    logger.info(f"Loaded {len(full_data_set)} rows from {input_file}")

    # 2. Inject outliers
    injector = CreditDeltaOutlierInjector(config=injector_config, severity=INJECTION_SEVERITY)
    full_data_set = injector.inject(full_data_set)

    injected_count = (full_data_set[main_column.RECORD_TYPE] != "OOS").sum() - \
                     (full_data_set[main_column.RECORD_TYPE] == "Train").sum()
    logger.info(f"Injected {injected_count} outlier rows")

    # 3. Normalizer + Orchestrator
    normalizer = FeatureNormalizer(features=column_set.QC_FEATURES)
    orchestrator = QCOrchestrator(
        normalizer=normalizer,
        engine_preset=engine_preset,
        split_identifier=main_column.TRADE_TYPE,
    )
    oos_scores = orchestrator.run(full_data_set)
    logger.info(f"Scored {len(oos_scores)} OOS rows")

    # 4. Merge scores back to full dataset
    output_handler = Output()
    score_columns = engine_preset.get_score_columns()
    merged = output_handler.attach_scores(full_data_set, oos_scores, score_cols=score_columns)

    # 5. Generate ROC report
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
    roc_png_path = os.path.join(REPORT_OUTPUT_DIR, output_filename)

    roc_results = evaluate_roc(
        merged_df=merged,
        score_columns=score_columns,
        title=report_title,
        output_path=roc_png_path,
    )

    # 6. Generate UpSet plot of True Positive intersections
    upset_filename = output_filename.replace("roc_curve", "upset_tp")
    upset_png_path = os.path.join(REPORT_OUTPUT_DIR, upset_filename)
    evaluate_upset(
        merged_df=merged,
        score_columns=score_columns,
        threshold=0.95,
        title=report_title.replace("ROC Curves", "True Positive Intersections"),
        output_path=upset_png_path,
    )

    # 7. Generate Performance Comparison chart (Recall, Specificity, Precision, F1)
    perf_filename = output_filename.replace("roc_curve", "performance")
    perf_png_path = os.path.join(REPORT_OUTPUT_DIR, perf_filename)
    evaluate_performance(
        merged_df=merged,
        score_columns=score_columns,
        threshold=0.95,
        title=report_title.replace("ROC Curves", "Performance Comparison"),
        output_path=perf_png_path,
    )

    return roc_results


def run_cds_report():
    """Run full CDS pipeline: load -> inject -> score -> ROC report."""
    logger.info("=" * 80)
    logger.info("Starting Credit Delta Single (CDS) Report")
    logger.info("=" * 80)
    
    return _run_credit_delta_report(
        input_file="CreditDeltaSingle_Input.csv",
        input_handler=CreditDeltaSingleInput(),
        column_set=cds_column,
        engine_preset=qc_engine_presets.preset_reactive_univariate_cds,
        injector_config=CreditDeltaInjectorConfig.cds_preset(),
        report_title="ROC Curves — Credit Delta Single (CDS)",
        output_filename="roc_curve_cds.png",
    )


def run_cdi_report():
    """Run full CDI pipeline: load -> inject -> score -> ROC report."""
    logger.info("=" * 80)
    logger.info("Starting Credit Delta Index (CDI) Report")
    logger.info("=" * 80)
    
    return _run_credit_delta_report(
        input_file="CreditDeltaIndex_Input_Train-OOS.csv",
        input_handler=CreditDeltaIndexInput(),
        column_set=cdi_column,
        engine_preset=qc_engine_presets.preset_robust_univariate_cdi,
        injector_config=CreditDeltaInjectorConfig.credit_delta_index_preset(),
        report_title="ROC Curves — Credit Delta Index (CDI)",
        output_filename="roc_curve_cdi.png",
    )


if __name__ == "__main__":
    import sys
    
    # Default: run both reports
    reports_to_run = ["cds", "cdi"]
    
    # Check for command-line argument to run specific report
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ["cds", "single"]:
            reports_to_run = ["cds"]
        elif arg in ["cdi", "index"]:
            reports_to_run = ["cdi"]
        elif arg in ["both", "all"]:
            reports_to_run = ["cds", "cdi"]
        else:
            print(f"Usage: python run_report.py [cds|cdi|both]")
            print(f"  cds/single - Run Credit Delta Single report only")
            print(f"  cdi/index  - Run Credit Delta Index report only")
            print(f"  both/all   - Run both reports (default)")
            sys.exit(1)
    
    results = {}
    
    if "cds" in reports_to_run:
        try:
            results["cds"] = run_cds_report()
            logger.info("CDS report completed successfully\n")
        except Exception as e:
            logger.error(f"CDS report failed: {e}", exc_info=True)
    
    if "cdi" in reports_to_run:
        try:
            results["cdi"] = run_cdi_report()
            logger.info("CDI report completed successfully\n")
        except Exception as e:
            logger.error(f"CDI report failed: {e}", exc_info=True)
    
    logger.info("=" * 80)
    logger.info("All reports completed")
    logger.info("=" * 80)
