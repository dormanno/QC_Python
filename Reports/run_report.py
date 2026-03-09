"""
Run QC pipeline with outlier injection and generate ROC evaluation report.

Usage:
    python Reports/run_report.py [report_type] [label]
    
    report_type: pnl, cds, cdi, pv, slices, or all (default: all)
    label: Optional custom label to add to filenames and chart titles
    
Examples:
    python Reports/run_report.py                    # run all reports
    python Reports/run_report.py pnl                # PnL only
    python Reports/run_report.py pnl "v2.0"         # PnL with label "v2.0"
    python Reports/run_report.py all "baseline"    # All reports with label "baseline"
"""

import os
import logging
import glob
import re
from datetime import datetime

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Engine import qc_engine_presets
from Engine.feature_normalizer import FeatureNormalizer
from IO.input import PnLInput, CreditDeltaSingleInput, CreditDeltaIndexInput, PVInput, PnLSlicesInput
from IO.output import Output
from column_names import pnl_column, cds_column, cdi_column, pv_column, pnl_slices_column, main_column, FeatureColumnSet
from QC_Orchestrator import QCOrchestrator
from Tests.outlier_injectors.base import OutlierInjector
from Tests.outlier_injectors.pnl import PnLOutlierInjector
from Tests.outlier_injectors.pnl_config import PnLInjectorConfig
from Tests.outlier_injectors.credit_delta import CreditDeltaOutlierInjector
from Tests.outlier_injectors.credit_delta_config import CreditDeltaInjectorConfig
from Tests.outlier_injectors.pv import PVOutlierInjector
from Tests.outlier_injectors.pv_config import PVInjectorConfig
from Tests.outlier_injectors.pnl_slices import PnLSlicesOutlierInjector
from Tests.outlier_injectors.pnl_slices_config import PnLSlicesInjectorConfig
from Reports.roc_evaluation import evaluate_roc
from Reports.upset_evaluation import evaluate_upset
from Reports.performance_evaluation import evaluate_performance
from Reports.recall_heatmap import evaluate_recall_heatmap
from Engine.qc_engine_presets import QCEnginePreset

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------- configuration ----------
INPUT_DIRECTORY = r"C:\Users\dorma\Documents\UEK_Backup\Test"
INJECTION_SEVERITY = 6  # SEVERITY_MEDIUM
REPORT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "Results")
# ------------------------------------


def _sanitize_token(value: str) -> str:
    """Sanitize value for use as filename token."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', value.strip())


def _get_next_run_number(timestamp: str, dataset_name: str, label: str = None) -> int:
    """Get next run number for the given day/dataset/label combination."""
    safe_dataset = _sanitize_token(dataset_name)
    safe_label = _sanitize_token(label) if label else None

    if safe_label:
        pattern = os.path.join(REPORT_OUTPUT_DIR, f"{timestamp}_*_{safe_label}_{safe_dataset}_*.png")
        run_regex = rf"^{re.escape(timestamp)}_(\d+)_{re.escape(safe_label)}_{re.escape(safe_dataset)}_.*\.png$"
    else:
        pattern = os.path.join(REPORT_OUTPUT_DIR, f"{timestamp}_*_{safe_dataset}_*.png")
        run_regex = rf"^{re.escape(timestamp)}_(\d+)_{re.escape(safe_dataset)}_.*\.png$"

    existing_files = glob.glob(pattern)
    run_numbers = []
    for filepath in existing_files:
        basename = os.path.basename(filepath)
        match = re.match(run_regex, basename)
        if match:
            run_numbers.append(int(match.group(1)))

    return max(run_numbers, default=0) + 1


def _build_report_filename(
    timestamp: str,
    run_number: int,
    dataset_name: str,
    chart_type: str,
    label: str = None,
) -> str:
    """Build report filename in format: date_run_optionalLabel_dataSetName_chartType.png."""
    safe_dataset = _sanitize_token(dataset_name)
    safe_chart = _sanitize_token(chart_type)
    if label:
        safe_label = _sanitize_token(label)
        return f"{timestamp}_{run_number}_{safe_label}_{safe_dataset}_{safe_chart}.png"
    return f"{timestamp}_{run_number}_{safe_dataset}_{safe_chart}.png"


def _run_report(
    input_file: str,
    input_handler,
    column_set: FeatureColumnSet,
    engine_preset: QCEnginePreset,
    injector: OutlierInjector,
    report_title: str,
    dataset_name: str,
    label: str = None,
) -> dict:
    """Generic report pipeline: load -> inject -> score -> ROC + supplementary charts.
    
    Args:
        input_file: Name of input CSV file (in INPUT_DIRECTORY).
        input_handler: Input handler instance (e.g., PnLInput(), CreditDeltaSingleInput()).
        column_set: Column set instance (e.g., pnl_column, cds_column).
        engine_preset: QCEnginePreset instance.
        injector: Pre-configured OutlierInjector subclass instance.
        report_title: Title for the ROC curve plot.
        dataset_name: Dataset identifier used in output filenames.
        label: Optional label to add to filenames and chart titles.
    
    Returns:
        Dict mapping score_name -> {fpr, tpr, thresholds, auc}.
    """
    # Add label to report title if provided
    if label:
        report_title = f"{report_title} [{label}]"
    input_path = os.path.join(INPUT_DIRECTORY, input_file)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # 1. Read & validate
    full_data_set = input_handler.read_and_validate(input_path, split_identifier=main_column.TRADE_TYPE)
    logger.info(f"Loaded {len(full_data_set)} rows from {input_file}")

    # 2. Inject outliers
    full_data_set = injector.inject(full_data_set)

    injected_count = (full_data_set[main_column.RECORD_TYPE] != "OOS").sum() - \
                     (full_data_set[main_column.RECORD_TYPE] == "Train").sum()
    logger.info(f"Injected {injected_count} outlier rows")

    # 3. Normalizer + Orchestrator
    families = engine_preset.qc_feature_families
    has_multiple_families = len(families) > 1
    normalizer = FeatureNormalizer(features=column_set.QC_FEATURES)
    orchestrator = QCOrchestrator(
        normalizer=normalizer,
        engine_preset=engine_preset,
        split_identifier=main_column.TRADE_TYPE,
        keep_family_scores=has_multiple_families,
    )
    oos_scores = orchestrator.run(full_data_set)
    logger.info(f"Scored {len(oos_scores)} OOS rows")

    # 4. Merge scores back to full dataset
    output_handler = Output()
    score_columns = engine_preset.get_score_columns(include_family_scores=has_multiple_families)
    merged = output_handler.attach_scores(full_data_set, oos_scores, score_cols=score_columns)

    # 5. Build naming context (date + run number + optional label)
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    run_number = _get_next_run_number(timestamp=timestamp, dataset_name=dataset_name, label=label)

    # 6. Generate ROC report
    roc_filename = _build_report_filename(
        timestamp=timestamp,
        run_number=run_number,
        dataset_name=dataset_name,
        chart_type="roc_curve",
        label=label,
    )
    roc_png_path = os.path.join(REPORT_OUTPUT_DIR, roc_filename)

    roc_results = evaluate_roc(
        merged_df=merged,
        score_columns=score_columns,
        title=report_title,
        output_path=roc_png_path,
    )

    # 7. Generate UpSet plot of True Positive intersections
    upset_filename = _build_report_filename(
        timestamp=timestamp,
        run_number=run_number,
        dataset_name=dataset_name,
        chart_type="upset_tp",
        label=label,
    )
    upset_png_path = os.path.join(REPORT_OUTPUT_DIR, upset_filename)
    evaluate_upset(
        merged_df=merged,
        score_columns=score_columns,
        threshold=0.95,
        title=report_title.replace("ROC Curves", "True Positive Intersections"),
        output_path=upset_png_path,
    )

    # 8. Generate Performance Comparison chart (Recall, Specificity, Precision, F1)
    perf_filename = _build_report_filename(
        timestamp=timestamp,
        run_number=run_number,
        dataset_name=dataset_name,
        chart_type="performance",
        label=label,
    )
    perf_png_path = os.path.join(REPORT_OUTPUT_DIR, perf_filename)
    evaluate_performance(
        merged_df=merged,
        score_columns=score_columns,
        threshold=0.95,
        title=report_title.replace("ROC Curves", "Performance Comparison"),
        output_path=perf_png_path,
    )

    # 9. Generate Recall Heatmap (per injection type)
    heatmap_filename = _build_report_filename(
        timestamp=timestamp,
        run_number=run_number,
        dataset_name=dataset_name,
        chart_type="recall_heatmap",
        label=label,
    )
    heatmap_png_path = os.path.join(REPORT_OUTPUT_DIR, heatmap_filename)
    evaluate_recall_heatmap(
        merged_df=merged,
        score_columns=score_columns,
        threshold=0.95,
        title=report_title.replace("ROC Curves", "Recall Heatmap"),
        output_path=heatmap_png_path,
    )

    # 10. Per-family reports (ROC + Performance) — only when multiple families exist
    if has_multiple_families:
        for family in families:
            family_score_cols = engine_preset.get_family_score_columns(family)
            family_tag = family.name
            family_dataset_name = f"{dataset_name}_{family_tag}"

            # Build label map: strip family prefix so charts show method names only
            prefix = f"{family_tag}_"
            fam_label_map = {
                col: col[len(prefix):] if col.startswith(prefix) else col
                for col in family_score_cols
            }

            # 9a. Per-family ROC
            fam_roc_filename = _build_report_filename(
                timestamp=timestamp,
                run_number=run_number,
                dataset_name=family_dataset_name,
                chart_type="roc_curve",
                label=label,
            )
            fam_roc_path = os.path.join(REPORT_OUTPUT_DIR, fam_roc_filename)
            fam_roc_title = f"{report_title} — {family_tag}"
            evaluate_roc(
                merged_df=merged,
                score_columns=family_score_cols,
                title=fam_roc_title,
                output_path=fam_roc_path,
                label_map=fam_label_map,
            )
            logger.info(f"Saved per-family ROC: {fam_roc_filename}")

            # 9b. Per-family Performance
            fam_perf_filename = _build_report_filename(
                timestamp=timestamp,
                run_number=run_number,
                dataset_name=family_dataset_name,
                chart_type="performance",
                label=label,
            )
            fam_perf_path = os.path.join(REPORT_OUTPUT_DIR, fam_perf_filename)
            fam_perf_title = report_title.replace("ROC Curves", "Performance Comparison") + f" — {family_tag}"
            evaluate_performance(
                merged_df=merged,
                score_columns=family_score_cols,
                threshold=0.95,
                title=fam_perf_title,
                output_path=fam_perf_path,
                label_map=fam_label_map,
            )
            logger.info(f"Saved per-family Performance: {fam_perf_filename}")

    return roc_results


def run_pnl_report(label: str = None):
    """Run full PnL pipeline: load -> inject -> score -> ROC report.
    
    Args:
        label: Optional label to add to filenames and chart titles.
    """
    logger.info("=" * 80)
    logger.info("Starting PnL Report")
    logger.info("=" * 80)

    config = PnLInjectorConfig.default_preset()
    injector = PnLOutlierInjector(config=config, severity=INJECTION_SEVERITY)

    return _run_report(
        input_file="PnL_Input_Train-OOS.csv",
        input_handler=PnLInput(),
        column_set=pnl_column,
        engine_preset=qc_engine_presets.preset_temporal_multivariate_pnl,
        injector=injector,
        report_title="ROC Curves — PnL",
        dataset_name="pnl",
        label=label,
    )


def run_cds_report(label: str = None):
    """Run full CDS pipeline: load -> inject -> score -> ROC report.
    
    Args:
        label: Optional label to add to filenames and chart titles.
    """
    logger.info("=" * 80)
    logger.info("Starting Credit Delta Single (CDS) Report")
    logger.info("=" * 80)
    
    config = CreditDeltaInjectorConfig.cds_preset()
    injector = CreditDeltaOutlierInjector(config=config, severity=INJECTION_SEVERITY)

    return _run_report(
        input_file="CreditDeltaSingle_Input.csv",
        input_handler=CreditDeltaSingleInput(),
        column_set=cds_column,
        engine_preset=qc_engine_presets.preset_reactive_univariate_cds,
        injector=injector,
        report_title="ROC Curves — Credit Delta Single (CDS)",
        dataset_name="cds",
        label=label,
    )


def run_cdi_report(label: str = None):
    """Run full CDI pipeline: load -> inject -> score -> ROC report.
    
    Args:
        label: Optional label to add to filenames and chart titles.
    """
    logger.info("=" * 80)
    logger.info("Starting Credit Delta Index (CDI) Report")
    logger.info("=" * 80)
    
    config = CreditDeltaInjectorConfig.credit_delta_index_preset()
    injector = CreditDeltaOutlierInjector(config=config, severity=INJECTION_SEVERITY)

    return _run_report(
        input_file="CreditDeltaIndex_Input_Train-OOS.csv",
        input_handler=CreditDeltaIndexInput(),
        column_set=cdi_column,
        engine_preset=qc_engine_presets.preset_robust_univariate_cdi,
        injector=injector,
        report_title="ROC Curves — Credit Delta Index (CDI)",
        dataset_name="cdi",
        label=label,
    )


def run_pv_report(label: str = None):
    """Run full PV pipeline: load -> inject -> score -> ROC report.
    
    Args:
        label: Optional label to add to filenames and chart titles.
    """
    logger.info("=" * 80)
    logger.info("Starting PV (Present Value) Report")
    logger.info("=" * 80)
    
    config = PVInjectorConfig.pv_preset()
    injector = PVOutlierInjector(config=config, severity=INJECTION_SEVERITY)

    return _run_report(
        input_file="PV_Train-OOS.csv",
        input_handler=PVInput(),
        column_set=pv_column,
        engine_preset=qc_engine_presets.preset_reactive_univariate_pv,
        injector=injector,
        report_title="ROC Curves — PV (Present Value)",
        dataset_name="pv",
        label=label,
    )


def run_pnl_slices_report(label: str = None):
    """Run full PnL Slices pipeline: load -> inject -> score -> ROC report.
    
    Args:
        label: Optional label to add to filenames and chart titles.
    """
    logger.info("=" * 80)
    logger.info("Starting PnL Slices Report")
    logger.info("=" * 80)
    
    config = PnLSlicesInjectorConfig.pnl_slices_preset()
    injector = PnLSlicesOutlierInjector(config=config, severity=INJECTION_SEVERITY)

    return _run_report(
        input_file="PnL_Slices_Train-OOS.csv",
        input_handler=PnLSlicesInput(),
        column_set=pnl_slices_column,
        engine_preset=qc_engine_presets.preset_per_family_pnl_slices,
        injector=injector,
        report_title="ROC Curves — PnL Slices",
        dataset_name="pnl_slices",
        label=label,
    )


if __name__ == "__main__":
    import sys
    
    # Default: run all five reports
    reports_to_run = ["pnl", "cds", "cdi", "pv", "slices"]
    label = None
    
    # Check for command-line argument to run specific report
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ["pnl", "profit"]:
            reports_to_run = ["pnl"]
        elif arg in ["cds", "single"]:
            reports_to_run = ["cds"]
        elif arg in ["cdi", "index"]:
            reports_to_run = ["cdi"]
        elif arg in ["pv", "present"]:
            reports_to_run = ["pv"]
        elif arg in ["slices", "pnl_slices"]:
            reports_to_run = ["slices"]
        elif arg in ["both", "all"]:
            reports_to_run = ["pnl", "cds", "cdi", "pv", "slices"]
        else:
            print(f"Usage: python Reports/run_report.py [report_type] [label]")
            print(f"  report_type:")
            print(f"    pnl/profit     - Run PnL report only")
            print(f"    cds/single     - Run Credit Delta Single report only")
            print(f"    cdi/index      - Run Credit Delta Index report only")
            print(f"    pv/present     - Run Present Value report only")
            print(f"    slices/pnl_slices - Run PnL Slices report only")
            print(f"    all            - Run all reports (default)")
            print(f"  label: Optional custom label to add to filenames and chart titles")
            sys.exit(1)
    
    # Check for optional label parameter
    if len(sys.argv) > 2:
        label = sys.argv[2]
        logger.info(f"Using label: '{label}'")
    
    results = {}
    
    if "pnl" in reports_to_run:
        try:
            results["pnl"] = run_pnl_report(label=label)
            logger.info("PnL report completed successfully\n")
        except Exception as e:
            logger.error(f"PnL report failed: {e}", exc_info=True)

    if "cds" in reports_to_run:
        try:
            results["cds"] = run_cds_report(label=label)
            logger.info("CDS report completed successfully\n")
        except Exception as e:
            logger.error(f"CDS report failed: {e}", exc_info=True)
    
    if "cdi" in reports_to_run:
        try:
            results["cdi"] = run_cdi_report(label=label)
            logger.info("CDI report completed successfully\n")
        except Exception as e:
            logger.error(f"CDI report failed: {e}", exc_info=True)
    
    if "pv" in reports_to_run:
        try:
            results["pv"] = run_pv_report(label=label)
            logger.info("PV report completed successfully\n")
        except Exception as e:
            logger.error(f"PV report failed: {e}", exc_info=True)
    
    if "slices" in reports_to_run:
        try:
            results["slices"] = run_pnl_slices_report(label=label)
            logger.info("PnL Slices report completed successfully\n")
        except Exception as e:
            logger.error(f"PnL Slices report failed: {e}", exc_info=True)
    
    logger.info("=" * 80)
    logger.info("All reports completed")
    logger.info("=" * 80)
