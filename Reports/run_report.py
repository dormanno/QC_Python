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
from column_names import pnl_column, cds_column, cdi_column, pv_column, pnl_slices_column, main_column, qc_column, FeatureColumnSet
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


def _get_next_run_number(timestamp: str, label: str = None) -> int:
    """Get next run number by scanning existing run subfolders under REPORT_OUTPUT_DIR."""
    safe_label = _sanitize_token(label) if label else None
    if safe_label:
        run_regex = rf"^{re.escape(timestamp)}_(\d+)_{re.escape(safe_label)}$"
    else:
        run_regex = rf"^{re.escape(timestamp)}_(\d+)$"

    run_numbers = []
    if os.path.isdir(REPORT_OUTPUT_DIR):
        for name in os.listdir(REPORT_OUTPUT_DIR):
            if os.path.isdir(os.path.join(REPORT_OUTPUT_DIR, name)):
                match = re.match(run_regex, name)
                if match:
                    run_numbers.append(int(match.group(1)))

    return max(run_numbers, default=0) + 1


def _build_run_folder_name(timestamp: str, run_number: int, label: str = None) -> str:
    """Build run output subfolder name: date_runNumber[_label]."""
    if label:
        safe_label = _sanitize_token(label)
        return f"{timestamp}_{run_number}_{safe_label}"
    return f"{timestamp}_{run_number}"


def _build_report_filename(dataset_name: str, chart_type: str) -> str:
    """Build report filename: dataSetName_chartType.png.

    Date, run number and label are encoded in the parent run folder name.
    """
    safe_dataset = _sanitize_token(dataset_name)
    safe_chart = _sanitize_token(chart_type)
    return f"{safe_dataset}_{safe_chart}.png"


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

    # Final report score selection:
    # - Multi-family: family aggregate scores + final EQAF only
    # - Single-family: method scores + final EQAF
    if has_multiple_families:
        report_score_columns = [f"{family.name}_AggScore" for family in families]
        report_score_columns.append(qc_column.AGGREGATED_SCORE)
    else:
        report_score_columns = engine_preset.get_score_columns(include_family_scores=False)
        report_score_columns = [c for c in report_score_columns if c != qc_column.QC_FLAG]

    # 5. Build naming context — create a dedicated run subfolder
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    run_number = _get_next_run_number(timestamp=timestamp, label=label)
    run_folder = _build_run_folder_name(timestamp, run_number, label)
    run_output_dir = os.path.join(REPORT_OUTPUT_DIR, run_folder)
    os.makedirs(run_output_dir, exist_ok=True)
    logger.info(f"Run output folder: {run_folder}")

    # 6. Generate ROC report
    roc_results = evaluate_roc(
        merged_df=merged,
        score_columns=report_score_columns,
        title=report_title,
        output_path=os.path.join(run_output_dir, _build_report_filename(dataset_name, "roc_curve")),
    )

    # 7. Generate UpSet plot of True Positive intersections
    evaluate_upset(
        merged_df=merged,
        score_columns=report_score_columns,
        threshold=0.95,
        title=report_title.replace("ROC Curves", "True Positive Intersections"),
        output_path=os.path.join(run_output_dir, _build_report_filename(dataset_name, "upset_tp")),
    )

    # 8. Generate Performance Comparison chart (Recall, Specificity, Precision, F1)
    evaluate_performance(
        merged_df=merged,
        score_columns=report_score_columns,
        threshold=0.95,
        title=report_title.replace("ROC Curves", "Performance Comparison"),
        output_path=os.path.join(run_output_dir, _build_report_filename(dataset_name, "performance")),
    )

    # 9. Generate Recall Heatmap (per injection type)
    evaluate_recall_heatmap(
        merged_df=merged,
        score_columns=report_score_columns,
        threshold=0.95,
        title=report_title.replace("ROC Curves", "Recall Heatmap"),
        output_path=os.path.join(run_output_dir, _build_report_filename(dataset_name, "recall_heatmap")),
    )

    # 10. Method-level charts for multi-family datasets (aggregated method scores + EQAF)
    if has_multiple_families:
        methods_score_columns = engine_preset.get_score_columns(include_family_scores=False)
        methods_score_columns = [c for c in methods_score_columns if c != qc_column.QC_FLAG]

        evaluate_roc(
            merged_df=merged,
            score_columns=methods_score_columns,
            title=report_title + " — Methods",
            output_path=os.path.join(run_output_dir, _build_report_filename(dataset_name, "roc_curve_methods")),
        )
        evaluate_upset(
            merged_df=merged,
            score_columns=methods_score_columns,
            threshold=0.95,
            title=report_title.replace("ROC Curves", "True Positive Intersections") + " — Methods",
            output_path=os.path.join(run_output_dir, _build_report_filename(dataset_name, "upset_tp_methods")),
        )
        evaluate_performance(
            merged_df=merged,
            score_columns=methods_score_columns,
            threshold=0.95,
            title=report_title.replace("ROC Curves", "Performance Comparison") + " — Methods",
            output_path=os.path.join(run_output_dir, _build_report_filename(dataset_name, "performance_methods")),
        )
        evaluate_recall_heatmap(
            merged_df=merged,
            score_columns=methods_score_columns,
            threshold=0.95,
            title=report_title.replace("ROC Curves", "Recall Heatmap") + " — Methods",
            output_path=os.path.join(run_output_dir, _build_report_filename(dataset_name, "recall_heatmap_methods")),
        )
        logger.info("Saved method-level charts for multi-family dataset")

    # 11. Per-family reports (ROC + Performance) — only when multiple families exist
    if has_multiple_families:
        for family in families:
            family_score_cols = engine_preset.get_family_score_columns(family)
            family_tag = family.name
            family_dataset_name = f"{dataset_name}_{family_tag}"

            # Build label map: strip family prefix so charts show method names only
            prefix = f"{family_tag}_"
            fam_label_map = {
                col: (col[len(prefix):] if col.startswith(prefix) else col).replace("_score", "")
                for col in family_score_cols
            }

            # 9a. Per-family ROC
            fam_roc_path = os.path.join(run_output_dir, _build_report_filename(family_dataset_name, "roc_curve"))
            evaluate_roc(
                merged_df=merged,
                score_columns=family_score_cols,
                title=f"{report_title} — {family_tag}",
                output_path=fam_roc_path,
                label_map=fam_label_map,
            )
            logger.info(f"Saved per-family ROC: {os.path.basename(fam_roc_path)}")

            # 9b. Per-family Performance
            fam_perf_path = os.path.join(run_output_dir, _build_report_filename(family_dataset_name, "performance"))
            evaluate_performance(
                merged_df=merged,
                score_columns=family_score_cols,
                threshold=0.95,
                title=report_title.replace("ROC Curves", "Performance Comparison") + f" — {family_tag}",
                output_path=fam_perf_path,
                label_map=fam_label_map,
            )
            logger.info(f"Saved per-family Performance: {os.path.basename(fam_perf_path)}")

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
