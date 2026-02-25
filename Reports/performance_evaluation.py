"""
Performance metrics evaluation for QC methods.

Computes per-method Recall, Specificity, Precision, and F1 using
injection-labeled data as ground truth and a fixed score threshold.
Produces a single 2×2 figure with stacked bars for the first three
metrics and a plain bar chart for F1.
"""

import os
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from column_names import main_column, qc_column


# Friendly display names for score columns (reuse from upset_evaluation)
_SCORE_DISPLAY_NAMES: Dict[str, str] = {
    qc_column.AGGREGATED_SCORE: "EQAF",
    qc_column.ROBUST_Z_SCORE: "Robust Z",
    qc_column.ECDF_SCORE: "ECDF",
    qc_column.HAMPEL_SCORE: "Hampel",
    qc_column.ROLLING_SCORE: "Rolling Z",
    qc_column.IF_SCORE: "Isolation\nForest",
    qc_column.LOF_SCORE: "LOF",
    qc_column.IQR_SCORE: "IQR",
}

# Columns to exclude from the evaluation
_EXCLUDED_COLUMNS = {qc_column.QC_FLAG, qc_column.STALE_SCORE}


def _friendly_name(score_col: str) -> str:
    """Map a raw score column name to a short display name."""
    return _SCORE_DISPLAY_NAMES.get(score_col, score_col.replace("_score", ""))


def compute_confusion_matrix(
    merged_df: pd.DataFrame,
    score_columns: List[str],
    threshold: float = 0.95,
) -> Dict[str, Dict[str, int]]:
    """Compute TP, FP, TN, FN per method score column.

    Ground truth:
        - Positive (outlier): RecordType is neither "Train" nor "OOS"
        - Negative (normal): RecordType == "OOS"
        Training rows are excluded entirely.

    Args:
        merged_df: Full dataset with scores and RecordType column.
        score_columns: Score column names from engine preset.
        threshold: Score >= threshold is classified as outlier.

    Returns:
        Dict mapping friendly method name -> {"TP": int, "FP": int,
        "TN": int, "FN": int}.
    """
    # Exclude training rows
    eval_df = merged_df[merged_df[main_column.RECORD_TYPE] != "Train"].copy()

    # Build ground truth: 1 = injected outlier, 0 = normal (OOS)
    y_true = (eval_df[main_column.RECORD_TYPE] != "OOS").astype(int).values

    method_cols = [
        c for c in score_columns
        if c not in _EXCLUDED_COLUMNS and c in eval_df.columns
    ]

    results: Dict[str, Dict[str, int]] = {}
    for col in method_cols:
        scores = eval_df[col].values
        predicted = (scores >= threshold).astype(int)

        tp = int(((predicted == 1) & (y_true == 1)).sum())
        fp = int(((predicted == 1) & (y_true == 0)).sum())
        tn = int(((predicted == 0) & (y_true == 0)).sum())
        fn = int(((predicted == 0) & (y_true == 1)).sum())

        results[_friendly_name(col)] = {"TP": tp, "FP": fp, "TN": tn, "FN": fn}

    return results


def compute_performance_metrics(
    confusion: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, float]]:
    """Derive Recall, Specificity, Precision, and F1 from confusion counts.

    Args:
        confusion: Output of ``compute_confusion_matrix``.

    Returns:
        Dict mapping method name -> {"recall", "specificity", "precision", "f1"}.
    """
    metrics: Dict[str, Dict[str, float]] = {}
    for method, cm in confusion.items():
        tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        metrics[method] = {
            "recall": recall,
            "specificity": specificity,
            "precision": precision,
            "f1": f1,
        }
    return metrics


def plot_performance(
    confusion: Dict[str, Dict[str, int]],
    metrics: Dict[str, Dict[str, float]],
    title: str = "Performance Comparison of Detection Methods",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> str:
    """Create a 2×2 figure with Recall, Specificity, Precision, and F1.

    Top-left:     Recall     — stacked bars (FN + TP)
    Top-right:    Specificity — stacked bars (FP + TN)
    Bottom-left:  Precision  — stacked bars (FP + TP)
    Bottom-right: F1         — plain bars (percentage)

    Args:
        confusion: Per-method confusion counts.
        metrics: Per-method derived metrics.
        title: Super-title for the figure.
        output_path: Destination PNG path.
        figsize: Figure size in inches.

    Returns:
        Absolute path to the saved PNG.
    """
    # Ensure EQAF is plotted last (rightmost) if present
    methods = [m for m in confusion if m != "EQAF"]
    if "EQAF" in confusion:
        methods.append("EQAF")

    x = np.arange(len(methods))
    bar_width = 0.55

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=40, fontweight="bold", y=0.98)

    # Color palette
    color_tp = "#4CAF50"   # green
    color_fn = "#9C27B0"   # purple
    color_fp = "#FF9800"   # orange
    color_tn = "#00838F"   # teal
    color_f1 = "#9E9E9E"   # grey

    # ── Recall (top-left): stacked FN + TP ──
    ax = axes[0, 0]
    tp_vals = [confusion[m]["TP"] for m in methods]
    fn_vals = [confusion[m]["FN"] for m in methods]

    bars_fn = ax.bar(x, fn_vals, bar_width, label="FN", color=color_fn)
    bars_tp = ax.bar(x, tp_vals, bar_width, bottom=fn_vals, label="TP", color=color_tp)

    # Annotate counts
    for i, (fn_v, tp_v) in enumerate(zip(fn_vals, tp_vals)):
        if fn_v > 0:
            ax.text(x[i], fn_v / 2, str(fn_v), ha="center", va="center",
                    fontsize=18, fontweight="bold", color="white")
        if tp_v > 0:
            ax.text(x[i], fn_v + tp_v / 2, str(tp_v), ha="center", va="center",
                    fontsize=18, fontweight="bold", color="white")

    ax.set_title("Recall", fontsize=32, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=20)
    ax.legend(fontsize=20, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # ── Specificity (top-right): stacked FP + TN ──
    ax = axes[0, 1]
    fp_vals = [confusion[m]["FP"] for m in methods]
    tn_vals = [confusion[m]["TN"] for m in methods]

    bars_fp = ax.bar(x, fp_vals, bar_width, label="FP", color=color_fp)
    bars_tn = ax.bar(x, tn_vals, bar_width, bottom=fp_vals, label="TN", color=color_tn)

    for i, (fp_v, tn_v) in enumerate(zip(fp_vals, tn_vals)):
        if fp_v > 0:
            ax.text(x[i], fp_v / 2, str(fp_v), ha="center", va="center",
                    fontsize=18, fontweight="bold", color="white")
        if tn_v > 0:
            ax.text(x[i], fp_v + tn_v / 2, str(tn_v), ha="center", va="center",
                    fontsize=18, fontweight="bold", color="white")

    ax.set_title("Specificity", fontsize=32, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=20)
    ax.legend(fontsize=20, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # ── Precision (bottom-left): stacked FP + TP ──
    ax = axes[1, 0]
    bars_fp = ax.bar(x, fp_vals, bar_width, label="FP", color=color_fp)
    bars_tp = ax.bar(x, tp_vals, bar_width, bottom=fp_vals, label="TP", color=color_tp)

    for i, (fp_v, tp_v) in enumerate(zip(fp_vals, tp_vals)):
        if fp_v > 0:
            ax.text(x[i], fp_v / 2, str(fp_v), ha="center", va="center",
                    fontsize=18, fontweight="bold", color="white")
        if tp_v > 0:
            ax.text(x[i], fp_v + tp_v / 2, str(tp_v), ha="center", va="center",
                    fontsize=18, fontweight="bold", color="white")

    ax.set_title("Precision", fontsize=32, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=20)
    ax.legend(fontsize=20, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # ── F1 (bottom-right): plain bars with percentage labels ──
    ax = axes[1, 1]
    f1_vals = [metrics[m]["f1"] * 100 for m in methods]

    bars_f1 = ax.bar(x, f1_vals, bar_width, color=color_f1)

    for i, f1_v in enumerate(f1_vals):
        ax.text(x[i], f1_v + 1, f"{f1_v:.0f}%", ha="center", va="bottom",
                fontsize=22, fontweight="bold")

    ax.set_title("F1", fontsize=32, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=20)
    ax.set_ylim(0, 105)
    ax.set_ylabel("%", fontsize=24)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "performance_comparison.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return os.path.abspath(output_path)


def evaluate_performance(
    merged_df: pd.DataFrame,
    score_columns: List[str],
    threshold: float = 0.95,
    title: str = "Performance Comparison of Detection Methods",
    output_path: Optional[str] = None,
) -> str:
    """End-to-end performance evaluation: confusion matrix → metrics → plot.

    Convenience wrapper matching the signature style of
    ``roc_evaluation.evaluate_roc`` and ``upset_evaluation.evaluate_upset``.

    Args:
        merged_df: Full dataset with scores attached.
        score_columns: Score column names from engine preset.
        threshold: Detection threshold for all scores (default 0.95).
        title: Plot super-title.
        output_path: Destination PNG path (optional).

    Returns:
        Absolute path to the saved PNG file.
    """
    confusion = compute_confusion_matrix(merged_df, score_columns, threshold)
    perf_metrics = compute_performance_metrics(confusion)

    saved = plot_performance(
        confusion=confusion,
        metrics=perf_metrics,
        title=title,
        output_path=output_path,
    )
    print(f"Performance comparison chart saved to: {saved}")

    # Print summary table
    print(f"\nPerformance Summary (threshold = {threshold}):")
    print(f"  {'Method':<20s} {'Recall':>8s} {'Specif.':>8s} {'Precis.':>8s} {'F1':>8s}"
          f"  {'TP':>5s} {'FP':>5s} {'TN':>5s} {'FN':>5s}")
    print("  " + "-" * 80)
    for method in confusion:
        cm = confusion[method]
        m = perf_metrics[method]
        # Replace newlines in method name for table readability
        display = method.replace("\n", " ")
        print(f"  {display:<20s} {m['recall']:>7.1%} {m['specificity']:>7.1%} "
              f"{m['precision']:>7.1%} {m['f1']:>7.1%}"
              f"  {cm['TP']:>5d} {cm['FP']:>5d} {cm['TN']:>5d} {cm['FN']:>5d}")

    return saved
