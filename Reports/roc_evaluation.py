"""
ROC curve evaluation for QC methods.

Computes per-method and ensemble (EQAF) ROC curves using injection-labeled
data as ground truth. RecordType == "OOS" is treated as negative (normal),
any injection label is treated as positive (outlier).
"""

import os
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from column_names import main_column, qc_column


def build_ground_truth(merged_df: pd.DataFrame) -> pd.Series:
    """Derive binary ground truth from RecordType column.

    Args:
        merged_df: DataFrame containing RecordType and score columns.
            Rows with RecordType == "Train" should already be excluded.

    Returns:
        pd.Series of int (0 = normal / OOS, 1 = injected outlier),
        aligned to merged_df index.
    """
    return (merged_df[main_column.RECORD_TYPE] != "OOS").astype(int)


def compute_roc_data(y_true: np.ndarray,
                     scores: np.ndarray) -> Dict:
    """Compute FPR, TPR, thresholds, and AUC for a single score column.

    Args:
        y_true: Binary ground truth (0/1).
        scores: Continuous anomaly scores in [0, 1].

    Returns:
        Dict with keys: fpr, tpr, thresholds, auc.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": roc_auc}


def plot_roc_curves(roc_results: Dict[str, Dict],
                    title: str = "ROC Curves — QC Methods & EQAF",
                    output_path: Optional[str] = None,
                    figsize: tuple = (10, 8)) -> str:
    """Plot ROC curves for multiple scores on one figure and save as PNG.

    Args:
        roc_results: Mapping of score_name -> dict with fpr, tpr, auc
            (as returned by compute_roc_data).
        title: Plot title.
        output_path: Destination PNG path. If None, saves to
            ``Reports/roc_curve.png`` relative to this file.
        figsize: Figure size in inches.

    Returns:
        Absolute path to the saved PNG file.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Color map: EQAF gets a distinct style, methods get tab colors
    method_names = [k for k in roc_results if k != qc_column.AGGREGATED_SCORE]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(method_names), 1)))

    for i, name in enumerate(method_names):
        data = roc_results[name]
        ax.plot(data["fpr"], data["tpr"],
                color=colors[i], lw=3.0,
                label=f"{name} (AUC = {data['auc']:.3f})")

    # Plot EQAF last so it's on top
    if qc_column.AGGREGATED_SCORE in roc_results:
        data = roc_results[qc_column.AGGREGATED_SCORE]
        ax.plot(data["fpr"], data["tpr"],
                color="black", lw=5.0, linestyle="--",
                label=f"{qc_column.AGGREGATED_SCORE} (AUC = {data['auc']:.3f})")

    # Diagonal reference
    ax.plot([0, 1], [0, 1], color="grey", lw=2, linestyle=":")

    ax.set_xlabel("False Positive Rate", fontsize=20)
    ax.set_ylabel("True Positive Rate", fontsize=20)
    ax.set_title(title, fontsize=24)
    ax.legend(loc="lower right", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "roc_curve.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def evaluate_roc(merged_df: pd.DataFrame,
                 score_columns: List[str],
                 title: str = "ROC Curves — QC Methods & EQAF",
                 output_path: Optional[str] = None) -> Dict[str, Dict]:
    """End-to-end ROC evaluation: build ground truth, compute ROC, plot & save.

    Args:
        merged_df: Full dataset with scores attached (output of
            ``Output.attach_scores``). Must contain RecordType column.
            Rows with RecordType == "Train" are excluded automatically.
        score_columns: List of score column names to evaluate
            (e.g. from ``preset.get_score_columns()``). The EQAF_Flag
            column is skipped automatically since it is categorical.
        title: Plot title.
        output_path: Destination PNG path (optional).

    Returns:
        Dict mapping score_name -> {fpr, tpr, thresholds, auc}.
    """
    # Exclude training rows
    eval_df = merged_df[merged_df[main_column.RECORD_TYPE] != "Train"].copy()

    # Drop rows with NaN scores (shouldn't happen for OOS, but be safe)
    numeric_score_cols = [c for c in score_columns if c != qc_column.QC_FLAG]
    eval_df = eval_df.dropna(subset=numeric_score_cols)

    if eval_df.empty:
        raise ValueError("No evaluable rows remain after filtering Train and NaN scores")

    y_true = build_ground_truth(eval_df)

    # Sanity check: need both classes
    if y_true.nunique() < 2:
        present = "positives" if y_true.iloc[0] == 1 else "negatives"
        raise ValueError(
            f"Ground truth contains only {present}. "
            f"Need both OOS (negative) and injected (positive) rows for ROC."
        )

    roc_results = {}
    for col in numeric_score_cols:
        scores = eval_df[col].values
        roc_results[col] = compute_roc_data(y_true.values, scores)

    saved_path = plot_roc_curves(roc_results, title=title, output_path=output_path)
    print(f"ROC curve saved to: {saved_path}")

    # Print AUC summary
    print("\nAUC Summary:")
    for col, data in roc_results.items():
        print(f"  {col:25s}  AUC = {data['auc']:.4f}")

    return roc_results
