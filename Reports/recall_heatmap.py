"""
Recall heatmap evaluation for QC methods.

Computes per-method, per-injection-type Recall using injection-labeled data
as ground truth and a fixed score threshold.  Produces a colour-coded
heatmap where rows are detection methods (+ EQAF) and columns are injection
scenario types (e.g. ScaleError, Drift, PointShock, …).
"""

import os
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from column_names import main_column, qc_column


# Friendly display names for score columns (reuse convention)
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


def compute_recall_by_injection_type(
    merged_df: pd.DataFrame,
    score_columns: List[str],
    threshold: float = 0.95,
) -> pd.DataFrame:
    """Compute Recall per method per injection scenario type.

    Ground truth is derived from ``RecordType``:
        - "Train" rows are excluded.
        - "OOS" rows are normal (negative).
        - All other values are injection type labels (positive).

    Args:
        merged_df: Full dataset with scores and RecordType column.
        score_columns: Score column names from engine preset.
        threshold: Score >= threshold is classified as outlier.

    Returns:
        DataFrame with methods as rows and injection types as columns,
        values are Recall (0–1).
    """
    # Exclude training rows
    eval_df = merged_df[merged_df[main_column.RECORD_TYPE] != "Train"].copy()

    # Identify injection types (everything that isn't OOS)
    injection_types = sorted(
        eval_df.loc[eval_df[main_column.RECORD_TYPE] != "OOS", main_column.RECORD_TYPE]
        .unique()
        .tolist()
    )

    # Determine method columns
    method_cols = [
        c for c in score_columns
        if c not in _EXCLUDED_COLUMNS and c in eval_df.columns
    ]

    # Compute Recall for each (method, injection_type) pair
    records = {}
    for col in method_cols:
        name = _friendly_name(col)
        row: Dict[str, float] = {}
        for inj_type in injection_types:
            mask = eval_df[main_column.RECORD_TYPE] == inj_type
            subset = eval_df.loc[mask]
            if len(subset) == 0:
                row[inj_type] = np.nan
                continue
            predicted = (subset[col].values >= threshold).astype(int)
            tp = int(predicted.sum())
            total = len(subset)
            row[inj_type] = tp / total
        records[name] = row

    recall_df = pd.DataFrame.from_dict(records, orient="index")
    recall_df.columns.name = "Injection Type"
    recall_df.index.name = "Method"

    # Ensure EQAF is last row if present
    if "EQAF" in recall_df.index:
        order = [m for m in recall_df.index if m != "EQAF"] + ["EQAF"]
        recall_df = recall_df.loc[order]

    return recall_df


def plot_recall_heatmap(
    recall_df: pd.DataFrame,
    title: str = "Recall Heatmap",
    output_path: Optional[str] = None,
    figsize: tuple = (14, 6),
) -> str:
    """Plot a colour-coded Recall heatmap and save as PNG.

    Cells are coloured on a continuous red (0%) → green (100%) gradient.
    Each cell is annotated with the percentage value.

    Args:
        recall_df: DataFrame from ``compute_recall_by_injection_type``.
        title: Plot title.
        output_path: Destination PNG path.  If None, saves to
            ``Reports/recall_heatmap.png`` relative to this file.
        figsize: Figure size in inches.

    Returns:
        Absolute path to the saved PNG file.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Build a red-to-green colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "red_green", ["#d32f2f", "#ff9800", "#4CAF50"], N=256
    )

    data = recall_df.values.astype(float)

    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Axis labels
    ax.set_xticks(np.arange(len(recall_df.columns)))
    ax.set_xticklabels(recall_df.columns, fontsize=16, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(recall_df.index)))
    ax.set_yticklabels(recall_df.index, fontsize=16)

    # Annotate each cell with percentage
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                text = "N/A"
                color = "grey"
            else:
                text = f"{val:.0%}"
                # Use white text for readability on dark backgrounds
                color = "white" if val < 0.75 else "white"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=16, fontweight="bold", color=color)

    ax.set_title(title, fontsize=24, fontweight="bold", pad=12)
    fig.tight_layout()

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "recall_heatmap.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return os.path.abspath(output_path)


def evaluate_recall_heatmap(
    merged_df: pd.DataFrame,
    score_columns: List[str],
    threshold: float = 0.95,
    title: str = "Recall Heatmap",
    output_path: Optional[str] = None,
) -> str:
    """End-to-end recall heatmap evaluation: compute recall → plot heatmap.

    Convenience wrapper matching the signature style of the other report
    evaluation functions.

    Args:
        merged_df: Full dataset with scores attached.
        score_columns: Score column names from engine preset.
        threshold: Detection threshold for all scores (default 0.95).
        title: Plot title.
        output_path: Destination PNG path (optional).

    Returns:
        Absolute path to the saved PNG file.
    """
    recall_df = compute_recall_by_injection_type(
        merged_df, score_columns, threshold
    )

    saved = plot_recall_heatmap(
        recall_df=recall_df,
        title=title,
        output_path=output_path,
    )
    print(f"Recall heatmap saved to: {saved}")

    # Print summary table
    print(f"\nRecall by Injection Type (threshold = {threshold}):")
    header_types = recall_df.columns.tolist()
    header = f"  {'Method':<20s}" + "".join(f" {t:>14s}" for t in header_types)
    print(header)
    print("  " + "-" * (20 + 15 * len(header_types)))
    for method in recall_df.index:
        display = method.replace("\n", " ")
        vals = "".join(
            f" {recall_df.loc[method, t]:>13.1%}" if not np.isnan(recall_df.loc[method, t]) else f" {'N/A':>13s}"
            for t in header_types
        )
        print(f"  {display:<20s}{vals}")

    return saved
