"""
UpSet plot evaluation for QC methods.

Produces an UpSet plot showing True Positive intersections across detection
methods, stacked by outlier scenario type (RecordType).  Uses a single
score threshold (default 0.95) to binarise every method score into
detected / not-detected.
"""

import os
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from column_names import main_column, qc_column

# Friendly display names for score columns
_SCORE_DISPLAY_NAMES: Dict[str, str] = {
    qc_column.AGGREGATED_SCORE: "EQAF",
    qc_column.ROBUST_Z_SCORE: "Robust",
    qc_column.ECDF_SCORE: "ECDF",
    qc_column.HAMPEL_SCORE: "Hampel",
    qc_column.ROLLING_SCORE: "Rolling",
    qc_column.IF_SCORE: "IF",
    qc_column.LOF_SCORE: "LOF",
    qc_column.IQR_SCORE: "IQR",
}

# Columns to always exclude from the UpSet plot
_EXCLUDED_COLUMNS = {qc_column.QC_FLAG, qc_column.STALE_SCORE}


def _friendly_name(score_col: str) -> str:
    """Map a raw score column name to a short display name."""
    return _SCORE_DISPLAY_NAMES.get(score_col, score_col.replace("_score", ""))


def build_upset_data(
    merged_df: pd.DataFrame,
    score_columns: List[str],
    threshold: float = 0.95,
) -> pd.DataFrame:
    """Build a DataFrame suitable for upsetplot from scored + labelled data.

    Only rows that are *injected outliers* (RecordType is neither "Train"
    nor "OOS") AND are detected by **at least one** method (score >= threshold)
    are included — i.e. True Positives.

    Args:
        merged_df: Full dataset with scores attached.  Must contain
            ``RecordType`` and all score columns.
        score_columns: Score column names from the engine preset.
        threshold: Score value at or above which a method is considered
            to have flagged the row.

    Returns:
        DataFrame with a boolean MultiIndex (one level per method using
        friendly names) and a ``RecordType`` column for stacking.
    """
    # Keep only injected-outlier rows
    eval_df = merged_df[
        (merged_df[main_column.RECORD_TYPE] != "Train")
        & (merged_df[main_column.RECORD_TYPE] != "OOS")
    ].copy()

    if eval_df.empty:
        raise ValueError("No injected-outlier rows found in the data.")

    # Determine method columns (exclude IsStale and QC_FLAG)
    method_cols = [
        c for c in score_columns if c not in _EXCLUDED_COLUMNS and c in eval_df.columns
    ]
    if not method_cols:
        raise ValueError("No valid method score columns found for UpSet plot.")

    # Build boolean detection matrix with friendly names
    friendly_names = [_friendly_name(c) for c in method_cols]
    detection = pd.DataFrame(
        {
            name: (eval_df[col].values >= threshold)
            for col, name in zip(method_cols, friendly_names)
        },
        index=eval_df.index,
    )

    # Keep only True Positives (detected by at least one method)
    any_detected = detection.any(axis=1)
    detection = detection.loc[any_detected]
    record_types = eval_df.loc[any_detected, main_column.RECORD_TYPE].values

    if detection.empty:
        raise ValueError(
            "No True Positives found — no injected outlier exceeded the "
            f"threshold ({threshold}) for any method."
        )

    # Create MultiIndex from boolean columns
    mi = pd.MultiIndex.from_frame(detection)
    upset_df = pd.DataFrame({"RecordType": record_types}, index=mi)
    return upset_df


def plot_upset_true_positives(
    merged_df: pd.DataFrame,
    score_columns: List[str],
    threshold: float = 0.95,
    title: str = "UpSet Plot of True Positive Intersections Across Detection Methods",
    output_path: Optional[str] = None,
    figsize: tuple = (14, 8),
) -> str:
    """Create and save an UpSet plot of True Positive detections.

    Args:
        merged_df: Full dataset (Train + OOS + injected) with score columns.
        score_columns: List of score column names from engine preset.
        threshold: Score threshold for flagging (default 0.95).
        title: Plot title.
        output_path: Destination PNG path.  Defaults to
            ``Reports/upset_true_positives.png``.
        figsize: Figure size in inches.

    Returns:
        Absolute path to the saved PNG file.
    """
    from upsetplot import UpSet

    upset_df = build_upset_data(merged_df, score_columns, threshold)

    upset = UpSet(
        upset_df,
        show_counts=False,
        sort_by="cardinality",
        show_percentages=False,
        intersection_plot_elements=0,
    )

    # Add stacked bar colouring by RecordType (outlier scenario)
    # This replaces the default black intersection bars with colored ones
    upset.add_stacked_bars(by="RecordType", elements=6)

    fig_obj = plt.figure(figsize=figsize)
    axes_dict = upset.plot(fig=fig_obj)

    # Manually annotate stacked bars with total counts
    stacked_ax_key = [k for k in axes_dict if k not in ("intersections", "matrix", "totals", "shading")]
    if stacked_ax_key:
        ax_stacked = axes_dict[stacked_ax_key[0]]
        # Compute total height per bar position from stacked patches
        bar_totals: Dict[float, float] = {}
        for bar in ax_stacked.patches:
            x = round(bar.get_x(), 6)
            bar_totals[x] = bar_totals.get(x, 0) + bar.get_height()
        for x, total in bar_totals.items():
            if total > 0:
                # Get bar width from first patch
                w = ax_stacked.patches[0].get_width()
                ax_stacked.text(
                    x + w / 2,
                    total,
                    str(int(total)),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # Move legend to the left, outside the stacked bars axes
    if stacked_ax_key:
        legend = ax_stacked.get_legend()
        if legend is not None:
            legend.set_bbox_to_anchor((-0.3, 1))
            legend.set_loc("upper right")

    fig_obj.suptitle(title, fontsize=16, y=1.02)

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), "upset_true_positives.png"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig_obj.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig_obj)

    return os.path.abspath(output_path)


def evaluate_upset(
    merged_df: pd.DataFrame,
    score_columns: List[str],
    threshold: float = 0.95,
    title: str = "UpSet Plot of True Positive Intersections Across Detection Methods",
    output_path: Optional[str] = None,
) -> str:
    """End-to-end UpSet evaluation: build data, plot & save.

    Convenience wrapper matching the signature style of
    ``roc_evaluation.evaluate_roc``.

    Args:
        merged_df: Full dataset with scores attached.
        score_columns: Score column names from engine preset.
        threshold: Detection threshold for all scores.
        title: Plot title.
        output_path: Destination PNG path (optional).

    Returns:
        Absolute path to the saved PNG file.
    """
    saved = plot_upset_true_positives(
        merged_df=merged_df,
        score_columns=score_columns,
        threshold=threshold,
        title=title,
        output_path=output_path,
    )
    print(f"UpSet plot saved to: {saved}")

    # Print summary of True Positive counts per intersection
    upset_df = build_upset_data(merged_df, score_columns, threshold)
    total_injected = (
        (merged_df[main_column.RECORD_TYPE] != "Train")
        & (merged_df[main_column.RECORD_TYPE] != "OOS")
    ).sum()
    total_tp = len(upset_df)
    print(f"\nTrue Positive summary: {total_tp} / {total_injected} "
          f"injected outliers detected ({total_tp / total_injected:.1%})")

    return saved
