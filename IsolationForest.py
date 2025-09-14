import numpy as np
from typing import Literal, Optional
from dataclasses import dataclass

from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

import pandas as pd
import ColumnNames as COL_NAME

# ----------------------------------------------------
# 3) Matrix builder for Isolation Forest (two modes)
# ----------------------------------------------------

@dataclass
class IFInput:
    X: np.ndarray                 # (n_samples, n_features)
    ids: pd.DataFrame             # columns: TradeName, Date
    feature_names: list           # list of feature names used
    X_pipeline: Pipeline          # fitted preprocessing pipeline (e.g., RobustScaler)


FeatureMode = Literal["time_series", "cross_sectional"]

def build_iforest_matrix(
    df: pd.DataFrame,
    mode: FeatureMode = "time_series",
    per_trade_normalize: bool = True,
    use_robust_scaler: bool = True,
) -> IFInput:
    """
    mode='time_series': compare each trade to its own history (use all dates for each TradeName).
    mode='cross_sectional': compare trades within the same date (use all trades for each Date).
    per_trade_normalize=True: within each TradeName, normalize level features by trade-level median absolute value.
    """
    # Choose features (you can tweak this set)
    base_feats = [
        COL_NAME.START,
        *COL_NAME.PNL_SLICES,
        COL_NAME.TOTAL, "SumSlices", COL_NAME.UNEXPLAINED,
        COL_NAME.TOTAL_JUMP, COL_NAME.UNEXPLAINED_JUMP
    ]

    work = df.copy()

    # Optional per-trade normalization to reduce heteroskedasticity across notionals
    if per_trade_normalize:
        level_feats = [COL_NAME.START, COL_NAME.END, *COL_NAME.PNL_SLICES, COL_NAME.TOTAL, "SumSlices", COL_NAME.UNEXPLAINED]
        # robust scale per trade by median absolute value (L1-like)
        def _norm_group(g: pd.DataFrame) -> pd.DataFrame:
            denom = g[level_feats].abs().median().replace(0, np.nan)
            g[level_feats] = g[level_feats].div(denom, axis=1)
            return g
        work = work.groupby(COL_NAME.TRADE, group_keys=False).apply(_norm_group)

    # Drop rows with all-NaN across chosen features
    feat_df = work[[COL_NAME.TRADE, COL_NAME.DATE] + base_feats].dropna(how="all", subset=base_feats).copy()

    # Depending on mode, we keep all rows but **scoring/evaluation** will subset by trade or by date.
    # For IsolationForest fitting, you will pass the whole X (rolling window) for the chosen scope.

    # Prepare X and ids
    feature_names = base_feats
    X_raw = feat_df[feature_names].astype(float).values
    ids = feat_df[[COL_NAME.TRADE, COL_NAME.DATE]].copy()

    # Build preprocessing pipeline
    steps = []
    if use_robust_scaler:
        steps.append(("robust", RobustScaler(with_centering=True, with_scaling=True)))
    X_pipeline = Pipeline(steps) if steps else None

    if X_pipeline:
        X = X_pipeline.fit_transform(X_raw)
    else:
        X = X_raw

    return IFInput(X=X, ids=ids, feature_names=feature_names, X_pipeline=X_pipeline)


# ---------------------------------------------
# 4) Fitting and scoring Isolation Forest
# ---------------------------------------------

def fit_isolation_forest(
    X: np.ndarray,
    n_estimators: int = 200,
    max_samples: int = 256,
    contamination: Optional[float] = 0.01,
    random_state: int = 42,
) -> IsolationForest:
    """
    Fit an IsolationForest. contamination controls the score->label threshold.
    For pure scoring without labeling, you can set contamination='auto' and threshold later manually.
    """
    clf = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,   # or 'auto'
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X)
    return clf


def score_isolation_forest(
    clf: IsolationForest,
    X: np.ndarray
) -> pd.DataFrame:
    """
    Returns raw anomaly scores in [−0.5, 0.5] via decision_function and also the
    scikit-learn anomaly label (+1 normal, −1 outlier). For interpretability,
    we also compute a monotone [0,1] score (higher=worse) as 1 - normalized_ranks.
    """
    # Sklearn API:
    # - decision_function: higher is more normal, typically in (-0.5, 0.5]
    # - predict: -1 = outlier, +1 = inlier
    decf = clf.decision_function(X)
    yhat = clf.predict(X)

    # Convert to a 0..1 "anomaly_intensity" where 1.0 is worst
    ranks = pd.Series(decf).rank(method="average")  # low decf => more anomalous
    anomaly_intensity = 1.0 - (ranks - 1) / (len(ranks) - 1 + 1e-12)

    return pd.DataFrame({
        "decision_function": decf,
        "sklearn_label": yhat,
        "anomaly_intensity": anomaly_intensity.values,
    })

