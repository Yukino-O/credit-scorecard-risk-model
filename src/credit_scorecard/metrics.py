from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def ks_statistic(y_true: pd.Series | np.ndarray, y_score: pd.Series | np.ndarray) -> float:
    frame = pd.DataFrame({"target": y_true, "score": y_score}).sort_values("score", ascending=False)
    total_bad = frame["target"].sum()
    total_good = len(frame) - total_bad
    if total_bad == 0 or total_good == 0:
        return 0.0
    frame["cum_bad"] = frame["target"].cumsum() / total_bad
    frame["cum_good"] = (1 - frame["target"]).cumsum() / total_good
    return float((frame["cum_bad"] - frame["cum_good"]).abs().max())


def score_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    expected_counts = pd.cut(expected, bins=edges, include_lowest=True).value_counts(sort=False)
    actual_counts = pd.cut(actual, bins=edges, include_lowest=True).value_counts(sort=False)
    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()
    eps = 1e-6
    return float(((actual_pct - expected_pct) * np.log((actual_pct + eps) / (expected_pct + eps))).sum())


def classification_metrics(y_true: pd.Series, probability_bad: np.ndarray, score: np.ndarray) -> dict[str, float]:
    auc = roc_auc_score(y_true, probability_bad)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y_true, probability_bad),
        "mean_score": float(np.mean(score)),
        "bad_rate": float(np.mean(y_true)),
    }


def approval_table(y_true: pd.Series, scores: np.ndarray, cutoffs: list[int]) -> pd.DataFrame:
    rows = []
    y = np.asarray(y_true)
    for cutoff in cutoffs:
        approved = scores >= cutoff
        approval_rate = approved.mean()
        default_rate = y[approved].mean() if approved.any() else np.nan
        rows.append(
            {
                "score_cutoff": cutoff,
                "approval_rate": approval_rate,
                "approved_default_rate": default_rate,
                "decline_rate": 1 - approval_rate,
            }
        )
    return pd.DataFrame(rows)
