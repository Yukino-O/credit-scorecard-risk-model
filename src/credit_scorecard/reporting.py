from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import RocCurveDisplay


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_score_distribution(train_scores, test_scores, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(train_scores, bins=25, alpha=0.6, label="Train")
    plt.hist(test_scores, bins=25, alpha=0.6, label="Test")
    plt.xlabel("Credit score")
    plt.ylabel("Applications")
    plt.title("Score distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_roc(y_true, probability_bad, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    RocCurveDisplay.from_predictions(y_true, probability_bad)
    plt.title("ROC curve on holdout set")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_bad_rate_by_score_band(y_true, scores, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"target": y_true, "score": scores})
    frame["score_band"] = pd.cut(frame["score"], bins=[300, 500, 550, 600, 650, 700, 750, 850], include_lowest=True)
    band = (
        frame.groupby("score_band", observed=True)
        .agg(applications=("target", "count"), bad_rate=("target", "mean"), mean_score=("score", "mean"))
        .reset_index()
    )
    band["score_band"] = band["score_band"].astype(str)

    plt.figure(figsize=(8, 5))
    plt.bar(band["score_band"], band["bad_rate"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Observed default rate")
    plt.title("Default rate by score band")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return band


def write_model_report(
    path: Path,
    train_metrics: dict,
    test_metrics: dict,
    psi: float,
    approval_table: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Credit Scorecard Model Report",
        "",
        "## Modelling Setup",
        "",
        "This project builds a retail credit risk scorecard with Weight of Evidence (WOE) binning and logistic regression. The data is synthetic and generated from a transparent risk equation, so the repository can be shared publicly without personal data.",
        "",
        "## Holdout Performance",
        "",
        "| Split | AUC | Gini | KS | Bad rate | Mean score |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Train | {train_metrics['auc']:.3f} | {train_metrics['gini']:.3f} | {train_metrics['ks']:.3f} | {train_metrics['bad_rate']:.3f} | {train_metrics['mean_score']:.1f} |",
        f"| Test | {test_metrics['auc']:.3f} | {test_metrics['gini']:.3f} | {test_metrics['ks']:.3f} | {test_metrics['bad_rate']:.3f} | {test_metrics['mean_score']:.1f} |",
        "",
        f"Population Stability Index between train and test scores: **{psi:.4f}**.",
        "",
        "## Policy Cutoff View",
        "",
        approval_table.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Strongest Predictors",
        "",
        feature_importance.head(8).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "Higher scores mean lower estimated default risk. A lender could use the cutoff table to trade off growth and risk appetite. For example, a higher cutoff usually lowers observed default rate among approved applicants but rejects more applications.",
        "",
        "## Limitations",
        "",
        "- The dataset is synthetic, so the project demonstrates modelling workflow rather than production lending performance.",
        "- Fair lending, affordability checks, reject inference, calibration over time, and regulatory sign-off would be required before any real deployment.",
        "- The scorecard is intentionally interpretable; a more advanced project could benchmark gradient boosting and calibrate it against this scorecard.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
