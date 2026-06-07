from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_scorecard.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, load_or_generate_dataset
from credit_scorecard.metrics import approval_table, classification_metrics, score_psi
from credit_scorecard.reporting import (
    plot_bad_rate_by_score_band,
    plot_roc,
    plot_score_distribution,
    save_json,
    write_model_report,
)
from credit_scorecard.scorecard import CreditScorecard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the credit scorecard modelling pipeline.")
    parser.add_argument("--rows", type=int, default=5000, help="Number of synthetic applications to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data generation and train/test split.")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"), help="Directory for outputs.")
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Regenerate the synthetic dataset even if the CSV already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.artifact_dir / "data" / "credit_applications.csv"
    data = load_or_generate_dataset(
        data_path,
        n_rows=args.rows,
        random_state=args.seed,
        force_regenerate=args.force_regenerate,
    )

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    train, test = train_test_split(
        data,
        test_size=0.3,
        stratify=data[TARGET],
        random_state=args.seed,
    )

    scorecard = CreditScorecard(NUMERIC_FEATURES, CATEGORICAL_FEATURES, max_bins=5)
    scorecard.fit(train[features], train[TARGET])

    train_probability = scorecard.predict_proba(train[features])
    test_probability = scorecard.predict_proba(test[features])
    train_scores = scorecard.score(train[features])
    test_scores = scorecard.score(test[features])

    train_metrics = classification_metrics(train[TARGET], train_probability, train_scores)
    test_metrics = classification_metrics(test[TARGET], test_probability, test_scores)
    psi = score_psi(pd.Series(train_scores), pd.Series(test_scores))
    policy = approval_table(test[TARGET], test_scores, cutoffs=[520, 560, 600, 640, 680])
    band_table = plot_bad_rate_by_score_band(
        test[TARGET],
        test_scores,
        args.artifact_dir / "figures" / "bad_rate_by_score_band.png",
    )
    plot_score_distribution(train_scores, test_scores, args.artifact_dir / "figures" / "score_distribution.png")
    plot_roc(test[TARGET], test_probability, args.artifact_dir / "figures" / "roc_curve.png")

    scorecard_table = scorecard.scorecard_table()
    feature_importance = scorecard.feature_importance()
    scored_test = test.copy()
    scored_test["probability_default"] = test_probability
    scored_test["credit_score"] = test_scores

    (args.artifact_dir / "reports").mkdir(parents=True, exist_ok=True)
    (args.artifact_dir / "models").mkdir(parents=True, exist_ok=True)
    scorecard_table.to_csv(args.artifact_dir / "reports" / "scorecard_table.csv", index=False)
    feature_importance.to_csv(args.artifact_dir / "reports" / "feature_importance.csv", index=False)
    policy.to_csv(args.artifact_dir / "reports" / "approval_cutoffs.csv", index=False)
    band_table.to_csv(args.artifact_dir / "reports" / "score_band_bad_rates.csv", index=False)
    scored_test.to_csv(args.artifact_dir / "data" / "scored_holdout.csv", index=False)
    joblib.dump(scorecard, args.artifact_dir / "models" / "credit_scorecard.joblib")

    metrics_payload = {
        "train": train_metrics,
        "test": test_metrics,
        "psi": psi,
        "n_rows": int(len(data)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }
    save_json(metrics_payload, args.artifact_dir / "reports" / "metrics.json")
    write_model_report(
        args.artifact_dir / "reports" / "model_report.md",
        train_metrics,
        test_metrics,
        psi,
        policy,
        feature_importance,
    )

    print("Credit scorecard pipeline completed.")
    print(f"Test AUC: {test_metrics['auc']:.3f}")
    print(f"Test KS: {test_metrics['ks']:.3f}")
    print(f"Score PSI: {psi:.4f}")


if __name__ == "__main__":
    main()
