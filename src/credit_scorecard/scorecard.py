from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_scorecard.binning import WoEBinner


@dataclass
class ScoreScale:
    base_score: int = 650
    base_odds_good_bad: float = 1.0
    points_to_double_odds: int = 50

    @property
    def factor(self) -> float:
        return self.points_to_double_odds / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds_good_bad)


class CreditScorecard:
    def __init__(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
        max_bins: int = 5,
        scale: ScoreScale | None = None,
    ):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.binner = WoEBinner(numeric_features, categorical_features, max_bins=max_bins)
        self.model = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")
        self.scale = scale or ScoreScale()
        self.feature_names_: list[str] = []

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "CreditScorecard":
        x_woe = self.binner.fit(x, y).transform(x)
        self.feature_names_ = list(x_woe.columns)
        self.model.fit(x_woe, y)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        x_woe = self.binner.transform(x)
        return self.model.predict_proba(x_woe)[:, 1]

    def score(self, x: pd.DataFrame) -> np.ndarray:
        probability_bad = np.clip(self.predict_proba(x), 1e-6, 1 - 1e-6)
        logit_bad = np.log(probability_bad / (1 - probability_bad))
        scores = self.scale.offset - self.scale.factor * logit_bad
        return np.round(scores).astype(int)

    def scorecard_table(self) -> pd.DataFrame:
        summary = self.binner.bin_summary().copy()
        coefficients = dict(zip(self.feature_names_, self.model.coef_[0]))
        summary["woe_feature"] = summary["feature"] + "_woe"
        summary["coefficient"] = summary["woe_feature"].map(coefficients)
        summary["points"] = -self.scale.factor * summary["coefficient"] * summary["woe"]
        summary["points"] = summary["points"].round(0).astype(int)
        summary["iv"] = summary["iv"].round(4)
        summary["bad_rate"] = summary["bad_rate"].round(4)
        summary["woe"] = summary["woe"].round(4)
        return summary[
            [
                "feature",
                "kind",
                "bin",
                "total",
                "good",
                "bad",
                "bad_rate",
                "woe",
                "iv",
                "coefficient",
                "points",
            ]
        ]

    def feature_importance(self) -> pd.DataFrame:
        rows = []
        for feature in self.numeric_features + self.categorical_features:
            feature_rows = self.scorecard_table().query("feature == @feature")
            rows.append(
                {
                    "feature": feature,
                    "information_value": feature_rows["iv"].sum(),
                    "abs_coefficient": abs(float(self.model.coef_[0][self.feature_names_.index(feature + "_woe")])),
                }
            )
        return pd.DataFrame(rows).sort_values("information_value", ascending=False).reset_index(drop=True)
