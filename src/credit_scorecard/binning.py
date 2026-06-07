from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BinRule:
    feature: str
    kind: str
    edges: list[float] | None
    table: pd.DataFrame
    default_woe: float = 0.0


class WoEBinner:
    """Fit Weight of Evidence transformations without using test data."""

    def __init__(self, numeric_features: list[str], categorical_features: list[str], max_bins: int = 5):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.max_bins = max_bins
        self.rules_: dict[str, BinRule] = {}

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "WoEBinner":
        target = pd.Series(y).astype(int).reset_index(drop=True)
        frame = x.reset_index(drop=True).copy()
        for feature in self.numeric_features:
            self.rules_[feature] = self._fit_numeric(feature, frame[feature], target)
        for feature in self.categorical_features:
            self.rules_[feature] = self._fit_categorical(feature, frame[feature], target)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        transformed = pd.DataFrame(index=x.index)
        for feature, rule in self.rules_.items():
            if rule.kind == "numeric":
                bins = pd.cut(
                    x[feature],
                    bins=rule.edges,
                    include_lowest=True,
                    duplicates="drop",
                )
                mapping = rule.table.set_index("bin")["woe"].to_dict()
                transformed[f"{feature}_woe"] = bins.astype(str).map(mapping).fillna(rule.default_woe)
            else:
                mapping = rule.table.set_index("bin")["woe"].to_dict()
                transformed[f"{feature}_woe"] = (
                    x[feature].astype("object").fillna("MISSING").astype(str).map(mapping).fillna(rule.default_woe)
                )
        return transformed.astype(float)

    def bin_summary(self) -> pd.DataFrame:
        summaries = []
        for rule in self.rules_.values():
            table = rule.table.copy()
            table.insert(0, "feature", rule.feature)
            table.insert(1, "kind", rule.kind)
            summaries.append(table)
        return pd.concat(summaries, ignore_index=True)

    def _fit_numeric(self, feature: str, values: pd.Series, target: pd.Series) -> BinRule:
        clean = values.astype(float)
        try:
            categories, edges = pd.qcut(clean, q=self.max_bins, retbins=True, duplicates="drop")
        except ValueError:
            edges = np.array([clean.min(), clean.max()])
            categories = pd.cut(clean, bins=edges, include_lowest=True, duplicates="drop")

        edges = np.unique(edges.astype(float))
        if len(edges) < 2:
            edges = np.array([clean.min() - 1.0, clean.max() + 1.0])
        edges[0] = -np.inf
        edges[-1] = np.inf
        bins = pd.cut(clean, bins=edges, include_lowest=True, duplicates="drop")
        table = _woe_table(bins.astype(str), target)
        ordered_bins = [str(interval) for interval in bins.cat.categories]
        table["bin"] = pd.Categorical(table["bin"], categories=ordered_bins, ordered=True)
        table = table.sort_values("bin").reset_index(drop=True)
        table["bin"] = table["bin"].astype(str)
        return BinRule(feature=feature, kind="numeric", edges=edges.tolist(), table=table)

    def _fit_categorical(self, feature: str, values: pd.Series, target: pd.Series) -> BinRule:
        bins = values.astype("object").fillna("MISSING").astype(str)
        table = _woe_table(bins, target).sort_values("bad_rate").reset_index(drop=True)
        return BinRule(feature=feature, kind="categorical", edges=None, table=table)


def _woe_table(bins: pd.Series, target: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"bin": bins, "target": target})
    grouped = frame.groupby("bin", observed=True)["target"]
    table = grouped.agg(total="count", bad="sum").reset_index()
    table["good"] = table["total"] - table["bad"]

    total_good = table["good"].sum()
    total_bad = table["bad"].sum()
    smoothing = 0.5

    table["bad_rate"] = table["bad"] / table["total"]
    table["good_dist"] = (table["good"] + smoothing) / (total_good + smoothing * len(table))
    table["bad_dist"] = (table["bad"] + smoothing) / (total_bad + smoothing * len(table))
    table["woe"] = np.log(table["good_dist"] / table["bad_dist"])
    table["iv"] = (table["good_dist"] - table["bad_dist"]) * table["woe"]
    return table[["bin", "total", "good", "bad", "bad_rate", "woe", "iv"]]
