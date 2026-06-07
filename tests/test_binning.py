import pandas as pd

from credit_scorecard.binning import WoEBinner


def test_woe_binner_transforms_numeric_and_categorical_features():
    x = pd.DataFrame(
        {
            "income": [25000, 32000, 47000, 59000, 82000, 91000],
            "home": ["RENT", "RENT", "OWN", "MORTGAGE", "OWN", "MORTGAGE"],
        }
    )
    y = pd.Series([1, 1, 0, 0, 0, 0])

    binner = WoEBinner(["income"], ["home"], max_bins=3).fit(x, y)
    transformed = binner.transform(x)
    summary = binner.bin_summary()

    assert list(transformed.columns) == ["income_woe", "home_woe"]
    assert transformed.isna().sum().sum() == 0
    assert {"feature", "bin", "woe", "iv"}.issubset(summary.columns)


def test_woe_binner_handles_unseen_category_with_default_woe():
    x_train = pd.DataFrame({"income": [20000, 30000, 60000, 90000], "home": ["RENT", "OWN", "OWN", "MORTGAGE"]})
    y_train = pd.Series([1, 1, 0, 0])
    x_new = pd.DataFrame({"income": [45000], "home": ["OTHER"]})

    binner = WoEBinner(["income"], ["home"], max_bins=2).fit(x_train, y_train)
    transformed = binner.transform(x_new)

    assert transformed.shape == (1, 2)
    assert transformed.iloc[0]["home_woe"] == 0.0
