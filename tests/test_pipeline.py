from credit_scorecard.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, generate_credit_applications
from credit_scorecard.metrics import classification_metrics
from credit_scorecard.scorecard import CreditScorecard


def test_scorecard_fits_and_scores_synthetic_data():
    data = generate_credit_applications(n_rows=600, random_state=7)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    train = data.iloc[:420]
    test = data.iloc[420:]

    model = CreditScorecard(NUMERIC_FEATURES, CATEGORICAL_FEATURES, max_bins=4).fit(train[features], train[TARGET])
    probability_bad = model.predict_proba(test[features])
    scores = model.score(test[features])
    metrics = classification_metrics(test[TARGET], probability_bad, scores)

    assert len(scores) == len(test)
    assert scores.min() > 250
    assert scores.max() < 900
    assert metrics["auc"] > 0.6
