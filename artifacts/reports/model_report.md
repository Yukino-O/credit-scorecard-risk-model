# Credit Scorecard Model Report

## Modelling Setup

This project builds a retail credit risk scorecard with Weight of Evidence (WOE) binning and logistic regression. The data is synthetic and generated from a transparent risk equation, so the repository can be shared publicly without personal data.

## Holdout Performance

| Split | AUC | Gini | KS | Bad rate | Mean score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 0.718 | 0.437 | 0.321 | 0.285 | 659.1 |
| Test | 0.707 | 0.414 | 0.314 | 0.285 | 659.2 |

Population Stability Index between train and test scores: **0.0143**.

## Policy Cutoff View

|   score_cutoff |   approval_rate |   approved_default_rate |   decline_rate |
|---------------:|----------------:|------------------------:|---------------:|
|        520.000 |           0.984 |                   0.276 |          0.016 |
|        560.000 |           0.941 |                   0.259 |          0.059 |
|        600.000 |           0.841 |                   0.232 |          0.159 |
|        640.000 |           0.656 |                   0.209 |          0.344 |
|        680.000 |           0.385 |                   0.133 |          0.615 |

## Strongest Predictors

| feature            |   information_value |   abs_coefficient |
|:-------------------|--------------------:|------------------:|
| delinquencies_2y   |              0.1909 |            0.7217 |
| credit_utilization |              0.1588 |            0.8984 |
| debt_to_income     |              0.1468 |            0.8989 |
| inquiries_6m       |              0.1229 |            0.6300 |
| loan_to_income     |              0.0575 |            0.7370 |
| home_ownership     |              0.0285 |            1.0578 |
| loan_purpose       |              0.0275 |            0.8854 |
| annual_income      |              0.0239 |            0.7574 |

## Interpretation

Higher scores mean lower estimated default risk. A lender could use the cutoff table to trade off growth and risk appetite. For example, a higher cutoff usually lowers observed default rate among approved applicants but rejects more applications.

## Limitations

- The dataset is synthetic, so the project demonstrates modelling workflow rather than production lending performance.
- Fair lending, affordability checks, reject inference, calibration over time, and regulatory sign-off would be required before any real deployment.
- The scorecard is intentionally interpretable; a more advanced project could benchmark gradient boosting and calibrate it against this scorecard.
