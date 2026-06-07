# Credit Scorecard Risk Model

This is a portfolio-style credit risk project built around a classic scorecard workflow: synthetic loan applications, WOE binning, logistic regression, score scaling, model validation, and a simple credit policy view.

The project is designed at a realistic undergraduate level for a mathematics student who wants to show applied modelling judgement, not just call a black-box classifier. The emphasis is interpretability: every score can be traced back to binned borrower characteristics and a logistic regression model.

## What The Project Does

- Generates a reproducible retail credit application dataset with no personal data.
- Trains a Weight of Evidence scorecard using only the training split.
- Converts model outputs into credit scores where higher scores mean lower risk.
- Reports AUC, Gini, KS, bad-rate stability, PSI, score-band bad rates, and approval cutoff trade-offs.
- Saves a fitted model, scoring table, holdout scores, plots, and a short model report.
- Includes pytest coverage for the binner and modelling pipeline.
- Includes a GitHub Actions workflow for repeatable testing.

## Repository Structure

```text
src/credit_scorecard/      Core package
scripts/run_pipeline.py    End-to-end training and reporting pipeline
tests/                     Unit and integration tests
artifacts/                 Generated data, reports, plots, and model files
docs/project_brief.md      Portfolio summary and resume bullets
run_pipeline.ps1           Windows one-command runner
```

The generated model file and synthetic row-level CSVs are ignored by Git because they can be recreated. The report tables and plots are kept in the repository so a reviewer can see the modelling results without running the code first.

## Quick Start

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\run_pipeline.py
.\.venv\Scripts\python.exe -m pytest
```

Or run:

```powershell
.\run_pipeline.ps1
```

Key outputs will be written to:

- `artifacts/reports/model_report.md`
- `artifacts/reports/scorecard_table.csv`
- `artifacts/reports/approval_cutoffs.csv`
- `artifacts/figures/roc_curve.png`
- `artifacts/figures/score_distribution.png`
- `artifacts/figures/bad_rate_by_score_band.png`
- `artifacts/models/credit_scorecard.joblib`

## Modelling Approach

The model follows a standard retail risk scorecard pattern:

1. Generate or load application data.
2. Split into train and holdout sets with stratification.
3. Bin numeric variables using training-set quantiles.
4. Estimate WOE and Information Value for each bin.
5. Fit a balanced logistic regression model on WOE-transformed features.
6. Convert probability of default into a points-based credit score.
7. Validate ranking performance and score stability.

The score scale uses:

- Base score: 650
- Base odds: 1 good account per 1 bad account
- Points to double odds: 50

## How To Read The Outputs

`model_report.md` is the best starting point. It summarises holdout performance and shows how different cutoffs change approval rate and observed default rate.

`scorecard_table.csv` is the interpretability layer. It lists every feature bin, WOE value, Information Value contribution, regression coefficient, and score points.

`approval_cutoffs.csv` turns the model into a decision view. A higher cutoff should approve fewer applicants with a lower expected default rate.

## Current Default Results

Using the default 5,000-row synthetic dataset:

- Holdout AUC: 0.707
- Holdout Gini: 0.414
- Holdout KS: 0.314
- Score PSI: 0.0143

## Limitations

This is not a production lending system. The dataset is synthetic, so the project is best read as a demonstration of modelling workflow, validation, and interpretability. A real lender would still need fairness testing, affordability checks, reject inference, monitoring, independent validation, and regulatory review.

## Resume Description

Built a reproducible credit risk scorecard in Python using WOE binning and logistic regression; generated score bands, AUC/KS/Gini validation, PSI stability checks, approval cutoff analysis, and interpretable scorecard outputs.
