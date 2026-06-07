# Project Brief

## One-line summary

A reproducible credit risk scorecard project that turns loan application features into an interpretable default-risk score using WOE binning and logistic regression.

## Why this project is useful

Credit scoring is a good risk analytics example because the model has to be explainable. A lender cannot only say that an applicant is risky; it also needs to understand which factors drove the decision, whether the model ranks risk well, and how a score cutoff changes approval rate.

This project keeps that workflow visible. The pipeline creates synthetic application data, trains the scorecard only on the training split, evaluates it on a holdout set, and writes both technical outputs and decision tables.

## Methods used

- Synthetic retail credit data generation
- Train/test split with stratification
- Weight of Evidence binning
- Information Value feature diagnostics
- Logistic regression
- Credit score scaling
- AUC, Gini, KS, PSI, and approval cutoff analysis
- Pytest validation and GitHub Actions CI

## Current results

On the default 5,000-row synthetic dataset, the holdout model reaches:

- AUC: 0.707
- Gini: 0.414
- KS: 0.314
- PSI: 0.0143

The result is intentionally moderate rather than over-optimised. It shows a realistic undergraduate-level modelling workflow with clear assumptions and reproducible outputs.

## Resume bullets

- Built a Python credit risk scorecard using WOE binning and logistic regression, with interpretable score points for each borrower feature band.
- Evaluated model ranking and stability using AUC, Gini, KS, PSI, score-band bad rates, and approval cutoff analysis.
- Packaged the project with reproducible scripts, generated reports, plots, pytest coverage, and GitHub Actions CI.
