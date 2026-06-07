from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "employment_years",
    "debt_to_income",
    "credit_utilization",
    "delinquencies_2y",
    "inquiries_6m",
    "loan_amount",
    "loan_to_income",
]

CATEGORICAL_FEATURES = [
    "home_ownership",
    "loan_purpose",
    "education_level",
]

TARGET = "default_flag"

FEATURE_NAME_ZH = {
    "age": "年龄",
    "annual_income": "年收入",
    "employment_years": "工作年限",
    "debt_to_income": "债务收入比",
    "credit_utilization": "信用额度使用率",
    "delinquencies_2y": "近两年逾期次数",
    "inquiries_6m": "近六个月查询次数",
    "loan_amount": "贷款金额",
    "loan_to_income": "贷款收入比",
    "home_ownership": "住房状态",
    "loan_purpose": "贷款用途",
    "education_level": "教育水平",
}

CATEGORY_VALUE_ZH = {
    "MORTGAGE": "按揭房",
    "RENT": "租房",
    "OWN": "自有住房",
    "OTHER": "其他",
    "debt_consolidation": "债务整合",
    "credit_card": "信用卡",
    "home_improvement": "住房改善",
    "medical": "医疗",
    "small_business": "小微经营",
    "education": "教育",
    "high_school": "高中",
    "bachelor": "本科",
    "master_plus": "硕士及以上",
    "MISSING": "缺失",
}


def generate_credit_applications(
    n_rows: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """生成一个可复现、结构接近真实业务的合成信贷申请数据集。

    目标变量由明确的风险方程加噪声生成，因此项目可以在不暴露真实借款人
    隐私数据的前提下复现实验。
    """
    rng = np.random.default_rng(random_state)

    age = np.clip(rng.normal(39, 11, n_rows).round(), 21, 70).astype(int)
    annual_income = np.clip(rng.lognormal(mean=10.85, sigma=0.45, size=n_rows), 18000, 185000)
    employment_years = np.clip((age - 21) * rng.beta(2.0, 3.8, n_rows), 0, 38).round(1)

    home_ownership = rng.choice(
        ["MORTGAGE", "RENT", "OWN", "OTHER"],
        p=[0.43, 0.38, 0.16, 0.03],
        size=n_rows,
    )
    loan_purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "medical", "small_business", "education"],
        p=[0.42, 0.24, 0.13, 0.08, 0.07, 0.06],
        size=n_rows,
    )
    education_level = rng.choice(
        ["high_school", "bachelor", "master_plus"],
        p=[0.35, 0.47, 0.18],
        size=n_rows,
    )

    debt_to_income = np.clip(rng.beta(2.4, 5.2, n_rows) + rng.normal(0, 0.035, n_rows), 0.02, 0.88)
    credit_utilization = np.clip(rng.beta(2.0, 3.0, n_rows) + rng.normal(0, 0.06, n_rows), 0.01, 0.99)
    delinquencies_2y = rng.poisson(np.clip(0.18 + 1.1 * credit_utilization + 0.75 * debt_to_income, 0.05, 2.2))
    inquiries_6m = rng.poisson(np.clip(0.3 + 1.6 * debt_to_income + 0.35 * delinquencies_2y, 0.1, 4.0))
    loan_amount = np.clip(
        rng.normal(11500, 4200, n_rows)
        + annual_income * rng.uniform(0.03, 0.18, n_rows)
        + (loan_purpose == "small_business") * rng.normal(4500, 1200, n_rows),
        1000,
        42000,
    )
    loan_to_income = loan_amount / annual_income

    purpose_risk = pd.Series(loan_purpose).map(
        {
            "debt_consolidation": 0.15,
            "credit_card": 0.12,
            "home_improvement": -0.10,
            "medical": 0.20,
            "small_business": 0.42,
            "education": -0.08,
        }
    ).to_numpy()
    home_risk = pd.Series(home_ownership).map(
        {"MORTGAGE": -0.25, "RENT": 0.16, "OWN": -0.18, "OTHER": 0.30}
    ).to_numpy()
    education_risk = pd.Series(education_level).map(
        {"high_school": 0.16, "bachelor": -0.05, "master_plus": -0.18}
    ).to_numpy()

    logit = (
        -3.25
        + 2.25 * debt_to_income
        + 1.55 * credit_utilization
        + 0.35 * delinquencies_2y
        + 0.18 * inquiries_6m
        + 1.45 * loan_to_income
        - 0.012 * (age - 35)
        - 0.025 * employment_years
        + purpose_risk
        + home_risk
        + education_risk
        + rng.normal(0, 0.28, n_rows)
    )
    probability_default = 1.0 / (1.0 + np.exp(-logit))
    default_flag = rng.binomial(1, probability_default)

    frame = pd.DataFrame(
        {
            "applicant_id": [f"APP-{i:06d}" for i in range(1, n_rows + 1)],
            "age": age,
            "annual_income": annual_income.round(2),
            "employment_years": employment_years,
            "debt_to_income": debt_to_income.round(4),
            "credit_utilization": credit_utilization.round(4),
            "delinquencies_2y": delinquencies_2y.astype(int),
            "inquiries_6m": inquiries_6m.astype(int),
            "loan_amount": loan_amount.round(2),
            "loan_to_income": loan_to_income.round(4),
            "home_ownership": home_ownership,
            "loan_purpose": loan_purpose,
            "education_level": education_level,
            "default_flag": default_flag.astype(int),
        }
    )
    return frame


def load_or_generate_dataset(
    path: Path,
    n_rows: int = 5000,
    random_state: int = 42,
    force_regenerate: bool = False,
) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force_regenerate:
        return pd.read_csv(path)
    data = generate_credit_applications(n_rows=n_rows, random_state=random_state)
    data.to_csv(path, index=False)
    return data
