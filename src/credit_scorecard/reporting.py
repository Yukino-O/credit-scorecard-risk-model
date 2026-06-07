from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
from sklearn.metrics import RocCurveDisplay

from credit_scorecard.data import CATEGORY_VALUE_ZH, FEATURE_NAME_ZH


def _configure_chinese_font() -> None:
    font_paths = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for font_path in font_paths:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def localize_approval_table(table: pd.DataFrame) -> pd.DataFrame:
    return table.rename(
        columns={
            "score_cutoff": "审批阈值",
            "approval_rate": "通过率",
            "approved_default_rate": "通过人群坏账率",
            "decline_rate": "拒绝率",
        }
    )


def localize_feature_importance(table: pd.DataFrame) -> pd.DataFrame:
    localized = table.copy()
    if "feature" in localized.columns:
        localized["feature"] = localized["feature"].map(FEATURE_NAME_ZH).fillna(localized["feature"])
    return localized.rename(
        columns={
            "feature": "特征",
            "information_value": "Information Value",
            "abs_coefficient": "系数绝对值",
        }
    )


def localize_scorecard_table(table: pd.DataFrame) -> pd.DataFrame:
    localized = table.copy()
    localized["feature"] = localized["feature"].map(FEATURE_NAME_ZH).fillna(localized["feature"])
    localized["kind"] = localized["kind"].replace({"numeric": "数值变量", "categorical": "类别变量"})
    localized["bin"] = localized["bin"].map(CATEGORY_VALUE_ZH).fillna(localized["bin"])
    return localized.rename(
        columns={
            "feature": "特征",
            "kind": "变量类型",
            "bin": "分箱",
            "total": "样本数",
            "good": "好样本数",
            "bad": "坏样本数",
            "bad_rate": "坏账率",
            "woe": "WOE",
            "iv": "IV",
            "coefficient": "回归系数",
            "points": "分数",
        }
    )


def localize_score_band_table(table: pd.DataFrame) -> pd.DataFrame:
    return table.rename(
        columns={
            "score_band": "分数段",
            "applications": "申请数量",
            "bad_rate": "坏账率",
            "mean_score": "平均分",
        }
    )


def plot_score_distribution(train_scores, test_scores, path: Path) -> None:
    _configure_chinese_font()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(train_scores, bins=25, alpha=0.6, label="训练集")
    plt.hist(test_scores, bins=25, alpha=0.6, label="测试集")
    plt.xlabel("信用分")
    plt.ylabel("申请数量")
    plt.title("信用分分布")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_roc(y_true, probability_bad, path: Path) -> None:
    _configure_chinese_font()
    path.parent.mkdir(parents=True, exist_ok=True)
    RocCurveDisplay.from_predictions(y_true, probability_bad, name="评分卡模型")
    plt.title("留出测试集 ROC 曲线")
    plt.xlabel("假正例率")
    plt.ylabel("真正例率")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_bad_rate_by_score_band(y_true, scores, path: Path) -> pd.DataFrame:
    _configure_chinese_font()
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
    plt.ylabel("观察违约率")
    plt.title("不同分数段的违约率")
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
    approval_display = localize_approval_table(approval_table)
    importance_display = localize_feature_importance(feature_importance)
    lines = [
        "# 信用评分卡模型报告",
        "",
        "## 建模设置",
        "",
        "本项目使用 Weight of Evidence (WOE) 分箱和逻辑回归构建零售信贷风险评分卡。数据由透明的风险方程合成生成，不包含真实个人信息，因此可以公开放入 GitHub 仓库。",
        "",
        "## 留出测试集表现",
        "",
        "| 数据集 | AUC | Gini | KS | 坏账率 | 平均分 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| 训练集 | {train_metrics['auc']:.3f} | {train_metrics['gini']:.3f} | {train_metrics['ks']:.3f} | {train_metrics['bad_rate']:.3f} | {train_metrics['mean_score']:.1f} |",
        f"| 测试集 | {test_metrics['auc']:.3f} | {test_metrics['gini']:.3f} | {test_metrics['ks']:.3f} | {test_metrics['bad_rate']:.3f} | {test_metrics['mean_score']:.1f} |",
        "",
        f"训练集与测试集信用分之间的 Population Stability Index (PSI)：**{psi:.4f}**。",
        "",
        "## 审批阈值视角",
        "",
        approval_display.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## 主要预测变量",
        "",
        importance_display.head(8).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 结果解释",
        "",
        "信用分越高，代表模型估计的违约风险越低。授信方可以使用审批阈值表在业务增长和风险偏好之间做权衡。通常审批阈值越高，通过人数越少，但通过人群的观察违约率也会下降。",
        "",
        "## 局限性",
        "",
        "- 数据集是合成的，因此本项目展示的是建模流程，而不是生产环境中的真实信贷表现。",
        "- 真实上线前仍需要公平性检验、还款能力评估、拒绝推断、长期校准、模型监控和监管审查。",
        "- 本项目有意选择可解释评分卡；进一步扩展时可以加入梯度提升树模型作为性能基准，并与评分卡结果对比校准。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
