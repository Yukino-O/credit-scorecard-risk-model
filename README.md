# 信用评分卡风控模型

项目围绕经典评分卡流程展开：合成贷款申请数据、WOE 分箱、逻辑回归、分数刻度转换、模型验证，以及简单的授信策略分析。

项目重点不是调用黑箱分类器，而是展示对风控建模流程的理解。模型强调可解释性：每个信用分都可以追溯到申请人特征分箱和逻辑回归系数。

## 项目功能

- 生成可复现的零售贷款申请数据，不包含真实个人信息。
- 只使用训练集拟合 WOE 评分卡，避免测试集信息泄露。
- 将违约概率转换为信用分，分数越高表示风险越低。
- 输出 AUC、Gini、KS、坏账率、PSI、分数段坏账率和审批阈值对比。
- 保存评分卡表、验证集评分、图表和模型报告。
- 使用 pytest 覆盖分箱模块和端到端建模流程。
- 配置 GitHub Actions，便于在远程仓库中重复测试。

## 仓库结构

```text
src/credit_scorecard/      核心代码包
scripts/run_pipeline.py    端到端训练与报告生成脚本
tests/                     单元测试和集成测试
artifacts/                 生成的数据、报告、图表和模型文件
docs/project_brief.md      项目简介和简历表述
run_pipeline.ps1           Windows 一键运行脚本
```


## 快速开始

在 Windows PowerShell 中运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\run_pipeline.py
.\.venv\Scripts\python.exe -m pytest
```

也可以直接运行：

```powershell
.\run_pipeline.ps1
```

主要输出文件：

- `artifacts/reports/model_report.md`
- `artifacts/reports/scorecard_table.csv`
- `artifacts/reports/approval_cutoffs.csv`
- `artifacts/figures/roc_curve.png`
- `artifacts/figures/score_distribution.png`
- `artifacts/figures/bad_rate_by_score_band.png`
- `artifacts/models/credit_scorecard.joblib`

## 建模方法

模型采用零售信贷风控中常见的评分卡流程：

1. 生成或读取贷款申请数据。
2. 按目标变量分层划分训练集和留出测试集。
3. 使用训练集分位数对数值变量分箱。
4. 计算每个分箱的 WOE 和 Information Value。
5. 在 WOE 转换后的特征上拟合带类别权重的逻辑回归。
6. 将违约概率转换为积分制信用分。
7. 验证模型排序能力和分数稳定性。

信用分刻度设置：

- 基准分：650
- 基准好坏账户比：1 个好账户对应 1 个坏账户
- 好坏账户比翻倍所需分数：50

## 如何阅读输出

`model_report.md` 是最适合先看的文件。它总结了留出测试集表现，并展示不同审批阈值对通过率和坏账率的影响。

`scorecard_table.csv` 是模型解释层。它列出每个特征分箱、WOE、Information Value 贡献、回归系数和对应分数。

`approval_cutoffs.csv` 将模型结果转化为策略视角。审批阈值越高，通常通过人数越少，但通过人群的预期违约率也越低。

## 当前默认结果

使用默认 5,000 行合成数据时，留出测试集结果为：

- 留出测试集 AUC：0.707
- 留出测试集 Gini：0.414
- 留出测试集 KS：0.314
- 分数 PSI：0.0143

## 局限性

这不是生产级信贷系统。数据是合成的，因此本项目更适合作为建模流程、验证方法和可解释性展示。真实金融机构上线类似模型前，还需要公平性检验、还款能力评估、拒绝推断、模型监控、独立验证和监管审查。
