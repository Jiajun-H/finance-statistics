# 中国宏观金融压力与银行风险统计分析

本项目用于金融统计学本科课程论文的代码部分。研究主题是：

> 中国宏观金融压力、信贷扩张与银行经营风险的动态统计分析。

核心变量来自中国金融机构信贷余额和商业银行监管指标，包括人民币贷款余额、人民币存款余额、存贷比、不良贷款率、净息差、拨备覆盖率、资本充足率和流动性比例。项目重点不是“画趋势图”，而是把这些指标组织成综合金融压力指数，并用滞后回归、平稳性检验、Granger 检验和 VAR 冲击响应分析动态关系。

## 环境与运行

在 WSL 终端进入本目录：

同步环境并运行分析：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync
UV_CACHE_DIR=/tmp/uv-cache uv run python analysis.py
```

如果你的 WSL 用户可以写默认 uv 缓存目录，也可以直接使用：

```bash
uv sync
uv run python analysis.py
```

运行后会生成：

- `outputs/tables/credit_metrics.csv`
- `outputs/tables/nfra_indicators.csv`
- `outputs/tables/analysis_dataset.csv`
- `outputs/tables/descriptive_stats.csv`
- `outputs/tables/correlations.csv`
- `outputs/tables/risk_pressure_weights.csv`
- `outputs/tables/risk_regime_summary.csv`
- `outputs/tables/regression_coefficients.csv`
- `outputs/tables/regression_fit.csv`
- `outputs/tables/lagged_regression_coefficients.csv`
- `outputs/tables/lagged_regression_fit.csv`
- `outputs/tables/stationarity_tests.csv`
- `outputs/tables/lagged_correlations.csv`
- `outputs/tables/univariate_lagged_regressions.csv`
- `outputs/tables/granger_tests.csv`
- `outputs/tables/var_coefficients.csv`
- `outputs/tables/var_fit.csv`
- `outputs/tables/var_irf.csv`
- `outputs/figures/credit_trends.png`
- `outputs/figures/loan_deposit_ratio.png`
- `outputs/figures/nim_npl.png`
- `outputs/figures/capital_provision.png`
- `outputs/figures/correlation_heatmap.png`
- `outputs/figures/regression_diagnostics.png`
- `outputs/figures/risk_pressure_index.png`
- `outputs/figures/risk_regime_summary.png`
- `outputs/figures/lagged_actual_vs_fitted.png`
- `outputs/figures/granger_pvalues.png`
- `outputs/figures/var_irf.png`

## 数据说明

项目内置两份可复现的整理数据：

- `data/manual/pbc_credit_summary.csv`：金融机构人民币贷款和存款季度余额，单位为万亿元。
- `data/manual/nfra_bank_indicators.csv`：商业银行监管指标，单位为百分比。

主要来源：

- 中国人民银行金融机构信贷收支统计：<https://www.pbc.gov.cn/diaochatongjisi/116219/116319/5225358/5225362/index.html>
- 国家金融监督管理总局 2024 年四季度监管指标：<https://www.nfra.gov.cn/cn/view/pages/ItemDetail.html?docId=1199327&itemId=915>
- 国家金融监督管理总局 2025 年二季度监管指标：<https://www.nfra.gov.cn/cn/view/pages/ItemDetail.html?docId=1221429&generaltype=0&itemId=915>

人民银行官网 Excel 下载结构可能随年度页面变化。代码支持尝试刷新人民银行 Excel 数据：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python analysis.py --refresh-pbc
```

课程论文定稿前，建议对 `data/manual/*.csv` 中的数据逐项核对并在论文参考文献或数据来源部分列明对应官网页面。

## 统计方法

代码会计算以下指标和模型：

- 贷款余额同比增速、环比增速。
- 存款余额同比增速、环比增速。
- 存贷比，即贷款余额 / 存款余额。
- 存贷缺口，即存款余额 - 贷款余额。
- 不良贷款率、净息差、拨备覆盖率、资本充足率、流动性比例的环比和同比百分点变化。
- 综合金融压力指数：先按金融含义统一方向并标准化，再用 PCA 第一主成分提取共同压力因子。
- 压力状态分组：按综合金融压力指数分成低压力、中压力、高压力，比较各组监管指标均值。
- 主要指标相关系数矩阵。
- HAC 稳健 OLS：用贷款同比增速、存贷比、净息差和资本充足率解释不良贷款率。
- 滞后 OLS：用上一季度金融指标解释本季度不良贷款率，减少同期相关导致的解释偏差。
- ADF 平稳性检验：重点检验不良贷款率、信贷增速、存贷比、净息差、资本充足率和综合压力指数的变化量，降低非平稳水平序列带来的伪回归风险。
- 滞后相关和单变量滞后回归：检验上一期或上两期金融变量变化是否与本期不良贷款率变化相关，更适合短季度样本。
- Granger 检验：作为补充检验，检验变量变化是否对不良贷款率变化有领先预测信息。
- VAR(1) 模型和冲击响应：使用变化量序列分析贷款增速变化、净息差变化、资本充足率变化对不良贷款率变化的动态影响。

## 论文写作建议

论文代码结果可以组织为五部分：

1. 信贷扩张与金融结构：根据贷款、存款和存贷比解释中国金融体系信贷扩张。
2. 综合金融压力度量：说明金融压力指数的构造逻辑、PCA 权重和低/中/高压力状态差异。
3. 盈利压力与信用风险：结合净息差、不良贷款率和风险压力指数讨论银行经营压力。
4. 动态解释模型：用 HAC OLS 和滞后 OLS 说明哪些变量与不良贷款率变化更相关。
5. 时间序列传导：优先用变化量 ADF、滞后相关和单变量滞后回归解释风险变化，再用 Granger 和 VAR 冲击响应作为补充动态证据。

## 测试

运行单元测试：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```
