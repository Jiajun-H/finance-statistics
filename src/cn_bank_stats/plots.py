"""Plotting functions for paper-ready figures."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import font_manager


def setup_chinese_font() -> None:
    candidates = [
        Path("/mnt/c/Windows/Fonts/msyh.ttc"),
        Path("/mnt/c/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    ]
    for font_path in candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font=plt.rcParams["font.sans-serif"][0])


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _visible_lines(*axes) -> list:
    return [
        line
        for axis in axes
        for line in axis.get_lines()
        if not line.get_label().startswith("_")
    ]


def plot_credit_trends(df: pd.DataFrame, output_path: Path) -> None:
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["date"], df["loans_trillion"], marker="o", label="人民币贷款余额")
    ax.plot(df["date"], df["deposits_trillion"], marker="s", label="人民币存款余额")
    ax.set_title("中国金融机构人民币贷款与存款余额")
    ax.set_xlabel("季度")
    ax.set_ylabel("万亿元")
    ax.legend()
    _save(fig, output_path)


def plot_loan_deposit_ratio(df: pd.DataFrame, output_path: Path) -> None:
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(df["date"], df["loan_to_deposit_ratio"], color="#1f77b4", marker="o")
    ax.set_title("存贷比趋势")
    ax.set_xlabel("季度")
    ax.set_ylabel("%")
    _save(fig, output_path)


def plot_nim_npl(df: pd.DataFrame, output_path: Path) -> None:
    setup_chinese_font()
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(df["date"], df["npl_ratio"], color="#b22222", marker="o", label="不良贷款率")
    ax1.set_ylabel("不良贷款率（%）", color="#b22222")
    ax1.tick_params(axis="y", labelcolor="#b22222")

    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["nim"], color="#2e8b57", marker="s", label="净息差")
    ax2.set_ylabel("净息差（%）", color="#2e8b57")
    ax2.tick_params(axis="y", labelcolor="#2e8b57")

    ax1.set_title("商业银行净息差与不良贷款率")
    ax1.set_xlabel("季度")
    lines = _visible_lines(ax1, ax2)
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    _save(fig, output_path)


def plot_capital_and_provision(df: pd.DataFrame, output_path: Path) -> None:
    setup_chinese_font()
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(
        df["date"],
        df["capital_adequacy"],
        color="#4b4f97",
        marker="o",
        label="资本充足率",
    )
    ax1.set_ylabel("资本充足率（%）", color="#4b4f97")
    ax1.tick_params(axis="y", labelcolor="#4b4f97")

    ax2 = ax1.twinx()
    ax2.plot(
        df["date"],
        df["provision_coverage"],
        color="#bf6f00",
        marker="s",
        label="拨备覆盖率",
    )
    ax2.set_ylabel("拨备覆盖率（%）", color="#bf6f00")
    ax2.tick_params(axis="y", labelcolor="#bf6f00")

    ax1.set_title("商业银行资本缓冲与风险抵补能力")
    ax1.set_xlabel("季度")
    lines = _visible_lines(ax1, ax2)
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    _save(fig, output_path)


def plot_correlation_heatmap(corr: pd.DataFrame, output_path: Path) -> None:
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("主要金融统计指标相关系数")
    _save(fig, output_path)


def plot_regression_diagnostics(model, output_path: Path) -> None:
    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fitted = model.fittedvalues
    residuals = model.resid
    sns.scatterplot(x=fitted, y=residuals, ax=axes[0], color="#1f77b4")
    axes[0].axhline(0, color="#333333", linewidth=1)
    axes[0].set_title("OLS残差与拟合值")
    axes[0].set_xlabel("拟合值")
    axes[0].set_ylabel("残差")

    sm.qqplot(residuals, line="45", ax=axes[1], markerfacecolor="#1f77b4")
    axes[1].set_title("残差Q-Q图")
    _save(fig, output_path)


def plot_risk_pressure_index(df: pd.DataFrame, output_path: Path) -> None:
    setup_chinese_font()
    plot_df = df.dropna(subset=["risk_pressure_index_z", "npl_ratio"])
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.axhline(0, color="#333333", linewidth=0.9, linestyle="--")
    ax1.plot(
        plot_df["date"],
        plot_df["risk_pressure_index_z"],
        color="#7b3294",
        marker="o",
        label="综合金融压力指数",
    )
    ax1.set_ylabel("标准化指数", color="#7b3294")
    ax1.tick_params(axis="y", labelcolor="#7b3294")

    ax2 = ax1.twinx()
    ax2.plot(plot_df["date"], plot_df["npl_ratio"], color="#b22222", marker="s", label="不良贷款率")
    ax2.set_ylabel("不良贷款率（%）", color="#b22222")
    ax2.tick_params(axis="y", labelcolor="#b22222")

    ax1.set_title("中国银行体系综合金融压力指数")
    ax1.set_xlabel("季度")
    lines = _visible_lines(ax1, ax2)
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    _save(fig, output_path)


def plot_actual_vs_fitted(fitted_df: pd.DataFrame, output_path: Path) -> None:
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(fitted_df["date"], fitted_df["actual"], marker="o", label="实际不良贷款率")
    ax.plot(fitted_df["date"], fitted_df["fitted"], marker="s", label="滞后OLS拟合值")
    ax.set_title("不良贷款率动态解释模型：实际值与拟合值")
    ax.set_xlabel("季度")
    ax.set_ylabel("%")
    ax.legend()
    _save(fig, output_path)


def plot_granger_results(granger_df: pd.DataFrame, output_path: Path) -> None:
    setup_chinese_font()
    plot_df = granger_df.dropna(subset=["min_p_value"]).copy()
    plot_df = plot_df.sort_values("min_p_value", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#b22222" if p < 0.05 else "#4c78a8" for p in plot_df["min_p_value"]]
    ax.barh(plot_df["cause"], plot_df["min_p_value"], color=colors)
    ax.axvline(0.05, color="#333333", linestyle="--", linewidth=1, label="5%显著性水平")
    ax.set_title("Granger预测关系检验：解释变量是否领先不良贷款率")
    ax.set_xlabel("最小p值")
    ax.set_ylabel("候选领先变量")
    ax.legend()
    _save(fig, output_path)


def plot_var_irf(
    irf_df: pd.DataFrame,
    output_path: Path,
    response: str = "npl_ratio_qoq_change_pp",
    impulses: tuple[str, ...] = (
        "loan_yoy_change_pp",
        "nim_qoq_change_pp",
        "capital_adequacy_qoq_change_pp",
    ),
) -> None:
    setup_chinese_font()
    plot_df = irf_df[(irf_df["response"] == response) & (irf_df["impulse"].isin(impulses))]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = {
        "loan_yoy_change_pp": "贷款增速变化冲击",
        "nim_qoq_change_pp": "净息差变化冲击",
        "capital_adequacy_qoq_change_pp": "资本充足率变化冲击",
    }
    for impulse, group in plot_df.groupby("impulse"):
        ax.plot(group["horizon"], group["effect"], marker="o", label=labels.get(impulse, impulse))
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_title("VAR冲击响应：主要变量冲击对不良贷款率变化的影响")
    ax.set_xlabel("冲击后季度数")
    ax.set_ylabel("响应值")
    ax.legend()
    _save(fig, output_path)


def plot_risk_regime_summary(regime_df: pd.DataFrame, output_path: Path) -> None:
    setup_chinese_font()
    value_cols = [
        "npl_ratio_mean",
        "nim_mean",
        "capital_adequacy_mean",
        "liquidity_ratio_mean",
    ]
    plot_df = regime_df[["risk_regime", *value_cols]].melt(
        id_vars="risk_regime",
        var_name="indicator",
        value_name="value",
    )
    labels = {
        "npl_ratio_mean": "不良贷款率",
        "nim_mean": "净息差",
        "capital_adequacy_mean": "资本充足率",
        "liquidity_ratio_mean": "流动性比例",
    }
    plot_df["indicator"] = plot_df["indicator"].map(labels)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_df, x="risk_regime", y="value", hue="indicator", ax=ax)
    ax.set_title("不同金融压力状态下的银行监管指标均值")
    ax.set_xlabel("压力状态")
    ax.set_ylabel("%")
    ax.legend(title="指标")
    _save(fig, output_path)
