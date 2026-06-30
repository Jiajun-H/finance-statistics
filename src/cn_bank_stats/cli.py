"""Command-line entry point for the analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from cn_bank_stats.data_sources import (
    build_project_paths,
    ensure_output_dirs,
    load_credit_data,
    load_nfra_data,
    merge_quarterly_data,
)
from cn_bank_stats.metrics import (
    build_analysis_dataset,
    correlation_table,
    descriptive_table,
    risk_pressure_component_weights,
    risk_regime_summary,
)
from cn_bank_stats.modeling import (
    DEFAULT_CHANGE_MODEL_X,
    DEFAULT_CHANGE_MODEL_Y,
    DEFAULT_GRANGER_CAUSES,
    DEFAULT_MODEL_X,
    DEFAULT_VAR_COLUMNS,
    granger_tests,
    lagged_correlation_table,
    lagged_fitted_values,
    run_lagged_ols,
    run_ols,
    run_var_model,
    stationarity_tests,
    univariate_lagged_regressions,
)
from cn_bank_stats.plots import (
    plot_actual_vs_fitted,
    plot_capital_and_provision,
    plot_correlation_heatmap,
    plot_credit_trends,
    plot_granger_results,
    plot_loan_deposit_ratio,
    plot_nim_npl,
    plot_regression_diagnostics,
    plot_risk_pressure_index,
    plot_risk_regime_summary,
    plot_var_irf,
)

app = typer.Typer(add_completion=False, help="中国银行体系金融统计课程项目")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def run_analysis(
    start_year: int = 2019,
    end_year: int = 2025,
    refresh_pbc: bool = False,
    project_root: Path | None = None,
) -> dict[str, Path]:
    paths = build_project_paths(project_root)
    ensure_output_dirs(paths)

    credit = load_credit_data(paths, refresh_pbc=refresh_pbc)
    nfra = load_nfra_data(paths)
    merged = merge_quarterly_data(credit, nfra)
    merged = merged[(merged["year"] >= start_year) & (merged["year"] <= end_year)]
    analysis_df = build_analysis_dataset(merged)

    corr_columns = [
        "loans_trillion",
        "deposits_trillion",
        "loan_yoy",
        "loan_to_deposit_ratio",
        "npl_ratio",
        "nim",
        "provision_coverage",
        "capital_adequacy",
        "liquidity_ratio",
        "risk_pressure_index_z",
        "npl_ratio_qoq_change_pp",
        "loan_yoy_change_pp",
        "loan_to_deposit_ratio_qoq_change_pp",
        "nim_qoq_change_pp",
        "capital_adequacy_qoq_change_pp",
        "risk_pressure_index_z_change",
    ]
    corr = correlation_table(analysis_df, corr_columns)
    desc = descriptive_table(analysis_df, corr_columns)
    risk_weights = risk_pressure_component_weights(analysis_df)
    risk_regimes = risk_regime_summary(analysis_df)
    model, coef_table, fit_table = run_ols(analysis_df, x_cols=DEFAULT_MODEL_X)
    lagged_model, lagged_coef_table, lagged_fit_table = run_lagged_ols(
        analysis_df,
        x_cols=DEFAULT_MODEL_X,
        lag=1,
    )
    lagged_fitted = lagged_fitted_values(
        analysis_df,
        lagged_model,
        x_cols=DEFAULT_MODEL_X,
        lag=1,
    )
    change_lagged_model, change_lagged_coef_table, change_lagged_fit_table = run_lagged_ols(
        analysis_df,
        y_col=DEFAULT_CHANGE_MODEL_Y,
        x_cols=DEFAULT_CHANGE_MODEL_X,
        lag=1,
    )
    change_lagged_fitted = lagged_fitted_values(
        analysis_df,
        change_lagged_model,
        y_col=DEFAULT_CHANGE_MODEL_Y,
        x_cols=DEFAULT_CHANGE_MODEL_X,
        lag=1,
    )
    stationarity_columns = [
        "npl_ratio_qoq_change_pp",
        "loan_yoy_change_pp",
        "loan_to_deposit_ratio_qoq_change_pp",
        "nim_qoq_change_pp",
        "capital_adequacy_qoq_change_pp",
        "risk_pressure_index_z_change",
    ]
    stationarity = stationarity_tests(analysis_df, stationarity_columns)
    granger = granger_tests(
        analysis_df,
        target=DEFAULT_CHANGE_MODEL_Y,
        causes=DEFAULT_GRANGER_CAUSES,
        maxlag=2,
    )
    lagged_correlations = lagged_correlation_table(
        analysis_df,
        target=DEFAULT_CHANGE_MODEL_Y,
        causes=DEFAULT_GRANGER_CAUSES,
        max_lag=2,
    )
    univariate_lagged = univariate_lagged_regressions(
        analysis_df,
        target=DEFAULT_CHANGE_MODEL_Y,
        causes=DEFAULT_GRANGER_CAUSES,
        max_lag=2,
    )
    _, var_coef_table, var_fit_table, var_irf_table = run_var_model(
        analysis_df,
        columns=DEFAULT_VAR_COLUMNS,
        lag=1,
        periods=8,
    )

    written = {
        "credit_metrics": paths.tables_dir / "credit_metrics.csv",
        "nfra_indicators": paths.tables_dir / "nfra_indicators.csv",
        "analysis_dataset": paths.tables_dir / "analysis_dataset.csv",
        "descriptive_stats": paths.tables_dir / "descriptive_stats.csv",
        "correlations": paths.tables_dir / "correlations.csv",
        "risk_pressure_weights": paths.tables_dir / "risk_pressure_weights.csv",
        "risk_regime_summary": paths.tables_dir / "risk_regime_summary.csv",
        "regression_coefficients": paths.tables_dir / "regression_coefficients.csv",
        "regression_fit": paths.tables_dir / "regression_fit.csv",
        "lagged_regression_coefficients": paths.tables_dir
        / "lagged_regression_coefficients.csv",
        "lagged_regression_fit": paths.tables_dir / "lagged_regression_fit.csv",
        "lagged_regression_fitted": paths.tables_dir / "lagged_regression_fitted.csv",
        "change_lagged_regression_coefficients": paths.tables_dir
        / "change_lagged_regression_coefficients.csv",
        "change_lagged_regression_fit": paths.tables_dir / "change_lagged_regression_fit.csv",
        "change_lagged_regression_fitted": paths.tables_dir / "change_lagged_regression_fitted.csv",
        "stationarity_tests": paths.tables_dir / "stationarity_tests.csv",
        "lagged_correlations": paths.tables_dir / "lagged_correlations.csv",
        "univariate_lagged_regressions": paths.tables_dir
        / "univariate_lagged_regressions.csv",
        "granger_tests": paths.tables_dir / "granger_tests.csv",
        "var_coefficients": paths.tables_dir / "var_coefficients.csv",
        "var_fit": paths.tables_dir / "var_fit.csv",
        "var_irf_table": paths.tables_dir / "var_irf.csv",
        "credit_trends": paths.figures_dir / "credit_trends.png",
        "loan_deposit_ratio": paths.figures_dir / "loan_deposit_ratio.png",
        "nim_npl": paths.figures_dir / "nim_npl.png",
        "capital_provision": paths.figures_dir / "capital_provision.png",
        "correlation_heatmap": paths.figures_dir / "correlation_heatmap.png",
        "regression_diagnostics": paths.figures_dir / "regression_diagnostics.png",
        "risk_pressure_index": paths.figures_dir / "risk_pressure_index.png",
        "risk_regime_summary_plot": paths.figures_dir / "risk_regime_summary.png",
        "lagged_actual_vs_fitted": paths.figures_dir / "lagged_actual_vs_fitted.png",
        "granger_pvalues": paths.figures_dir / "granger_pvalues.png",
        "var_irf_plot": paths.figures_dir / "var_irf.png",
    }

    _write_csv(analysis_df, written["credit_metrics"])
    _write_csv(nfra, written["nfra_indicators"])
    _write_csv(analysis_df, written["analysis_dataset"])
    _write_csv(desc, written["descriptive_stats"])
    _write_csv(corr.reset_index().rename(columns={"index": "indicator"}), written["correlations"])
    _write_csv(risk_weights, written["risk_pressure_weights"])
    _write_csv(risk_regimes, written["risk_regime_summary"])
    _write_csv(coef_table, written["regression_coefficients"])
    _write_csv(fit_table, written["regression_fit"])
    _write_csv(lagged_coef_table, written["lagged_regression_coefficients"])
    _write_csv(lagged_fit_table, written["lagged_regression_fit"])
    _write_csv(lagged_fitted, written["lagged_regression_fitted"])
    _write_csv(change_lagged_coef_table, written["change_lagged_regression_coefficients"])
    _write_csv(change_lagged_fit_table, written["change_lagged_regression_fit"])
    _write_csv(change_lagged_fitted, written["change_lagged_regression_fitted"])
    _write_csv(stationarity, written["stationarity_tests"])
    _write_csv(lagged_correlations, written["lagged_correlations"])
    _write_csv(univariate_lagged, written["univariate_lagged_regressions"])
    _write_csv(granger, written["granger_tests"])
    _write_csv(var_coef_table, written["var_coefficients"])
    _write_csv(var_fit_table, written["var_fit"])
    _write_csv(var_irf_table, written["var_irf_table"])

    plot_credit_trends(analysis_df, written["credit_trends"])
    plot_loan_deposit_ratio(analysis_df, written["loan_deposit_ratio"])
    plot_nim_npl(analysis_df.dropna(subset=["npl_ratio", "nim"]), written["nim_npl"])
    plot_capital_and_provision(
        analysis_df.dropna(subset=["capital_adequacy", "provision_coverage"]),
        written["capital_provision"],
    )
    plot_correlation_heatmap(corr, written["correlation_heatmap"])
    plot_regression_diagnostics(model, written["regression_diagnostics"])
    plot_risk_pressure_index(analysis_df, written["risk_pressure_index"])
    plot_risk_regime_summary(risk_regimes, written["risk_regime_summary_plot"])
    plot_actual_vs_fitted(lagged_fitted, written["lagged_actual_vs_fitted"])
    plot_granger_results(granger, written["granger_pvalues"])
    plot_var_irf(var_irf_table, written["var_irf_plot"])

    return written


@app.command()
def main(
    start_year: Annotated[int, typer.Option(help="分析起始年份")] = 2019,
    end_year: Annotated[int, typer.Option(help="分析结束年份")] = 2025,
    refresh_pbc: Annotated[bool, typer.Option(help="尝试从人民银行官网下载并刷新信贷数据")] = False,
    project_root: Annotated[Path, typer.Option(help="项目根目录")] = Path("."),
) -> None:
    written = run_analysis(
        start_year=start_year,
        end_year=end_year,
        refresh_pbc=refresh_pbc,
        project_root=project_root,
    )
    typer.echo("分析完成，已生成以下文件：")
    for name, path in written.items():
        typer.echo(f"- {name}: {path}")
