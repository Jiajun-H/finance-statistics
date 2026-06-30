"""Regression and correlation helpers."""

from __future__ import annotations

import warnings

import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

DEFAULT_MODEL_Y = "npl_ratio"
DEFAULT_MODEL_X = ["loan_yoy", "loan_to_deposit_ratio", "nim", "capital_adequacy"]
DEFAULT_LAGGED_X = ["loan_yoy", "loan_to_deposit_ratio", "nim", "capital_adequacy"]
DEFAULT_CHANGE_MODEL_Y = "npl_ratio_qoq_change_pp"
DEFAULT_CHANGE_MODEL_X = [
    "loan_yoy_change_pp",
    "loan_to_deposit_ratio_qoq_change_pp",
    "nim_qoq_change_pp",
    "capital_adequacy_qoq_change_pp",
]
DEFAULT_GRANGER_CAUSES = [
    "loan_yoy_change_pp",
    "loan_to_deposit_ratio_qoq_change_pp",
    "nim_qoq_change_pp",
    "capital_adequacy_qoq_change_pp",
    "risk_pressure_index_z_change",
]
DEFAULT_VAR_COLUMNS = [
    "npl_ratio_qoq_change_pp",
    "loan_yoy_change_pp",
    "nim_qoq_change_pp",
    "capital_adequacy_qoq_change_pp",
]


def _model_tables(
    model,
    variable_names: list[str],
    dependent_variable: str,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = pd.Series(model.params, index=variable_names)
    bse = pd.Series(model.bse, index=variable_names)
    tvalues = pd.Series(model.tvalues, index=variable_names)
    pvalues = pd.Series(model.pvalues, index=variable_names)
    coef_table = pd.DataFrame(
        {
            "model": model_name,
            "variable": variable_names,
            "coef": params.values,
            "std_err": bse.values,
            "t_value": tvalues.values,
            "p_value": pvalues.values,
        }
    ).round(6)
    fit_table = pd.DataFrame(
        [
            {
                "model": model_name,
                "dependent_variable": dependent_variable,
                "nobs": int(model.nobs),
                "r_squared": round(float(model.rsquared), 6),
                "adj_r_squared": round(float(model.rsquared_adj), 6),
                "aic": round(float(model.aic), 6),
                "bic": round(float(model.bic), 6),
                "durbin_watson": round(float(durbin_watson(model.resid)), 6),
                "f_statistic": round(float(model.fvalue), 6) if model.fvalue is not None else None,
                "f_p_value": round(float(model.f_pvalue), 6)
                if model.f_pvalue is not None
                else None,
            }
        ]
    )
    return coef_table, fit_table


def prepare_regression_frame(
    df: pd.DataFrame,
    y_col: str = DEFAULT_MODEL_Y,
    x_cols: list[str] | None = None,
) -> pd.DataFrame:
    regressors = x_cols or DEFAULT_MODEL_X
    cols = ["period", y_col, *regressors]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Regression columns are missing: {missing}")
    return df[cols].dropna().reset_index(drop=True)


def run_ols(
    df: pd.DataFrame,
    y_col: str = DEFAULT_MODEL_Y,
    x_cols: list[str] | None = None,
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame, pd.DataFrame]:
    regressors = x_cols or DEFAULT_MODEL_X
    model_frame = prepare_regression_frame(df, y_col=y_col, x_cols=regressors)
    if len(model_frame) <= len(regressors) + 2:
        raise ValueError("Not enough observations for the OLS model.")

    y = model_frame[y_col]
    x = sm.add_constant(model_frame[regressors], has_constant="add")
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    coef_table, fit_table = _model_tables(
        model,
        list(x.columns),
        dependent_variable=y_col,
        model_name="contemporaneous_hac_ols",
    )
    return model, coef_table, fit_table


def add_lagged_columns(df: pd.DataFrame, columns: list[str], lag: int = 1) -> pd.DataFrame:
    out = df.sort_values("date").copy()
    for col in columns:
        out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def run_lagged_ols(
    df: pd.DataFrame,
    y_col: str = DEFAULT_MODEL_Y,
    x_cols: list[str] | None = None,
    lag: int = 1,
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame, pd.DataFrame]:
    regressors = x_cols or DEFAULT_LAGGED_X
    lagged_df = add_lagged_columns(df, regressors, lag=lag)
    lagged_cols = [f"{col}_lag{lag}" for col in regressors]
    cols = ["period", y_col, *lagged_cols]
    model_frame = lagged_df[cols].dropna().reset_index(drop=True)
    if len(model_frame) <= len(lagged_cols) + 2:
        raise ValueError("Not enough observations for the lagged OLS model.")

    y = model_frame[y_col]
    x = sm.add_constant(model_frame[lagged_cols], has_constant="add")
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    coef_table, fit_table = _model_tables(
        model,
        list(x.columns),
        dependent_variable=y_col,
        model_name=f"lag{lag}_hac_ols",
    )
    return model, coef_table, fit_table


def lagged_fitted_values(
    df: pd.DataFrame,
    model,
    y_col: str = DEFAULT_MODEL_Y,
    x_cols: list[str] | None = None,
    lag: int = 1,
) -> pd.DataFrame:
    regressors = x_cols or DEFAULT_LAGGED_X
    lagged_cols = [f"{col}_lag{lag}" for col in regressors]
    frame = add_lagged_columns(df, regressors, lag=lag)
    model_frame = frame[["period", "date", y_col, *lagged_cols]].dropna().copy()
    model_frame["fitted"] = model.fittedvalues
    return model_frame[["period", "date", y_col, "fitted"]].rename(columns={y_col: "actual"})


def lagged_correlation_table(
    df: pd.DataFrame,
    target: str = DEFAULT_CHANGE_MODEL_Y,
    causes: list[str] | None = None,
    max_lag: int = 2,
) -> pd.DataFrame:
    cause_cols = causes or DEFAULT_GRANGER_CAUSES
    rows = []
    for cause in cause_cols:
        if target not in df or cause not in df:
            rows.append(
                {
                    "target": target,
                    "cause": cause,
                    "lag": None,
                    "nobs": 0,
                    "correlation": None,
                    "p_value": None,
                    "result": "变量缺失",
                }
            )
            continue
        for lag in range(1, max_lag + 1):
            pair = pd.DataFrame(
                {
                    "target": pd.to_numeric(df[target], errors="coerce"),
                    "cause_lagged": pd.to_numeric(df[cause], errors="coerce").shift(lag),
                }
            ).dropna()
            if len(pair) < 6 or pair["target"].nunique() < 3 or pair["cause_lagged"].nunique() < 3:
                rows.append(
                    {
                        "target": target,
                        "cause": cause,
                        "lag": lag,
                        "nobs": len(pair),
                        "correlation": None,
                        "p_value": None,
                        "result": "样本不足",
                    }
                )
                continue
            corr, p_value = pearsonr(pair["cause_lagged"], pair["target"])
            rows.append(
                {
                    "target": target,
                    "cause": cause,
                    "lag": lag,
                    "nobs": len(pair),
                    "correlation": round(float(corr), 6),
                    "p_value": round(float(p_value), 6),
                    "result": "显著滞后相关" if p_value < 0.05 else "未通过5%显著性",
                }
            )
    return pd.DataFrame(rows)


def univariate_lagged_regressions(
    df: pd.DataFrame,
    target: str = DEFAULT_CHANGE_MODEL_Y,
    causes: list[str] | None = None,
    max_lag: int = 2,
) -> pd.DataFrame:
    cause_cols = causes or DEFAULT_GRANGER_CAUSES
    rows = []
    for cause in cause_cols:
        if target not in df or cause not in df:
            continue
        for lag in range(1, max_lag + 1):
            frame = pd.DataFrame(
                {
                    "target": pd.to_numeric(df[target], errors="coerce"),
                    "cause_lagged": pd.to_numeric(df[cause], errors="coerce").shift(lag),
                }
            ).dropna()
            too_few_observations = len(frame) < 8
            too_little_variation = (
                frame["target"].nunique() < 3 or frame["cause_lagged"].nunique() < 3
            )
            if too_few_observations or too_little_variation:
                rows.append(
                    {
                        "target": target,
                        "cause": cause,
                        "lag": lag,
                        "nobs": len(frame),
                        "coef": None,
                        "std_err": None,
                        "t_value": None,
                        "p_value": None,
                        "r_squared": None,
                        "result": "样本不足",
                    }
                )
                continue
            model = sm.OLS(
                frame["target"],
                sm.add_constant(frame[["cause_lagged"]], has_constant="add"),
            ).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
            p_value = float(model.pvalues["cause_lagged"])
            rows.append(
                {
                    "target": target,
                    "cause": cause,
                    "lag": lag,
                    "nobs": int(model.nobs),
                    "coef": round(float(model.params["cause_lagged"]), 6),
                    "std_err": round(float(model.bse["cause_lagged"]), 6),
                    "t_value": round(float(model.tvalues["cause_lagged"]), 6),
                    "p_value": round(p_value, 6),
                    "r_squared": round(float(model.rsquared), 6),
                    "result": "显著滞后回归" if p_value < 0.05 else "未通过5%显著性",
                }
            )
    return pd.DataFrame(rows)


def stationarity_tests(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna() if col in df else pd.Series()
        if len(series) < 8 or series.nunique() < 3:
            rows.append(
                {
                    "variable": col,
                    "nobs": len(series),
                    "adf_stat": None,
                    "p_value": None,
                    "used_lag": None,
                    "result": "样本不足",
                }
            )
            continue
        try:
            maxlag = min(2, max(0, len(series) // 4 - 1))
            stat, p_value, used_lag, nobs, *_ = adfuller(series, maxlag=maxlag, autolag="AIC")
            rows.append(
                {
                    "variable": col,
                    "nobs": int(nobs),
                    "adf_stat": round(float(stat), 6),
                    "p_value": round(float(p_value), 6),
                    "used_lag": int(used_lag),
                    "result": "平稳" if p_value < 0.05 else "可能非平稳",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "variable": col,
                    "nobs": len(series),
                    "adf_stat": None,
                    "p_value": None,
                    "used_lag": None,
                    "result": f"检验失败: {exc}",
                }
            )
    return pd.DataFrame(rows)


def granger_tests(
    df: pd.DataFrame,
    target: str = DEFAULT_CHANGE_MODEL_Y,
    causes: list[str] | None = None,
    maxlag: int = 2,
) -> pd.DataFrame:
    cause_cols = causes or DEFAULT_GRANGER_CAUSES
    rows = []
    for cause in cause_cols:
        pair = df[[target, cause]].dropna() if target in df and cause in df else pd.DataFrame()
        if len(pair) <= maxlag * 3 + 2:
            rows.append(
                {
                    "target": target,
                    "cause": cause,
                    "best_lag": None,
                    "min_p_value": None,
                    "result": "样本不足",
                }
            )
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = grangercausalitytests(pair[[target, cause]], maxlag=maxlag, verbose=False)
            p_values = {
                lag: tests[0]["ssr_ftest"][1]
                for lag, tests in result.items()
                if "ssr_ftest" in tests[0]
            }
            best_lag = min(p_values, key=p_values.get)
            p_value = p_values[best_lag]
            rows.append(
                {
                    "target": target,
                    "cause": cause,
                    "best_lag": int(best_lag),
                    "min_p_value": round(float(p_value), 6),
                    "result": "存在Granger预测关系" if p_value < 0.05 else "未通过5%显著性",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "target": target,
                    "cause": cause,
                    "best_lag": None,
                    "min_p_value": None,
                    "result": f"检验失败: {exc}",
                }
            )
    return pd.DataFrame(rows)


def run_var_model(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    lag: int = 1,
    periods: int = 8,
) -> tuple[object, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    var_cols = columns or DEFAULT_VAR_COLUMNS
    frame = df[var_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) <= len(var_cols) * lag + 4:
        raise ValueError("Not enough observations for the VAR model.")

    fitted = VAR(frame).fit(lag)
    coef_rows = []
    for equation in fitted.names:
        params = fitted.params[equation]
        stderr = fitted.stderr[equation]
        tvalues = fitted.tvalues[equation]
        pvalues = fitted.pvalues[equation]
        for variable in params.index:
            coef_rows.append(
                {
                    "equation": equation,
                    "variable": variable,
                    "coef": params[variable],
                    "std_err": stderr[variable],
                    "t_value": tvalues[variable],
                    "p_value": pvalues[variable],
                }
            )
    coef_table = pd.DataFrame(coef_rows).round(6)
    fit_table = pd.DataFrame(
        [
            {
                "model": f"VAR({lag})",
                "nobs": int(fitted.nobs),
                "variables": ", ".join(fitted.names),
                "aic": round(float(fitted.aic), 6),
                "bic": round(float(fitted.bic), 6),
                "hqic": round(float(fitted.hqic), 6),
                "fpe": f"{float(fitted.fpe):.6e}",
            }
        ]
    )

    irf = fitted.irf(periods)
    irf_rows = []
    for horizon in range(periods + 1):
        for response_idx, response in enumerate(fitted.names):
            for impulse_idx, impulse in enumerate(fitted.names):
                irf_rows.append(
                    {
                        "horizon": horizon,
                        "response": response,
                        "impulse": impulse,
                        "effect": irf.irfs[horizon, response_idx, impulse_idx],
                    }
                )
    irf_table = pd.DataFrame(irf_rows).round(8)
    return fitted, coef_table, fit_table, irf_table
