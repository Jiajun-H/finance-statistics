"""Financial-statistics indicators used by the course project."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_credit_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("date").copy()
    out["loan_yoy"] = out["loans_trillion"].pct_change(4) * 100
    out["deposit_yoy"] = out["deposits_trillion"].pct_change(4) * 100
    out["loan_qoq"] = out["loans_trillion"].pct_change() * 100
    out["deposit_qoq"] = out["deposits_trillion"].pct_change() * 100
    out["loan_to_deposit_ratio"] = out["loans_trillion"] / out["deposits_trillion"] * 100
    out["credit_gap_trillion"] = out["deposits_trillion"] - out["loans_trillion"]
    out["loan_yoy_change_pp"] = out["loan_yoy"].diff()
    out["deposit_yoy_change_pp"] = out["deposit_yoy"].diff()
    out["loan_to_deposit_ratio_qoq_change_pp"] = out["loan_to_deposit_ratio"].diff()
    out["credit_gap_qoq_change_trillion"] = out["credit_gap_trillion"].diff()
    return out


def add_indicator_changes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("date").copy()
    indicator_cols = [
        "npl_ratio",
        "nim",
        "provision_coverage",
        "capital_adequacy",
        "liquidity_ratio",
    ]
    for col in indicator_cols:
        if col in out.columns:
            out[f"{col}_qoq_change_pp"] = out[col].diff()
            out[f"{col}_yoy_change_pp"] = out[col].diff(4)
    return out


def build_analysis_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return build_risk_pressure_index(add_indicator_changes(add_credit_metrics(df)))


def correlation_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [col for col in columns if col in df.columns]
    corr = df[available].corr(numeric_only=True)
    return corr.round(4)


def descriptive_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [col for col in columns if col in df.columns]
    desc = df[available].describe().T
    return desc.rename_axis("indicator").reset_index().round(4)


def standardize(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return series * 0
    return (series - series.mean()) / std


def build_risk_pressure_index(df: pd.DataFrame) -> pd.DataFrame:
    """Build a PCA-style composite index for China banking risk pressure.

    Positive values mean higher pressure. The signs are chosen from financial
    logic before PCA: higher bad loans, faster credit expansion and higher
    loan-to-deposit ratio raise pressure; lower NIM, provision coverage, capital
    adequacy and liquidity ratio also raise pressure.
    """

    out = df.sort_values("date").copy()
    component_specs = {
        "npl_ratio": 1.0,
        "loan_yoy": 1.0,
        "loan_to_deposit_ratio": 1.0,
        "nim": -1.0,
        "provision_coverage": -1.0,
        "capital_adequacy": -1.0,
        "liquidity_ratio": -1.0,
    }
    available = [col for col in component_specs if col in out.columns]
    for col in available:
        out[f"{col}_pressure_z"] = standardize(out[col] * component_specs[col])

    z_cols = [f"{col}_pressure_z" for col in available]
    complete = out[z_cols].dropna()
    out["risk_pressure_index"] = pd.NA
    out["risk_pressure_index_method"] = "insufficient_data"
    if len(complete) >= 4 and len(z_cols) >= 2:
        matrix = complete.to_numpy(dtype=float)
        _, _, vh = np.linalg.svd(matrix, full_matrices=False)
        scores = matrix @ vh[0]
        score_series = pd.Series(scores, index=complete.index)
        npl_corr = (
            score_series.corr(out.loc[complete.index, "npl_ratio"])
            if "npl_ratio" in out.columns
            else 1
        )
        if npl_corr < 0:
            score_series = -score_series
        out.loc[complete.index, "risk_pressure_index"] = score_series
        out.loc[complete.index, "risk_pressure_index_method"] = "pca_first_component"
    elif z_cols:
        out["risk_pressure_index"] = out[z_cols].mean(axis=1)
        out["risk_pressure_index_method"] = "signed_zscore_average"

    out["risk_pressure_index"] = pd.to_numeric(out["risk_pressure_index"], errors="coerce")
    out["risk_pressure_index_z"] = standardize(out["risk_pressure_index"])
    out["risk_pressure_index_change"] = out["risk_pressure_index"].diff()
    out["risk_pressure_index_z_change"] = out["risk_pressure_index_z"].diff()
    return out


def risk_pressure_component_weights(df: pd.DataFrame) -> pd.DataFrame:
    z_cols = [col for col in df.columns if col.endswith("_pressure_z")]
    complete = df[z_cols].dropna()
    if len(complete) < 4 or len(z_cols) < 2:
        return pd.DataFrame(columns=["component", "weight", "abs_weight"])

    matrix = complete.to_numpy(dtype=float)
    _, _, vh = np.linalg.svd(matrix, full_matrices=False)
    weights = pd.Series(vh[0], index=z_cols)
    scores = matrix @ vh[0]
    score_series = pd.Series(scores, index=complete.index)
    if "npl_ratio" in df.columns and score_series.corr(df.loc[complete.index, "npl_ratio"]) < 0:
        weights = -weights

    result = (
        weights.rename("weight")
        .reset_index()
        .rename(columns={"index": "component"})
        .assign(
            component=lambda x: x["component"].str.replace("_pressure_z", "", regex=False),
            abs_weight=lambda x: x["weight"].abs(),
        )
        .sort_values("abs_weight", ascending=False)
        .reset_index(drop=True)
    )
    return result.round(6)


def risk_regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    needed = [
        "risk_pressure_index_z",
        "npl_ratio",
        "nim",
        "loan_yoy",
        "loan_to_deposit_ratio",
        "capital_adequacy",
        "liquidity_ratio",
    ]
    available = [col for col in needed if col in df.columns]
    frame = df.dropna(subset=["risk_pressure_index_z"]).copy()
    if frame.empty:
        return pd.DataFrame()

    frame["risk_regime"] = pd.qcut(
        frame["risk_pressure_index_z"],
        q=3,
        labels=["低压力", "中压力", "高压力"],
        duplicates="drop",
    )
    summary = (
        frame.groupby("risk_regime", observed=True)[available]
        .mean()
        .rename(columns={col: f"{col}_mean" for col in available})
        .reset_index()
    )
    summary["quarters"] = frame.groupby("risk_regime", observed=True).size().to_numpy()
    return summary.round(4)
