from __future__ import annotations

import pandas as pd

from cn_bank_stats.metrics import (
    add_credit_metrics,
    add_indicator_changes,
    build_analysis_dataset,
    risk_pressure_component_weights,
    risk_regime_summary,
)
from cn_bank_stats.modeling import prepare_regression_frame


def _quarter_frame() -> pd.DataFrame:
    periods = pd.period_range("2020Q1", periods=8, freq="Q")
    return pd.DataFrame(
        {
            "period": periods.astype(str),
            "date": periods.to_timestamp(how="end").normalize(),
            "loans_trillion": [100, 102, 104, 106, 110, 112, 116, 120],
            "deposits_trillion": [140, 141, 143, 145, 150, 154, 158, 160],
            "npl_ratio": [1.9, 1.8, 1.7, 1.65, 1.6, 1.55, 1.5, 1.45],
            "nim": [2.1, 2.0, 1.95, 1.9, 1.85, 1.82, 1.8, 1.78],
            "provision_coverage": [190, 192, 195, 198, 200, 202, 205, 208],
            "capital_adequacy": [14.0, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7],
            "liquidity_ratio": [60, 61, 62, 63, 64, 65, 66, 67],
        }
    )


def test_credit_metrics_compute_yoy_and_loan_to_deposit_ratio() -> None:
    out = add_credit_metrics(_quarter_frame())
    assert round(out.loc[4, "loan_yoy"], 4) == 10.0
    assert round(out.loc[4, "deposit_yoy"], 4) == round((150 / 140 - 1) * 100, 4)
    assert round(out.loc[0, "loan_to_deposit_ratio"], 4) == round(100 / 140 * 100, 4)


def test_indicator_changes_use_percentage_points() -> None:
    out = add_indicator_changes(_quarter_frame())
    assert round(out.loc[1, "npl_ratio_qoq_change_pp"], 4) == -0.1
    assert round(out.loc[4, "nim_yoy_change_pp"], 4) == -0.25


def test_regression_frame_drops_incomplete_rows() -> None:
    df = add_credit_metrics(_quarter_frame())
    frame = prepare_regression_frame(df)
    assert len(frame) == 4
    assert set(["period", "npl_ratio", "loan_yoy", "loan_to_deposit_ratio"]).issubset(frame)


def test_risk_pressure_index_outputs_weights_and_regimes() -> None:
    df = build_analysis_dataset(_quarter_frame())
    weights = risk_pressure_component_weights(df)
    regimes = risk_regime_summary(df)

    assert df["risk_pressure_index_z"].notna().sum() == 4
    assert not weights.empty
    assert {"component", "weight", "abs_weight"}.issubset(weights.columns)
    assert not regimes.empty
    assert "risk_regime" in regimes.columns
