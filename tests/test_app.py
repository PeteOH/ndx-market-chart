from __future__ import annotations

import numpy as np
import pandas as pd

from ndx_chart.app import normalise_history
from ndx_chart.charts import build_chart_html
from ndx_chart.dataset import build_analysis_dataset


def test_normalise_history_preserves_adjusted_prices() -> None:
    index = pd.DatetimeIndex(["2026-01-02", "2026-01-05"], tz="America/New_York")
    raw = pd.DataFrame(
        {
            "Open": [100.0, 102.0],
            "High": [103.0, 104.0],
            "Low": [99.0, 101.0],
            "Close": [102.0, 103.0],
            "Adj Close": [101.5, 102.5],
            "Volume": [1_000, 1_100],
            "Dividends": [0.0, 0.5],
            "Stock Splits": [0.0, 0.0],
        },
        index=index,
    )

    result = normalise_history(raw, series_id="TQQQ", source_symbol="TQQQ")

    assert result["series"].tolist() == ["TQQQ", "TQQQ"]
    assert result["session_date"].dt.tz is None
    assert result["adj_close"].tolist() == [101.5, 102.5]
    assert result["capital_gains"].tolist() == [0.0, 0.0]


def test_chart_contains_rulesets_and_new_moving_averages() -> None:
    sessions = pd.bdate_range("2024-01-02", periods=320)
    datasets = {
        series_id: build_analysis_dataset(_canonical_frame(series_id, sessions, scale))
        for series_id, scale in (("NDX", 20_000.0), ("TQQQ", 60.0), ("SQQQ", 300.0))
    }

    html = build_chart_html(datasets)

    assert "Ruleset highlights" in html
    assert "Buy / Sell" in html
    assert "SMA(10)" in html
    assert "SMA slope(7)" in html
    assert "SMA slope(10)" in html
    assert "ATR(5)" in html
    assert "ATR(7)" in html
    assert "ATR(10)" in html
    assert "+ Group" in html
    assert "Use + Group to create nested AND/OR parentheses" in html
    assert '"version": 2' in html
    assert "Only ruleset JSON versions 1 and 2 are supported" in html
    assert 'aria-label="ETF to trade"' in html
    assert '"etf": "TQQQ"' in html
    assert "ruleset.etf + ' price'" in html
    assert '"rule_etf":"TQQQ"' in html
    assert '"rule_etf":"SQQQ"' in html
    assert "Entry" in html
    assert "Current" in html


def test_analysis_dataset_calculates_all_atr_periods() -> None:
    sessions = pd.bdate_range("2026-01-02", periods=30)

    result = build_analysis_dataset(_canonical_frame("NDX", sessions, 20_000.0))

    for period in (5, 7, 10, 14):
        column = f"atr_{period}"
        assert column in result
        assert result[column].iloc[: period - 1].isna().all()
        assert pd.notna(result[column].iloc[period - 1])


def _canonical_frame(series_id: str, sessions: pd.DatetimeIndex, scale: float) -> pd.DataFrame:
    path = scale * (1.0 + np.linspace(0.0, 0.25, len(sessions)))
    return pd.DataFrame(
        {
            "series": series_id,
            "session_date": sessions,
            "open": path * 0.998,
            "high": path * 1.01,
            "low": path * 0.99,
            "close": path,
            "adj_close": path,
            "volume": np.arange(len(sessions)) + 1_000_000,
            "stock_splits": 0.0,
        }
    )
