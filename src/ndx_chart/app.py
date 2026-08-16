from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from ndx_chart.charts import build_chart_html
from ndx_chart.dataset import build_analysis_dataset

LOGGER = logging.getLogger(__name__)

SERIES: tuple[tuple[str, str], ...] = (
    ("NDX", "^NDX"),
    ("TQQQ", "TQQQ"),
    ("SQQQ", "SQQQ"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download NDX/TQQQ/SQQQ history and build the interactive chart."
    )
    parser.add_argument(
        "--start",
        type=_iso_date,
        default=date(2010, 1, 1),
        help="First requested market date in YYYY-MM-DD format (default: 2010-01-01).",
    )
    parser.add_argument(
        "--end",
        type=_iso_date,
        default=date.today(),
        help="Last requested market date in YYYY-MM-DD format (default: today).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/index.html"),
        help="HTML output path (default: dist/index.html).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show each download, calculation, and output step.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    for noisy_logger in ("urllib3", "peewee", "yfinance"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    if args.end < args.start:
        raise SystemExit("--end must be on or after --start")

    datasets: dict[str, pd.DataFrame] = {}
    for series_id, source_symbol in SERIES:
        LOGGER.info(
            "Downloading %s (%s) from %s to %s",
            series_id,
            source_symbol,
            args.start,
            args.end,
        )
        canonical = download_history(series_id, source_symbol, args.start, args.end)
        LOGGER.info("Calculating indicators for %s (%s sessions)", series_id, len(canonical))
        datasets[series_id] = build_analysis_dataset(canonical)

    LOGGER.info("Rendering the interactive HTML chart")
    chart = build_chart_html(datasets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(chart, encoding="utf-8")
    LOGGER.info("Chart written to %s (%s bytes)", args.output.resolve(), args.output.stat().st_size)
    return 0


def download_history(
    series_id: str,
    source_symbol: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Return one Yahoo Finance history in the chart's canonical shape."""
    raw = yf.Ticker(source_symbol).history(
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=True,
        repair=False,
        keepna=False,
        raise_errors=True,
    )
    if raw.empty:
        raise RuntimeError(f"No market data returned for {source_symbol}")
    return normalise_history(raw, series_id=series_id, source_symbol=source_symbol)


def normalise_history(
    raw: pd.DataFrame,
    *,
    series_id: str,
    source_symbol: str,
) -> pd.DataFrame:
    source = raw.copy()
    source_index = pd.DatetimeIndex(pd.to_datetime(source.index))
    if source_index.tz is not None:
        source_index = source_index.tz_localize(None)
    source.index = source_index.normalize()

    adjusted_close = source["Adj Close"] if "Adj Close" in source else source["Close"]
    canonical = pd.DataFrame(
        {
            "series": series_id,
            "source_symbol": source_symbol,
            "session_date": source.index,
            "open": source["Open"],
            "high": source["High"],
            "low": source["Low"],
            "close": source["Close"],
            "adj_close": adjusted_close,
            "volume": source.get("Volume", 0.0),
            "dividends": source.get("Dividends", 0.0),
            "stock_splits": source.get("Stock Splits", 0.0),
            "capital_gains": source.get("Capital Gains", 0.0),
        }
    )
    numeric_columns = [
        column
        for column in canonical.columns
        if column not in {"series", "source_symbol", "session_date"}
    ]
    canonical[numeric_columns] = canonical[numeric_columns].apply(pd.to_numeric, errors="coerce")
    action_columns = ["dividends", "stock_splits", "capital_gains"]
    canonical[action_columns] = canonical[action_columns].fillna(0.0)
    return canonical.sort_values("session_date").reset_index(drop=True)


def _iso_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from exc
