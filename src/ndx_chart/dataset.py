from __future__ import annotations

import pandas as pd

from ndx_chart import indicators
from ndx_chart.moving_averages import (
    compute_moving_averages,
    moving_average_column,
)

# The export retains a few additional conventional windows for future chart
# experiments, while the visible picker uses CHART_SMA_WINDOWS below.
ANALYSIS_MOVING_AVERAGE_WINDOWS: tuple[int, ...] = (
    3,
    5,
    7,
    10,
    20,
    25,
    50,
    100,
    150,
    200,
    250,
)

CHART_SMA_WINDOWS: tuple[int, ...] = (3, 5, 7, 10, 20, 50, 200, 250)
CHART_SLOPE_WINDOWS: tuple[int, ...] = (5, 7, 10, 20, 50, 200, 250)

EMA_CHART_SPAN = 5
EMA_FAST_SPAN = 20
EMA_SLOW_SPAN = 50

PRICE_COLUMN = "adj_close"


def build_analysis_dataset(
    canonical: pd.DataFrame,
    *,
    windows: tuple[int, ...] = ANALYSIS_MOVING_AVERAGE_WINDOWS,
) -> pd.DataFrame:
    """Calculate one symbol's adjusted-price chart indicators.

    Adjusted close supplies every price-derived indicator. High and low are
    rescaled to the same basis before ATR and Chandelier Exit are calculated.
    Each lookback stays null until enough historical sessions exist.
    """
    sma_features = compute_moving_averages(canonical, price_column=PRICE_COLUMN, windows=windows)
    ordered = canonical.sort_values("session_date").reset_index(drop=True)
    price = ordered[PRICE_COLUMN]

    result = sma_features.rename(columns={PRICE_COLUMN: "price"})[
        ["series", "session_date", "price"]
    ].copy()
    for window in windows:
        result[moving_average_column(window)] = sma_features[moving_average_column(window)]

    for window in CHART_SLOPE_WINDOWS:
        column = moving_average_column(window)
        if column in sma_features:
            moving_average = sma_features[column]
        else:
            moving_average = compute_moving_averages(
                canonical, price_column=PRICE_COLUMN, windows=(window,)
            )[column]
        result[f"sma_slope_{window}"] = indicators.moving_average_slope(moving_average)

    result[f"ema_{EMA_CHART_SPAN}"] = indicators.ema(price, EMA_CHART_SPAN)
    ema_fast = indicators.ema(price, EMA_FAST_SPAN)
    ema_slow = indicators.ema(price, EMA_SLOW_SPAN)
    result[f"ema_{EMA_FAST_SPAN}"] = ema_fast
    result[f"ema_{EMA_SLOW_SPAN}"] = ema_slow
    result[f"price_above_ema_{EMA_FAST_SPAN}"] = _flag(price > ema_fast, ema_fast.notna())
    result[f"price_above_ema_{EMA_SLOW_SPAN}"] = _flag(price > ema_slow, ema_slow.notna())
    result[f"ema_{EMA_FAST_SPAN}_slope"] = indicators.ema_slope(price, span=EMA_FAST_SPAN)

    for period in (2, 3, 14):
        result[f"rsi_{period}"] = indicators.rsi(price, period=period)

    macd_frame = indicators.macd(price)
    result["macd_line"] = macd_frame["macd_line"]
    result["macd_signal"] = macd_frame["macd_signal"]
    result["macd_histogram"] = macd_frame["macd_histogram"]

    adjusted = indicators.adjusted_high_low(ordered)
    result["adj_high"] = adjusted["adj_high"]
    result["adj_low"] = adjusted["adj_low"]
    result["atr_14"] = indicators.average_true_range(
        adjusted["adj_high"], adjusted["adj_low"], price
    )
    chandelier = indicators.chandelier_exit(adjusted["adj_high"], adjusted["adj_low"], price)
    result["chandelier_long"] = chandelier["chandelier_long"]
    result["chandelier_short"] = chandelier["chandelier_short"]

    result["bb_width_20"] = indicators.bollinger_band_width(price)
    result["consecutive_streak"] = indicators.consecutive_streak(price)

    result["volume"] = ordered["volume"]
    result["volume_vs_avg_20"] = indicators.volume_vs_average(ordered["volume"])

    result["split_ratio"] = ordered["stock_splits"].apply(_format_split_ratio)

    result["return_1d"] = price.pct_change()
    result["drawdown_from_peak"] = indicators.drawdown_from_peak(price)
    result["distance_from_52w_high"] = indicators.distance_from_rolling_high(price)
    result["distance_from_52w_low"] = indicators.distance_from_rolling_low(price)

    # Computed independently of `windows` (a golden/death cross is always
    # SMA 50 vs SMA 200, by definition) rather than reused from
    # `sma_features`, which may not include one or both if a caller passed
    # a custom, narrower `windows`.
    cross_features = compute_moving_averages(
        canonical, price_column=PRICE_COLUMN, windows=(50, 200)
    )
    sma_50 = cross_features[moving_average_column(50)]
    sma_200 = cross_features[moving_average_column(200)]
    result["golden_cross"] = _flag(sma_50 > sma_200, sma_50.notna() & sma_200.notna())

    return result


def _flag(condition: pd.Series, valid: pd.Series) -> pd.Series:
    """`condition` as 1.0/0.0, null wherever `valid` is False -- e.g. "is
    price above its EMA" is meaningless before the EMA itself has warmed
    up, and must stay null rather than silently reading as 0 (`False`)."""
    return condition.astype(float).where(valid)


def _format_split_ratio(coefficient: float) -> str | None:
    """Yahoo's `stock_splits` convention: `coefficient` is how many new
    shares each existing share becomes that session (0.0 every other
    session -- no split). `2.0` is a 2-for-1 forward split, formatted here
    as `"2:1"`; `0.25` is a 1-for-4 reverse split (the pattern SQQQ's decay
    forces periodically), formatted as `"1:4"`. Returns `None` (empty in
    the CSV) on a no-split session rather than `"0:1"` or similar, so the
    column reads as blank except on the handful of real split sessions --
    see docs/ANALYSIS_CHARTS.md.
    """
    if pd.isna(coefficient) or coefficient == 0.0:
        return None
    if coefficient >= 1.0:
        return f"{_trim(coefficient)}:1"
    return f"1:{_trim(1.0 / coefficient)}"


def _trim(value: float) -> str:
    """`2.0` -> `"2"`, `2.5` -> `"2.5"` -- a whole-number split ratio (the
    overwhelming common case) reads as "2:1", not "2.0:1"."""
    return f"{value:g}"


def build_vix_reference(vix_canonical: pd.DataFrame) -> pd.DataFrame:
    """VIX's own level and 1-session point change, for `attach_cross_references`
    to merge into every ETF's dataset. Uses `close`, not `adj_close`: VIX is
    an index level, not a tradeable share, so it has no corporate actions to
    adjust for and Yahoo reports the two identically -- `close` is the more
    semantically honest column to reach for here.

    "VIX change" is a point change (`vix_level.diff()`, e.g. "VIX +2.3"),
    not a percentage -- the conventional way it is quoted in financial
    media, and how the CBOE itself reports it.
    """
    ordered = vix_canonical.sort_values("session_date").reset_index(drop=True)
    vix_level = ordered["close"]
    return pd.DataFrame(
        {
            "session_date": ordered["session_date"],
            "vix_level": vix_level,
            "vix_change": vix_level.diff(),
        }
    )


def attach_cross_references(
    dataset: pd.DataFrame,
    symbol_datasets: dict[str, pd.DataFrame],
    vix: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add every symbol's own 1-day return (`qqq_return`, `tqqq_return`,
    `sqqq_return` -- including this dataset's own symbol, so every file has
    the same, self-consistent set of reference columns) and VIX's level and
    change, joined on `session_date`. This is deliberately a separate step
    from `build_analysis_dataset`, which only ever sees one symbol's
    canonical data -- `analysis.service.AnalysisService.build` computes
    every symbol's own dataset first, then calls this once per symbol to
    merge the other two in.

    Left-joined: a row keeps every one of `dataset`'s own sessions even
    where a cross-referenced symbol or VIX has no data for that date (e.g.
    QQQ's dataset starts in 2010-01, TQQQ/SQQQ do not exist until
    2010-02-11 -- `tqqq_return`/`sqqq_return` are null for QQQ's own
    pre-inception sessions, not a dropped row).
    """
    result = dataset
    for symbol_id, other in symbol_datasets.items():
        if symbol_id not in {"QQQ", "TQQQ", "SQQQ"}:
            continue
        column = f"{symbol_id.lower()}_return"
        reference = other[["session_date", "return_1d"]].rename(columns={"return_1d": column})
        result = result.merge(reference, on="session_date", how="left")
    if vix is not None:
        result = result.merge(
            vix[["session_date", "vix_level", "vix_change"]], on="session_date", how="left"
        )
    else:
        result = result.assign(vix_level=pd.NA, vix_change=pd.NA)
    return result
