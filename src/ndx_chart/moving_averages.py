from __future__ import annotations

import numpy as np
import pandas as pd

# Conventional windows available to callers; the chart selects its visible
# subset in ndx_chart.dataset.
MOVING_AVERAGE_WINDOWS: tuple[int, ...] = (3, 7, 25, 50, 100, 150, 200, 250)


def moving_average_column(window: int) -> str:
    return f"sma_{window}"


def compute_moving_averages(
    frame: pd.DataFrame,
    *,
    price_column: str = "adj_close",
    windows: tuple[int, ...] = MOVING_AVERAGE_WINDOWS,
) -> pd.DataFrame:
    """Compute simple moving averages of `price_column` for one series.

    `frame` must already be restricted to a single series' canonical rows.
    Rows are sorted ascending by `session_date` before the rolling windows are
    applied, so each output row uses only that row's own session and prior
    sessions. A window is `NaN` until enough prior sessions exist (marking the
    warm-up period as unavailable rather than back-filling it).
    """
    ordered = frame.sort_values("session_date").reset_index(drop=True)
    result = ordered[["series", "session_date", price_column]].copy()
    for window in windows:
        result[moving_average_column(window)] = (
            ordered[price_column].rolling(window=window, min_periods=window).mean()
        )
    return result


def rate_of_change(series: pd.Series, *, periods: int = 1) -> pd.Series:
    """`series`'s own fractional change over `periods` sessions -- e.g.
    `0.01` means `series` is 1% higher than it was `periods` sessions ago.
    Null wherever `series` itself is null (its own warm-up) or fewer than
    `periods` prior rows exist yet -- never filled or estimated, the same
    convention `compute_moving_averages` uses for its own warm-up period.
    """
    return series.pct_change(periods=periods)


def ema_column(span: int) -> str:
    return f"ema_{span}"


def compute_ema(
    frame: pd.DataFrame,
    *,
    price_column: str = "adj_close",
    span: int,
) -> pd.Series:
    """Exponential moving average of `price_column` for one series, using
    the standard recursive (`adjust=False`) EMA convention most charting
    platforms use -- each session's value is a weighted blend of that
    session's price and the *previous session's EMA*, rather than a
    weighted average recomputed over the full history the way pandas'
    default `adjust=True` does.

    `frame` must already be restricted to a single series' canonical
    rows; it is sorted by `session_date` the same way
    `compute_moving_averages` sorts its own input, so a Series returned
    from this function aligns row-for-row with a `compute_moving_averages`
    result built from the same `frame`. Null for the first `span - 1`
    sessions -- the same explicit, never-back-filled warm-up convention
    `compute_moving_averages` uses for its own windows.
    """
    ordered = frame.sort_values("session_date").reset_index(drop=True)
    return ordered[price_column].ewm(span=span, adjust=False, min_periods=span).mean()


def rolling_zscore(series: pd.Series, *, window: int) -> pd.Series:
    """`series`'s own Z-score against its trailing rolling mean and
    standard deviation over `window` sessions (inclusive of the current
    one) -- how many standard deviations the current value sits from its
    own recent average, adapting as the series' own level and volatility
    drift over time (unlike a Z-score against one fixed, whole-history
    mean/std).

    Null wherever `series` itself is null, fewer than `window` prior rows
    exist yet, or the rolling standard deviation is exactly zero (would
    otherwise divide to +/-infinity) -- never filled or estimated, the
    same warm-up convention every other function in this module uses.
    `series` is used in whatever order it is already in; unlike
    `compute_moving_averages`/`compute_ema`, this function does not sort
    by `session_date` itself, since it is typically applied to a
    derived series (e.g. a price/EMA deviation) that is already ordered.
    """
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    zscore = (series - rolling_mean) / rolling_std
    return zscore.replace([np.inf, -np.inf], np.nan)


def rsi_column(period: int) -> str:
    return f"rsi_{period}"


def compute_rsi(
    frame: pd.DataFrame,
    *,
    price_column: str = "adj_close",
    period: int,
) -> pd.Series:
    """`strategies.rsi2_tqqq_v1`'s own signal (with `period=2`, Larry
    Connors' original "RSI(2)"): the Relative Strength Index of
    `price_column`'s own session-to-session changes over a trailing
    `period`-session window, using Wilder's exponential smoothing
    (`ewm(alpha=1/period, adjust=False)`, the same recursive convention
    `compute_ema` uses for its own moving average) of the average gain and
    average loss rather than a plain rolling mean of either.

    `frame` must already be restricted to a single series' canonical rows;
    it is sorted by `session_date` the same way `compute_moving_averages`
    and `compute_ema` sort their own input. Null for the first `period`
    sessions (`price_column`'s own first session has no prior session to
    diff against, then `period` more for the smoothed averages' own
    warm-up) -- never filled or estimated, the same convention every other
    function in this module uses. A session with no losses at all in its
    smoothed window scores `100.0` (maximally overbought, avoiding a
    divide-by-zero on `average_loss`); a session with no gains *and* no
    losses (price exactly flat throughout the window) scores the
    conventional neutral `50.0`.
    """
    ordered = frame.sort_values("session_date").reset_index(drop=True)
    price = ordered[price_column]
    change = price.diff()
    gain = change.clip(lower=0.0)
    loss = -change.clip(upper=0.0)
    average_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    average_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    relative_strength = average_gain / average_loss
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    rsi = rsi.where(average_loss != 0, other=100.0)
    rsi = rsi.where(~((average_gain == 0) & (average_loss == 0)), other=50.0)
    return rsi.where(~(average_gain.isna() | average_loss.isna()))


def rolling_quantile(series: pd.Series, *, window: int, quantile: float) -> pd.Series:
    """`series`'s own trailing `quantile` (0 to 1) over `window` sessions
    (inclusive of the current one) -- e.g. `quantile=0.95` gives, for each
    session, the value `series` has been below on 95% of the `window`
    prior sessions. Used for a threshold that recalibrates to how extreme
    `series` has actually been *recently*, rather than a fixed constant
    chosen once from a much longer (or different) history.

    Null wherever `series` itself is null or fewer than `window` prior
    rows exist yet -- never filled or estimated, the same warm-up
    convention `rolling_zscore` uses. `series` is used in whatever order
    it is already in, the same as `rolling_zscore` -- see that function's
    docstring for why.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be between 0 and 1, got {quantile}")
    return series.rolling(window=window, min_periods=window).quantile(quantile)
