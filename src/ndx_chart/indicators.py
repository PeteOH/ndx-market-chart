from __future__ import annotations

import numpy as np
import pandas as pd

# Standard, widely used lookback periods. They are conventional chart defaults,
# not tuned or validated trading rules.
RSI_PERIOD = 14
MACD_FAST_SPAN = 12
MACD_SLOW_SPAN = 26
MACD_SIGNAL_SPAN = 9
ATR_PERIOD = 14
BOLLINGER_WINDOW = 20
BOLLINGER_NUM_STD = 2.0
VOLUME_AVERAGE_WINDOW = 20
TRAILING_YEAR_SESSIONS = 252
CHANDELIER_PERIOD = 22
CHANDELIER_MULTIPLIER = 3.0


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average, null for the first `span - 1` sessions.

    Mathematically an EMA is defined from the first observation, but its
    early values are dominated by the seed and not a meaningful trend
    signal -- nulling the warm-up period matches
    `features.moving_averages.compute_moving_averages`'s SMA convention
    (null, never back-filled, until `span` sessions of history exist), so
    every indicator in this module treats "not enough history yet" the
    same way.
    """
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def ema_slope(series: pd.Series, *, span: int, lookback: int = 5) -> pd.Series:
    """The EMA's own rate of change over `lookback` sessions -- e.g. +0.02
    means the EMA is 2% higher than it was `lookback` sessions ago. A crude
    but standard trend-strength proxy: a flattening slope near zero often
    precedes a reversal, before price itself crosses the EMA."""
    return ema(series, span).pct_change(periods=lookback)


def moving_average_slope(moving_average: pd.Series, *, lookback: int = 1) -> pd.Series:
    """Fractional change in a moving average over ``lookback`` sessions.

    This keeps slopes comparable across a 20,000-point index and a $50 ETF:
    ``0.01`` means the moving average rose 1%, regardless of price level.
    The chart's ``sma_slope(N)`` means the one-session change of SMA(N).
    """
    return moving_average.pct_change(periods=lookback)


def rsi(price: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's Relative Strength Index. 0-100; conventionally >70
    "overbought", <30 "oversold" (not a rule this project applies anywhere,
    just the standard reading). Null for the first `period` sessions.

    Wilder's original smoothing is an EMA with alpha = 1/period (not the
    plain SMA some simplified descriptions use) -- `pandas.Series.ewm`'s
    `alpha` parameter matches this directly.
    """
    delta = price.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    # Ordinary float division already gives the right edge-case answers: an
    # all-gains window has avg_loss 0 so rs is +inf, giving RSI 100; an
    # all-losses window has avg_gain 0 so rs is 0, giving RSI 0. Only a
    # perfectly flat window (avg_gain and avg_loss both 0) divides 0/0 to
    # NaN and needs an explicit override, to the conventional neutral 50.
    rs = avg_gain / avg_loss
    result = 100.0 - (100.0 / (1.0 + rs))
    perfectly_flat = (avg_gain == 0.0) & (avg_loss == 0.0)
    return result.where(~perfectly_flat, 50.0)


def macd(
    price: pd.Series,
    *,
    fast_span: int = MACD_FAST_SPAN,
    slow_span: int = MACD_SLOW_SPAN,
    signal_span: int = MACD_SIGNAL_SPAN,
) -> pd.DataFrame:
    """Standard MACD: `macd_line` = fast EMA - slow EMA, `macd_signal` = EMA
    of `macd_line`, `macd_histogram` = `macd_line` - `macd_signal` (the
    conventional bar-chart panel; its sign flips are the "MACD crossover"
    signal). Each column is null until enough history feeds its own EMA."""
    macd_line = ema(price, fast_span) - ema(price, slow_span)
    signal_line = ema(macd_line, signal_span)
    return pd.DataFrame(
        {
            "macd_line": macd_line,
            "macd_signal": signal_line,
            "macd_histogram": macd_line - signal_line,
        }
    )


def adjusted_high_low(frame: pd.DataFrame) -> pd.DataFrame:
    """`high` and `low` rescaled by the same cumulative dividend adjustment
    already baked into `adj_close` (Yahoo provides an adjusted close but not
    an adjusted high/low). `close` is itself already split-adjusted across
    the entire fetched history the same way `adj_close` is -- only
    dividends distinguish the two -- so `adjustment_ratio = adj_close /
    close` isolates the dividend adjustment alone; a split contributes
    nothing to this ratio since it is already reflected identically in both
    the numerator and the denominator. A `close` of zero or NaN (never
    observed in real data, guarded for anyway) yields a NaN ratio rather
    than a divide error.
    """
    ratio = frame["adj_close"] / frame["close"].replace(0.0, np.nan)
    return pd.DataFrame({"adj_high": frame["high"] * ratio, "adj_low": frame["low"] * ratio})


def average_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD
) -> pd.Series:
    """Wilder's Average True Range: a volatility measure, in price units,
    that (unlike a simple high-low range) accounts for gaps by including
    the distance from the previous close. `high`/`low`/`close` must already
    be on the same (split-adjusted) basis -- see `adjusted_high_low`."""
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def chandelier_exit(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    period: int = CHANDELIER_PERIOD,
    multiplier: float = CHANDELIER_MULTIPLIER,
) -> pd.DataFrame:
    """Conventional long and short Chandelier Exit levels.

    The long exit is the period high minus ``multiplier`` times Wilder ATR;
    the short exit is the period low plus the same amount. The standard
    defaults are 22 sessions and 3 ATR. Inputs must share an adjusted price
    basis, exactly like :func:`average_true_range`.
    """
    atr = average_true_range(high, low, close, period=period)
    rolling_high = high.rolling(window=period, min_periods=period).max()
    rolling_low = low.rolling(window=period, min_periods=period).min()
    return pd.DataFrame(
        {
            "chandelier_long": rolling_high - multiplier * atr,
            "chandelier_short": rolling_low + multiplier * atr,
        }
    )


def bollinger_band_width(
    price: pd.Series, window: int = BOLLINGER_WINDOW, num_std: float = BOLLINGER_NUM_STD
) -> pd.Series:
    """Bollinger Band width, normalised by the middle band: `(upper -
    lower) / sma`, where `upper`/`lower` are the `window`-session SMA +/-
    `num_std` sample standard deviations. Normalised (a fraction, not raw
    price units) so it is comparable across symbols at very different price
    levels -- QQQ at $700 and SQQQ at $60 are not on the same absolute
    scale, but their band widths as a fraction of price are. A narrowing
    width ("squeeze") conventionally precedes a sharp move; it does not say
    which direction."""
    rolling = price.rolling(window=window, min_periods=window)
    middle = rolling.mean()
    std = rolling.std()
    return (2 * num_std * std) / middle


def consecutive_streak(price: pd.Series) -> pd.Series:
    """Signed run length of consecutive up (positive) or down (negative)
    sessions: +3 means today is the 3rd straight higher-close session, -1
    means today broke a streak by closing lower. Resets to 0 on a flat
    session (close unchanged) and is NaN for the first session (no prior
    close to compare against). Not vectorisable in the usual pandas sense
    (each value depends on the running streak, not a fixed window), so this
    is a plain sequential scan -- clear and easily checked by hand, and
    cheap at this data's size (thousands of rows, not millions)."""
    direction = np.sign(price.diff())
    streak: list[float] = []
    current = 0
    for value in direction:
        if pd.isna(value):
            streak.append(np.nan)
            current = 0
            continue
        if value > 0:
            current = current + 1 if current > 0 else 1
        elif value < 0:
            current = current - 1 if current < 0 else -1
        else:
            current = 0
        streak.append(current)
    return pd.Series(streak, index=price.index)


def volume_vs_average(volume: pd.Series, window: int = VOLUME_AVERAGE_WINDOW) -> pd.Series:
    """Today's volume as a multiple of its own `window`-session average
    (1.5 means 50% above average). Volume is the provider's actual
    per-session traded share count and needs no split adjustment the way
    price does -- it already reflects whatever share count was outstanding
    that session."""
    return volume / volume.rolling(window=window, min_periods=window).mean()


def drawdown_from_peak(price: pd.Series) -> pd.Series:
    """Fractional distance below the running to-date peak (0 = at a new
    high), matching `backtesting.engine`'s `drawdown_fraction` convention
    (`(peak - value) / peak`, always >= 0) so the two are directly
    comparable."""
    peak = price.cummax()
    return (peak - price) / peak


def distance_from_rolling_high(price: pd.Series, window: int = TRAILING_YEAR_SESSIONS) -> pd.Series:
    """Fractional distance below the trailing `window`-session high (0 = at
    that high; -0.1 means 10% below it). Defaults to a trailing year (252
    trading sessions), the conventional "52-week high" window."""
    rolling_high = price.rolling(window=window, min_periods=window).max()
    return (price - rolling_high) / rolling_high


def distance_from_rolling_low(price: pd.Series, window: int = TRAILING_YEAR_SESSIONS) -> pd.Series:
    """Fractional distance above the trailing `window`-session low (0 = at
    that low; 0.1 means 10% above it). Defaults to a trailing year (252
    trading sessions), the conventional "52-week low" window."""
    rolling_low = price.rolling(window=window, min_periods=window).min()
    return (price - rolling_low) / rolling_low
