"""
Trading strategies for KOSPI backtesting.

Each strategy returns a signal series:
  1 = Buy signal
 -1 = Sell signal
  0 = Hold/No signal
"""

from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

# ============================================================================
# MOMENTUM STRATEGIES
# ============================================================================


def momentum_simple(lookback: int = 20) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Simple momentum: Buy when price > price[lookback days ago].

    Args:
        lookback: Lookback period in days
    """

    def signal_func(data: pd.DataFrame) -> pd.Series:
        close = data["Close"]
        momentum = close / close.shift(lookback) - 1
        signal = pd.Series(0, index=data.index)
        signal[momentum > 0] = 1
        signal[momentum < 0] = -1
        return signal

    return signal_func


def momentum_dual(
    fast: int = 10, slow: int = 30
) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Dual momentum: Buy when fast momentum > slow momentum.
    """

    def signal_func(data: pd.DataFrame) -> pd.Series:
        close = data["Close"]
        fast_mom = close / close.shift(fast) - 1
        slow_mom = close / close.shift(slow) - 1
        signal = pd.Series(0, index=data.index)
        signal[(fast_mom > 0) & (fast_mom > slow_mom)] = 1
        signal[(fast_mom < 0) & (fast_mom < slow_mom)] = -1
        return signal

    return signal_func


def momentum_12_1(
    skip_recent: int = 21, lookback: int = 252
) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Classic 12-1 momentum: 12-month momentum skipping most recent month.
    """

    def signal_func(data: pd.DataFrame) -> pd.Series:
        close = data["Close"]
        # Return from lookback to skip_recent days ago
        past_price = close.shift(lookback)
        recent_price = close.shift(skip_recent)
        momentum = (recent_price / past_price) - 1
        signal = pd.Series(0, index=data.index)
        signal[momentum > 0] = 1
        signal[momentum < 0] = -1
        return signal

    return signal_func


# ============================================================================
# MOVING AVERAGE STRATEGIES
# ============================================================================


def ma_crossover(fast: int = 5, slow: int = 20) -> Callable[[pd.DataFrame], pd.Series]:
    """
    MA crossover: Buy when fast MA > slow MA.
    """

    def signal_func(data: pd.DataFrame) -> pd.Series:
        close = data["Close"]
        fast_ma = close.rolling(fast).mean()
        slow_ma = close.rolling(slow).mean()

        signal = pd.Series(0, index=data.index)
        # Buy when fast crosses above slow
        signal[(fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))] = 1
        # Sell when fast crosses below slow
        signal[(fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))] = -1

        return signal

    return signal_func


def ma_trend(period: int = 20) -> Callable[[pd.DataFrame], pd.Series]:
    """
    MA trend: Stay long when price > MA, else flat.
    """

    def signal_func(data: pd.DataFrame) -> pd.Series:
        close = data["Close"]
        ma = close.rolling(period).mean()

        signal = pd.Series(0, index=data.index)
        signal[close > ma] = 1
        signal[close < ma] = -1

        return signal

    return signal_func


def triple_ma(
    fast: int = 5, mid: int = 20, slow: int = 60
) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Triple MA: Buy when fast > mid > slow, sell when fast < mid < slow.
    """

    def signal_func(data: pd.DataFrame) -> pd.Series:
        close = data["Close"]
        fast_ma = close.rolling(fast).mean()
        mid_ma = close.rolling(mid).mean()
        slow_ma = close.rolling(slow).mean()

        signal = pd.Series(0, index=data.index)
        signal[(fast_ma > mid_ma) & (mid_ma > slow_ma)] = 1
        signal[(fast_ma < mid_ma) & (mid_ma < slow_ma)] = -1

        return signal

    return signal_func


def golden_death_cross(
    fast: int = 50, slow: int = 200
) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Golden/Death cross: 50-day MA vs 200-day MA.
    """
    return ma_crossover(fast, slow)


# ============================================================================
# MEAN REVERSION STRATEGIES
# ============================================================================


def rsi_reversal(
    period: int = 14, oversold: int = 30, overbought: int = 70
) -> Callable[[pd.DataFrame], pd.Series]:
    """
    RSI reversal: Buy when RSI < oversold, sell when RSI > overbought.
    """

    def signal_func(data: pd.DataFrame) -> pd.Series:
        close = data["Close"]
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        signal = pd.Series(0, index=data.index)
        signal[rsi < oversold] = 1
        signal[rsi > overbought] = -1

        return signal

    return signal_func


def bollinger_bands(
    period: int = 20, std_dev: float = 2.0
) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Bollinger Bands: Buy at lower band, sell at upper band.
    """

    def signal_func(data: pd.DataFrame) -> pd.Series:
        close = data["Close"]
        ma = close.rolling(period).mean()
        std = close.rolling(period).std()

        upper = ma + std_dev * std
        lower = ma - std_dev * std

        signal = pd.Series(0, index=data.index)
        signal[close <= lower] = 1
        signal[close >= upper] = -1

        return signal

    return signal_func


def mean_reversion_zscore(
    period: int = 20, entry_z: float = -2.0, exit_z: float = 0
) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Z-score mean reversion: Enter when z-score below threshold.
    """

    def signal_func(data: pd.DataFrame) -> pd.Series:
        close = data["Close"]
        ma = close.rolling(period).mean()
        std = close.rolling(period).std()
        zscore = (close - ma) / std

        signal = pd.Series(0, index=data.index)
        # Buy when z-score is very negative
        signal[zscore <= entry_z] = 1
        # Sell when z-score returns to normal
        signal[zscore >= -entry_z] = -1

        return signal

    return signal_func


# ============================================================================
# BREAKOUT STRATEGIES
# ============================================================================


def donchian_channel(period: int = 20) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Donchian channel breakout: Buy at new highs, sell at new lows.
    """

    def signal_func(data: pd.DataFrame) -> pd.Series:
        high = data["High"]
        low = data["Low"]
        close = data["Close"]

        upper = high.rolling(period).max()
        lower = low.rolling(period).min()

        signal = pd.Series(0, index=data.index)
        # Buy on upper breakout
        signal[close >= upper.shift(1)] = 1
        # Sell on lower breakdown
        signal[close <= lower.shift(1)] = -1

        return signal

    return signal_func


def range_breakout(
    lookback: int = 20, breakout_pct: float = 0.02
) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Range breakout: Buy when price breaks above range by certain percentage.
    """

    def signal_func(data: pd.DataFrame) -> pd.Series:
        high = data["High"]
        low = data["Low"]
        close = data["Close"]

        range_high = high.rolling(lookback).max()
        range_low = low.rolling(lookback).min()

        signal = pd.Series(0, index=data.index)
        # Buy on breakout above range
        signal[close > range_high.shift(1) * (1 + breakout_pct)] = 1
        # Sell on breakdown below range
        signal[close < range_low.shift(1) * (1 - breakout_pct)] = -1

        return signal

    return signal_func


def volume_breakout(
    price_period: int = 20, volume_mult: float = 2.0
) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Volume breakout: Buy on high volume price breakout.
    """

    def signal_func(data: pd.DataFrame) -> pd.Series:
        close = data["Close"]
        volume = data["Volume"]
        high = data["High"]

        price_high = high.rolling(price_period).max()
        avg_volume = volume.rolling(price_period).mean()

        signal = pd.Series(0, index=data.index)
        # Buy on price breakout with high volume
        signal[(close > price_high.shift(1)) & (volume > avg_volume * volume_mult)] = 1
        # Sell signal when volume drops
        signal[(volume < avg_volume * 0.5)] = -1

        return signal

    return signal_func


# ============================================================================
# TREND FOLLOWING STRATEGIES
# ============================================================================


def supertrend_strategy(
    atr_period: int = 10, multiplier: float = 3.0
) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Supertrend indicator strategy.
    """

    def signal_func(data: pd.DataFrame) -> pd.Series:
        high = data["High"]
        low = data["Low"]
        close = data["Close"]

        # Calculate ATR
        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(atr_period).mean()

        # Basic bands
        hl2 = (high + low) / 2
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr

        # Final bands with trailing logic
        final_upper = upper_band.copy()
        final_lower = lower_band.copy()
        supertrend = pd.Series(index=data.index, dtype=float)
        direction = pd.Series(1, index=data.index)

        for i in range(1, len(data)):
            # Upper band
            if (
                upper_band.iloc[i] < final_upper.iloc[i - 1]
                or close.iloc[i - 1] > final_upper.iloc[i - 1]
            ):
                final_upper.iloc[i] = upper_band.iloc[i]
            else:
                final_upper.iloc[i] = final_upper.iloc[i - 1]

            # Lower band
            if (
                lower_band.iloc[i] > final_lower.iloc[i - 1]
                or close.iloc[i - 1] < final_lower.iloc[i - 1]
            ):
                final_lower.iloc[i] = lower_band.iloc[i]
            else:
                final_lower.iloc[i] = final_lower.iloc[i - 1]

            # Direction and supertrend
            if i == 0:
                supertrend.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
            elif supertrend.iloc[i - 1] == final_upper.iloc[i - 1]:
                if close.iloc[i] > final_upper.iloc[i]:
                    supertrend.iloc[i] = final_lower.iloc[i]
                    direction.iloc[i] = 1
                else:
                    supertrend.iloc[i] = final_upper.iloc[i]
                    direction.iloc[i] = -1
            else:
                if close.iloc[i] < final_lower.iloc[i]:
                    supertrend.iloc[i] = final_upper.iloc[i]
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = final_lower.iloc[i]
                    direction.iloc[i] = 1

        signal = pd.Series(0, index=data.index)
        signal[(direction == 1) & (direction.shift(1) == -1)] = 1
        signal[(direction == -1) & (direction.shift(1) == 1)] = -1

        return signal

    return signal_func


def adx_trend(
    period: int = 14, threshold: int = 25
) -> Callable[[pd.DataFrame], pd.Series]:
    """
    ADX trend following: Trade only when ADX > threshold.
    """

    def signal_func(data: pd.DataFrame) -> pd.Series:
        high = data["High"]
        low = data["Low"]
        close = data["Close"]

        # +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        plus_dm[(plus_dm < minus_dm)] = 0
        minus_dm[(minus_dm < plus_dm)] = 0

        # ATR
        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(period).mean()

        # +DI and -DI
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        # DX and ADX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx = dx.rolling(period).mean()

        signal = pd.Series(0, index=data.index)
        # Buy when +DI > -DI and ADX is strong
        signal[(plus_di > minus_di) & (adx > threshold)] = 1
        signal[(plus_di < minus_di) & (adx > threshold)] = -1

        return signal

    return signal_func


# ============================================================================
# VOLATILITY STRATEGIES
# ============================================================================


def volatility_contraction(
    lookback: int = 20, threshold: float = 0.5
) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Volatility contraction: Enter when volatility contracts (squeeze).
    """

    def signal_func(data: pd.DataFrame) -> pd.Series:
        close = data["Close"]
        returns = close.pct_change()
        volatility = returns.rolling(lookback).std()
        vol_percentile = volatility.rolling(252).apply(
            lambda x: (
                (x.iloc[-1] - x.min()) / (x.max() - x.min())
                if x.max() != x.min()
                else 0.5
            )
        )

        ma = close.rolling(lookback).mean()

        signal = pd.Series(0, index=data.index)
        # Buy when volatility is low and price above MA
        signal[(vol_percentile < threshold) & (close > ma)] = 1
        signal[(vol_percentile < threshold) & (close < ma)] = -1

        return signal

    return signal_func


def atr_trailing_stop(
    atr_period: int = 14, multiplier: float = 2.0
) -> Callable[[pd.DataFrame], pd.Series]:
    """
    ATR trailing stop: Use ATR for stop placement.
    """

    def signal_func(data: pd.DataFrame) -> pd.Series:
        high = data["High"]
        low = data["Low"]
        close = data["Close"]

        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(atr_period).mean()

        # Trailing stop
        stop_long = close - multiplier * atr
        stop_short = close + multiplier * atr

        signal = pd.Series(0, index=data.index)
        # Simple trend following with ATR stop
        signal[close > close.rolling(20).mean()] = 1
        signal[close < close.rolling(20).mean()] = -1

        return signal

    return signal_func


# ============================================================================
# VOLUME-BASED STRATEGIES
# ============================================================================


def obv_trend(period: int = 20) -> Callable[[pd.DataFrame], pd.Series]:
    """
    OBV (On Balance Volume) trend following.
    """

    def signal_func(data: pd.DataFrame) -> pd.Series:
        close = data["Close"]
        volume = data["Volume"]

        # OBV calculation
        obv = (volume * np.sign(close.diff())).cumsum()
        obv_ma = obv.rolling(period).mean()

        signal = pd.Series(0, index=data.index)
        signal[(obv > obv_ma) & (close > close.rolling(period).mean())] = 1
        signal[(obv < obv_ma) & (close < close.rolling(period).mean())] = -1

        return signal

    return signal_func


def vwap_reversion(period: int = 20) -> Callable[[pd.DataFrame], pd.Series]:
    """
    VWAP mean reversion strategy.
    """

    def signal_func(data: pd.DataFrame) -> pd.Series:
        close = data["Close"]
        volume = data["Volume"]
        high = data["High"]
        low = data["Low"]

        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).rolling(period).sum() / volume.rolling(
            period
        ).sum()

        deviation = (close - vwap) / vwap

        signal = pd.Series(0, index=data.index)
        signal[deviation < -0.02] = 1  # Buy when below VWAP
        signal[deviation > 0.02] = -1  # Sell when above VWAP

        return signal

    return signal_func


# ============================================================================
# PATTERN-BASED STRATEGIES
# ============================================================================


def gap_fade(min_gap: float = 0.02) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Gap fade: Trade against opening gaps.
    """

    def signal_func(data: pd.DataFrame) -> pd.Series:
        open_price = data["Open"]
        close = data["Close"]

        gap = (open_price / close.shift(1)) - 1

        signal = pd.Series(0, index=data.index)
        # Fade gap up (sell)
        signal[gap > min_gap] = -1
        # Fade gap down (buy)
        signal[gap < -min_gap] = 1

        return signal

    return signal_func


def inside_day_breakout() -> Callable[[pd.DataFrame], pd.Series]:
    """
    Inside day breakout: Trade breakouts from narrow range days.
    """

    def signal_func(data: pd.DataFrame) -> pd.Series:
        high = data["High"]
        low = data["Low"]
        close = data["Close"]

        # Inside day: today's range within yesterday's range
        inside_day = (high < high.shift(1)) & (low > low.shift(1))

        signal = pd.Series(0, index=data.index)
        # Buy on breakout above inside day high
        signal[(inside_day.shift(1)) & (close > high.shift(1))] = 1
        # Sell on breakdown below inside day low
        signal[(inside_day.shift(1)) & (close < low.shift(1))] = -1

        return signal

    return signal_func


def nr7_breakout() -> Callable[[pd.DataFrame], pd.Series]:
    """
    NR7 (Narrowest Range of 7 days) breakout.
    """

    def signal_func(data: pd.DataFrame) -> pd.Series:
        high = data["High"]
        low = data["Low"]
        close = data["Close"]

        daily_range = high - low
        min_range_7 = daily_range.rolling(7).min()

        # NR7 day: smallest range in last 7 days
        nr7 = daily_range == min_range_7

        signal = pd.Series(0, index=data.index)
        # Buy on breakout above NR7 high
        signal[(nr7.shift(1)) & (close > high.shift(1))] = 1
        # Sell on breakdown below NR7 low
        signal[(nr7.shift(1)) & (close < low.shift(1))] = -1

        return signal

    return signal_func


# ============================================================================
# COMBINED STRATEGIES
# ============================================================================


def macd_rsi_combo(
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    rsi_period: int = 14,
    rsi_oversold: int = 30,
    rsi_overbought: int = 70,
) -> Callable[[pd.DataFrame], pd.Series]:
    """
    MACD + RSI combination strategy.
    """

    def signal_func(data: pd.DataFrame) -> pd.Series:
        close = data["Close"]

        # MACD
        fast_ema = close.ewm(span=macd_fast).mean()
        slow_ema = close.ewm(span=macd_slow).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=macd_signal).mean()
        macd_hist = macd_line - signal_line

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(rsi_period).mean()
        avg_loss = loss.rolling(rsi_period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        signal = pd.Series(0, index=data.index)
        # Buy: MACD bullish crossover + RSI not overbought
        signal[(macd_hist > 0) & (macd_hist.shift(1) < 0) & (rsi < rsi_overbought)] = 1
        # Sell: MACD bearish crossover + RSI not oversold
        signal[(macd_hist < 0) & (macd_hist.shift(1) > 0) & (rsi > rsi_oversold)] = -1

        return signal

    return signal_func


def trend_momentum_filter(
    trend_period: int = 200, momentum_period: int = 20
) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Trade momentum only in direction of long-term trend.
    """

    def signal_func(data: pd.DataFrame) -> pd.Series:
        close = data["Close"]

        # Long-term trend
        trend_ma = close.rolling(trend_period).mean()
        uptrend = close > trend_ma

        # Momentum
        momentum = close / close.shift(momentum_period) - 1
        mom_positive = momentum > 0

        signal = pd.Series(0, index=data.index)
        # Buy: uptrend + positive momentum
        signal[uptrend & mom_positive] = 1
        # Sell: downtrend + negative momentum
        signal[~uptrend & ~mom_positive] = -1

        return signal

    return signal_func


# ============================================================================
# STRATEGY REGISTRY
# ============================================================================


def get_all_strategies() -> Dict[str, Tuple[Callable, Dict]]:
    """
    Get all available strategies with their default parameters.

    Returns:
        Dict of strategy_name -> (strategy_factory, default_params)
    """
    strategies = {
        # Momentum
        "momentum_20d": (momentum_simple, {"lookback": 20}),
        "momentum_60d": (momentum_simple, {"lookback": 60}),
        "momentum_120d": (momentum_simple, {"lookback": 120}),
        "momentum_dual_10_30": (momentum_dual, {"fast": 10, "slow": 30}),
        "momentum_dual_20_60": (momentum_dual, {"fast": 20, "slow": 60}),
        "momentum_12_1": (momentum_12_1, {"skip_recent": 21, "lookback": 252}),
        # Moving Average
        "ma_cross_5_20": (ma_crossover, {"fast": 5, "slow": 20}),
        "ma_cross_10_50": (ma_crossover, {"fast": 10, "slow": 50}),
        "ma_cross_20_60": (ma_crossover, {"fast": 20, "slow": 60}),
        "golden_cross": (golden_death_cross, {"fast": 50, "slow": 200}),
        "ma_trend_20": (ma_trend, {"period": 20}),
        "ma_trend_50": (ma_trend, {"period": 50}),
        "triple_ma": (triple_ma, {"fast": 5, "mid": 20, "slow": 60}),
        # Mean Reversion
        "rsi_14": (rsi_reversal, {"period": 14, "oversold": 30, "overbought": 70}),
        "rsi_14_extreme": (
            rsi_reversal,
            {"period": 14, "oversold": 20, "overbought": 80},
        ),
        "bollinger_20_2": (bollinger_bands, {"period": 20, "std_dev": 2.0}),
        "bollinger_20_2.5": (bollinger_bands, {"period": 20, "std_dev": 2.5}),
        "zscore_reversion": (mean_reversion_zscore, {"period": 20, "entry_z": -2.0}),
        # Breakout
        "donchian_20": (donchian_channel, {"period": 20}),
        "donchian_55": (donchian_channel, {"period": 55}),
        "range_breakout": (range_breakout, {"lookback": 20, "breakout_pct": 0.02}),
        "volume_breakout": (volume_breakout, {"price_period": 20, "volume_mult": 2.0}),
        # Trend Following
        "supertrend_10_3": (supertrend_strategy, {"atr_period": 10, "multiplier": 3.0}),
        "supertrend_14_2": (supertrend_strategy, {"atr_period": 14, "multiplier": 2.0}),
        "adx_trend": (adx_trend, {"period": 14, "threshold": 25}),
        "atr_trailing": (atr_trailing_stop, {"atr_period": 14, "multiplier": 2.0}),
        # Volatility
        "vol_contraction": (volatility_contraction, {"lookback": 20, "threshold": 0.3}),
        # Volume
        "obv_trend": (obv_trend, {"period": 20}),
        "vwap_reversion": (vwap_reversion, {"period": 20}),
        # Pattern
        "gap_fade": (gap_fade, {"min_gap": 0.02}),
        "inside_day": (inside_day_breakout, {}),
        "nr7_breakout": (nr7_breakout, {}),
        # Combined
        "macd_rsi": (macd_rsi_combo, {}),
        "trend_momentum": (
            trend_momentum_filter,
            {"trend_period": 200, "momentum_period": 20},
        ),
    }

    return strategies


def create_strategy(
    name: str, params: Optional[Dict] = None
) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Create a strategy by name.

    Args:
        name: Strategy name from get_all_strategies()
        params: Optional parameter overrides

    Returns:
        Signal function
    """
    strategies = get_all_strategies()

    if name not in strategies:
        raise ValueError(
            f"Unknown strategy: {name}. Available: {list(strategies.keys())}"
        )

    factory, default_params = strategies[name]

    # Merge parameters
    final_params = {**default_params}
    if params:
        final_params.update(params)

    return factory(**final_params)
