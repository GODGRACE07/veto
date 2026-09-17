"""
Deterministic Signal Module
============================
No LLM calls here. This module reads raw OHLCV price data and produces
reproducible, rule-based technical signals. It exists to give the
consensus scorer a ground-truth anchor that is NOT subject to LLM
hallucination -- see the design rationale in README.md.

Every function here is pure: same input -> same output, always.
That determinism is the whole point -- it's what makes the agent's
decisions auditable and reproducible instead of "trust the LLM".
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Literal
import pandas as pd
import numpy as np


TrendLabel = Literal["strong_downtrend", "downtrend", "sideways", "uptrend", "strong_uptrend"]


@dataclass
class DeterministicSignal:
    """
    A single, timestamped, reproducible read of market state.
    This is what the LLM's reasoning gets checked against.
    """
    ticker: str
    as_of: str  # ISO timestamp, UTC
    current_price: float
    ma5: float
    ma10: float
    ma20: float
    price_above_ma5: bool
    price_above_ma10: bool
    price_above_ma20: bool
    trend_score: float          # continuous score, roughly -3 to +3
    trend_label: TrendLabel
    volatility_20d: float       # annualized, realized vol over 20 trading days
    momentum_5d_pct: float      # 5-day rate of change, percent
    bullish_confidence: int     # 0-100, rule-based only
    bearish_confidence: int     # 0-100, rule-based only

    def to_dict(self) -> dict:
        return asdict(self)


def _moving_average(prices: pd.Series, window: int) -> float:
    if len(prices) < window:
        raise ValueError(
            f"Need at least {window} price points, got {len(prices)}. "
            f"Cannot compute a reliable moving average on insufficient history."
        )
    return float(prices.tail(window).mean())


def _trend_score(prices: pd.Series, window: int = 20) -> float:
    """
    Fits a simple linear regression over the trailing `window` prices.
    Returns a scalar combining direction and strength: normalized slope.
    This is deliberately simple and auditable -- not a black box.
    """
    if len(prices) < window:
        raise ValueError(f"Need at least {window} price points for trend scoring, got {len(prices)}.")
    y = prices.tail(window).to_numpy()
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    # Normalize slope by average price level so the score is comparable across tickers/price ranges
    avg_price = y.mean()
    if avg_price == 0:
        return 0.0
    normalized_slope = (slope * window) / avg_price  # total % move implied by the fitted line over the window
    return float(normalized_slope * 10)  # scale into a roughly -3..+3 range for typical moves


def _trend_label(score: float) -> TrendLabel:
    if score >= 1.5:
        return "strong_uptrend"
    elif score >= 0.4:
        return "uptrend"
    elif score > -0.4:
        return "sideways"
    elif score > -1.5:
        return "downtrend"
    else:
        return "strong_downtrend"


def _realized_volatility(prices: pd.Series, window: int = 20, trading_days_per_year: int = 252) -> float:
    if len(prices) < window + 1:
        raise ValueError(f"Need at least {window + 1} price points for volatility, got {len(prices)}.")
    returns = prices.tail(window + 1).pct_change().dropna()
    daily_std = returns.std()
    return float(daily_std * np.sqrt(trading_days_per_year))


def _momentum_pct(prices: pd.Series, window: int = 5) -> float:
    if len(prices) < window + 1:
        raise ValueError(f"Need at least {window + 1} price points for momentum, got {len(prices)}.")
    start = prices.iloc[-(window + 1)]
    end = prices.iloc[-1]
    if start == 0:
        return 0.0
    return float((end - start) / start * 100)


def compute_deterministic_signal(ticker: str, price_history: pd.DataFrame) -> DeterministicSignal:
    """
    price_history: DataFrame with at least columns ['date', 'close'], sorted ascending by date,
                   containing ONLY data available as-of the decision time (caller's responsibility
                   -- this function does not know "now", it trusts the caller not to leak future rows).

    Requires at least 21 rows (20 for MA20/vol + 1 for momentum lookback baseline).
    """
    if "close" not in price_history.columns:
        raise ValueError("price_history must contain a 'close' column")
    if len(price_history) < 21:
        raise ValueError(
            f"Insufficient history for {ticker}: got {len(price_history)} rows, need >= 21. "
            f"This is a hard requirement -- we do not compute signals on partial windows, "
            f"because a silently-degraded signal is worse than a loud failure."
        )

    prices = price_history["close"].reset_index(drop=True)
    current_price = float(prices.iloc[-1])

    ma5 = _moving_average(prices, 5)
    ma10 = _moving_average(prices, 10)
    ma20 = _moving_average(prices, 20)

    trend_score = _trend_score(prices, window=20)
    trend_label = _trend_label(trend_score)

    volatility_20d = _realized_volatility(prices, window=20)
    momentum_5d = _momentum_pct(prices, window=5)

    # Rule-based confidence scoring -- fully transparent, no learned weights.
    bullish = 50
    bearish = 50
    if current_price > ma5:
        bullish += 8
        bearish -= 8
    else:
        bearish += 8
        bullish -= 8
    if current_price > ma10:
        bullish += 6
        bearish -= 6
    else:
        bearish += 6
        bullish -= 6
    if current_price > ma20:
        bullish += 6
        bearish -= 6
    else:
        bearish += 6
        bullish -= 6
    bullish += int(np.clip(trend_score * 8, -20, 20))
    bearish -= int(np.clip(trend_score * 8, -20, 20))
    bullish += int(np.clip(momentum_5d, -15, 15))
    bearish -= int(np.clip(momentum_5d, -15, 15))

    bullish = int(np.clip(bullish, 0, 100))
    bearish = int(np.clip(bearish, 0, 100))

    return DeterministicSignal(
        ticker=ticker,
        as_of=datetime.now(timezone.utc).isoformat(),
        current_price=current_price,
        ma5=round(ma5, 4),
        ma10=round(ma10, 4),
        ma20=round(ma20, 4),
        price_above_ma5=current_price > ma5,
        price_above_ma10=current_price > ma10,
        price_above_ma20=current_price > ma20,
        trend_score=round(trend_score, 4),
        trend_label=trend_label,
        volatility_20d=round(volatility_20d, 4),
        momentum_5d_pct=round(momentum_5d, 4),
        bullish_confidence=bullish,
        bearish_confidence=bearish,
    )


def signal_agrees_with_direction(signal: DeterministicSignal, proposed_direction: Literal["buy", "sell", "hold"]) -> bool:
    """
    Checks whether the deterministic signal supports the LLM's proposed action.
    This is the concrete implementation of "does the LLM's stated reasoning
    match ground-truth price action" from the architecture design.

    Deliberately conservative: for 'buy', we require the trend to be at least
    'uptrend' (not merely 'sideways leaning up'). For 'sell', mirror that.
    'hold' always agrees, since choosing not to act cannot contradict price data.
    """
    if proposed_direction == "hold":
        return True
    if proposed_direction == "buy":
        return signal.trend_label in ("uptrend", "strong_uptrend") and signal.bullish_confidence >= 55
    if proposed_direction == "sell":
        return signal.trend_label in ("downtrend", "strong_downtrend") and signal.bearish_confidence >= 55
    raise ValueError(f"Unknown direction: {proposed_direction}")
