"""
Data Client
===========
Fetches real historical price data for the deterministic signal
module. Uses `yfinance` (free, no API key required) as the price
source for tokenized-stock underlyings.

CRITICAL DESIGN CONSTRAINT: every function here takes an explicit
`as_of` cutoff and guarantees it never returns a row dated after that
cutoff. This is the data-layer half of preventing look-ahead bias --
signals.py assumes whatever DataFrame it's given contains no future
information, and this is the module responsible for upholding that
guarantee when pulling from a live source.

Requires: pip install yfinance
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
import pandas as pd


class DataClientError(Exception):
    pass


def fetch_price_history(
    ticker: str,
    as_of: datetime,
    lookback_days: int = 60,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns ['date', 'close'], sorted ascending,
    containing ONLY rows with date <= as_of. Fetches extra lookback
    buffer (lookback_days) to comfortably cover the 21-row minimum the
    signal module requires even after weekends/holidays are excluded.

    Raises DataClientError if fewer than 21 usable rows remain after
    filtering to as_of -- this mirrors signals.py's own hard minimum,
    so the failure surfaces here with a clear, ticker-specific message
    rather than as an opaque error two layers deeper.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise DataClientError(
            "The 'yfinance' package is not installed. Run: pip install yfinance --break-system-packages"
        )

    start = as_of - timedelta(days=lookback_days)
    # yfinance's `end` parameter is exclusive of the end date itself in some
    # versions -- add one day and then explicitly filter below, so the
    # as_of cutoff enforcement doesn't silently depend on yfinance's own
    # (undocumented-in-places) date boundary behavior.
    end = as_of + timedelta(days=1)

    try:
        raw = yf.download(ticker, start=start.date(), end=end.date(), progress=False, auto_adjust=True)
    except Exception as e:
        raise DataClientError(f"yfinance download failed for {ticker}: {e}")

    if raw.empty:
        raise DataClientError(
            f"yfinance returned no data for {ticker} between {start.date()} and {end.date()}. "
            f"Check the ticker symbol and date range."
        )

    raw = raw.reset_index()
    # yfinance sometimes returns MultiIndex columns for single-ticker downloads
    # depending on version -- normalize defensively rather than assuming a shape.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]

    df = raw[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
    df["date"] = pd.to_datetime(df["date"])

    # Explicit, unconditional enforcement of the as_of cutoff -- this is the
    # actual leakage guard, independent of whatever yfinance's date-range
    # semantics happen to be.
    as_of_naive = pd.Timestamp(as_of).tz_localize(None) if as_of.tzinfo else pd.Timestamp(as_of)
    df["date"] = df["date"].dt.tz_localize(None)
    df = df[df["date"] <= as_of_naive].sort_values("date").reset_index(drop=True)

    if len(df) < 21:
        raise DataClientError(
            f"Only {len(df)} usable rows for {ticker} as of {as_of.date()} after enforcing the "
            f"cutoff -- need at least 21. Try a larger lookback_days value."
        )

    return df[["date", "close"]]


def summarize_signal_for_prompt(signal) -> str:
    """
    Formats a DeterministicSignal into a short text block for inclusion
    in an LLM prompt. Kept separate from the DeterministicSignal class
    itself so the prompt format can be iterated on without touching the
    signal computation logic.
    """
    return (
        f"Current price: {signal.current_price:.2f}\n"
        f"5-day MA: {signal.ma5:.2f} | 10-day MA: {signal.ma10:.2f} | 20-day MA: {signal.ma20:.2f}\n"
        f"Trend: {signal.trend_label} (score: {signal.trend_score:.2f})\n"
        f"20-day realized volatility (annualized): {signal.volatility_20d:.2%}\n"
        f"5-day momentum: {signal.momentum_5d_pct:+.2f}%\n"
        f"Rule-based bullish confidence: {signal.bullish_confidence}/100 | "
        f"bearish confidence: {signal.bearish_confidence}/100"
    )
