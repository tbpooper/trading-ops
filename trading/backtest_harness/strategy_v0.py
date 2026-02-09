"""Strategy v0 (baseline) — mirrors the current TradingView Pine logic.

This is intentionally simple and deterministic:
- Trend: EMA(fast) vs EMA(slow)
- Signal:
  - Long: trendUp and close < emaFast and RSI >= rsiMinLong
  - Short: trendDn and close > emaFast and RSI <= rsiMaxShort
- Stop: ATR * atrMult (from entry bar close)
- TP: RR * stop distance

Execution model:
- Enter at next bar open (simplification)
- Stops/targets checked using bar high/low
- If both TP and SL touched in same bar, we assume worst-case (SL first) to be conservative.

PnL model:
- Uses MNQ tick size/value: tick=0.25, tickValue=$0.50 per MNQ.

This module produces a sequence of closed trades that can be fed into Lucid rule simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

from trading.backtest_harness.tv_csv import Candle

Side = Literal["long", "short"]


@dataclass(frozen=True)
class StrategyParams:
    ema_fast: int = 20
    ema_slow: int = 50
    rsi_len: int = 14
    rsi_min_long: float = 45
    rsi_max_short: float = 55
    atr_len: int = 14
    atr_mult: float = 2.0
    rr: float = 1.5


@dataclass(frozen=True)
class Trade:
    side: Side
    entry_ts: int
    exit_ts: int
    entry: float
    exit: float
    sl: float
    tp: float
    pnl_dollars: float


TICK = 0.25
TICK_VALUE = 0.50  # $ per tick per MNQ


def ema(series: List[float], length: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(series)
    if length <= 0 or len(series) == 0:
        return out
    k = 2.0 / (length + 1.0)
    # seed with SMA
    if len(series) < length:
        return out
    sma = sum(series[:length]) / length
    out[length - 1] = sma
    prev = sma
    for i in range(length, len(series)):
        prev = (series[i] - prev) * k + prev
        out[i] = prev
    return out


def rsi(series: List[float], length: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(series)
    if length <= 0 or len(series) < length + 1:
        return out

    gains = []
    losses = []
    for i in range(1, length + 1):
        ch = series[i] - series[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))

    avg_gain = sum(gains) / length
    avg_loss = sum(losses) / length

    def rs_to_rsi(ag, al):
        if al == 0:
            return 100.0
        rs = ag / al
        return 100.0 - (100.0 / (1.0 + rs))

    out[length] = rs_to_rsi(avg_gain, avg_loss)

    for i in range(length + 1, len(series)):
        ch = series[i] - series[i - 1]
        gain = max(ch, 0.0)
        loss = max(-ch, 0.0)
        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length
        out[i] = rs_to_rsi(avg_gain, avg_loss)

    return out


def true_range(high: float, low: float, prev_close: float) -> float:
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def atr(candles: List[Candle], length: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(candles)
    if length <= 0 or len(candles) < length + 1:
        return out

    trs: List[float] = []
    for i in range(1, len(candles)):
        trs.append(true_range(candles[i].high, candles[i].low, candles[i - 1].close))

    # first ATR is SMA of first `length` TRs
    if len(trs) < length:
        return out
    first = sum(trs[:length]) / length
    out[length] = first  # aligns with candle index length
    prev = first
    for i in range(length + 1, len(candles)):
        tr = trs[i - 1]
        prev = (prev * (length - 1) + tr) / length
        out[i] = prev

    return out


def dollars_from_points(points: float, qty_mnq: int = 1) -> float:
    # points -> ticks
    ticks = points / TICK
    return ticks * TICK_VALUE * qty_mnq


def generate_trades(candles: List[Candle], p: StrategyParams, qty_mnq: int = 1) -> List[Trade]:
    closes = [c.close for c in candles]
    opens = [c.open for c in candles]

    ema_fast = ema(closes, p.ema_fast)
    ema_slow = ema(closes, p.ema_slow)
    r = rsi(closes, p.rsi_len)
    a = atr(candles, p.atr_len)

    trades: List[Trade] = []

    in_pos: Optional[Side] = None
    entry_i = -1
    entry_px = 0.0
    sl = 0.0
    tp = 0.0

    # iterate bars; we enter at next bar open, so signals evaluated at i-1
    for i in range(1, len(candles)):
        # manage open position on bar i
        if in_pos is not None:
            hi = candles[i].high
            lo = candles[i].low

            hit_sl = (lo <= sl) if in_pos == "long" else (hi >= sl)
            hit_tp = (hi >= tp) if in_pos == "long" else (lo <= tp)

            exit_px = None
            # conservative: if both touched, assume SL
            if hit_sl and hit_tp:
                exit_px = sl
            elif hit_sl:
                exit_px = sl
            elif hit_tp:
                exit_px = tp

            if exit_px is not None:
                pnl_points = (exit_px - entry_px) if in_pos == "long" else (entry_px - exit_px)
                pnl = dollars_from_points(pnl_points, qty_mnq=qty_mnq)
                trades.append(
                    Trade(
                        side=in_pos,
                        entry_ts=candles[entry_i].ts,
                        exit_ts=candles[i].ts,
                        entry=entry_px,
                        exit=exit_px,
                        sl=sl,
                        tp=tp,
                        pnl_dollars=pnl,
                    )
                )
                in_pos = None

        # if flat, check signal from previous bar (i-1) and enter at bar i open
        if in_pos is None:
            j = i - 1
            if ema_fast[j] is None or ema_slow[j] is None or r[j] is None or a[j] is None:
                continue

            trend_up = ema_fast[j] > ema_slow[j]
            trend_dn = ema_fast[j] < ema_slow[j]

            long_sig = trend_up and candles[j].close < ema_fast[j] and r[j] >= p.rsi_min_long
            short_sig = trend_dn and candles[j].close > ema_fast[j] and r[j] <= p.rsi_max_short

            if long_sig:
                in_pos = "long"
            elif short_sig:
                in_pos = "short"
            else:
                continue

            entry_i = i
            entry_px = opens[i]
            stop_dist = a[j] * p.atr_mult
            if in_pos == "long":
                sl = entry_px - stop_dist
                tp = entry_px + stop_dist * p.rr
            else:
                sl = entry_px + stop_dist
                tp = entry_px - stop_dist * p.rr

    return trades
