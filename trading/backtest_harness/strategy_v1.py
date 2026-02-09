"""Strategy v1 — v0 + risk-based sizing + optional filters.

Adds:
- Risk-based MNQ sizing (micros) from stop distance, capped at max_micros.
- Optional session/time filter (UTC minutes)
- Optional volatility filter (ATR min)

Execution model unchanged:
- signal on bar j, enter at next bar open (i)
- SL/TP checked on bar high/low; if both touched -> SL first (conservative)

IMPORTANT: this is still a simplified fills model and uses UTC day boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from trading.backtest_harness.tv_csv import Candle
from trading.backtest_harness.strategy_v0 import ema, rsi, atr, dollars_from_points, TICK, TICK_VALUE

Side = Literal["long", "short"]


@dataclass(frozen=True)
class Filters:
    # Session filter in UTC minutes-of-day (0..1439). If None => no filter.
    session_start_min_utc: Optional[int] = None
    session_end_min_utc: Optional[int] = None

    # Volatility filter: require ATR (points) >= atr_min_points
    atr_min_points: Optional[float] = None


@dataclass(frozen=True)
class Params:
    ema_fast: int = 20
    ema_slow: int = 50
    rsi_len: int = 14
    rsi_min_long: float = 45
    rsi_max_short: float = 55
    atr_len: int = 14
    atr_mult: float = 2.0
    rr: float = 1.5

    # risk/sizing
    risk_per_trade_dollars: float = 100.0
    max_micros: int = 20

    filters: Filters = Filters()


@dataclass(frozen=True)
class Trade:
    side: Side
    qty: int
    entry_ts: int
    exit_ts: int
    entry: float
    exit: float
    sl: float
    tp: float
    pnl_dollars: float


def minutes_utc(ts: int) -> int:
    # ts is unix seconds UTC
    import datetime

    d = datetime.datetime.utcfromtimestamp(ts)
    return d.hour * 60 + d.minute


def in_session(ts: int, f: Filters) -> bool:
    if f.session_start_min_utc is None or f.session_end_min_utc is None:
        return True
    m = minutes_utc(ts)
    if f.session_start_min_utc <= f.session_end_min_utc:
        return f.session_start_min_utc <= m <= f.session_end_min_utc
    # wrap midnight
    return m >= f.session_start_min_utc or m <= f.session_end_min_utc


def calc_qty_from_stop(stop_dist_points: float, risk_per_trade_dollars: float, max_micros: int) -> int:
    stop_ticks = max(1.0, stop_dist_points / TICK)
    qty = int(risk_per_trade_dollars // (stop_ticks * TICK_VALUE))
    if qty < 1:
        qty = 1
    if qty > max_micros:
        qty = max_micros
    return qty


def generate_trades(candles: List[Candle], p: Params) -> List[Trade]:
    closes = [c.close for c in candles]
    opens = [c.open for c in candles]

    ema_fast = ema(closes, p.ema_fast)
    ema_slow = ema(closes, p.ema_slow)
    r = rsi(closes, p.rsi_len)
    a = atr(candles, p.atr_len)

    trades: List[Trade] = []

    in_pos: Optional[Side] = None
    qty = 0
    entry_i = -1
    entry_px = 0.0
    sl = 0.0
    tp = 0.0

    for i in range(1, len(candles)):
        # manage open position on bar i
        if in_pos is not None:
            hi = candles[i].high
            lo = candles[i].low

            hit_sl = (lo <= sl) if in_pos == "long" else (hi >= sl)
            hit_tp = (hi >= tp) if in_pos == "long" else (lo <= tp)

            exit_px = None
            if hit_sl and hit_tp:
                exit_px = sl
            elif hit_sl:
                exit_px = sl
            elif hit_tp:
                exit_px = tp

            if exit_px is not None:
                pnl_points = (exit_px - entry_px) if in_pos == "long" else (entry_px - exit_px)
                pnl = dollars_from_points(pnl_points, qty_mnq=qty)
                trades.append(
                    Trade(
                        side=in_pos,
                        qty=qty,
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

        # if flat, check signal on j=i-1 and enter at i open
        if in_pos is None:
            j = i - 1
            if ema_fast[j] is None or ema_slow[j] is None or r[j] is None or a[j] is None:
                continue

            # filters on signal bar
            if not in_session(candles[j].ts, p.filters):
                continue
            if p.filters.atr_min_points is not None and a[j] < p.filters.atr_min_points:
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
            qty = calc_qty_from_stop(stop_dist, p.risk_per_trade_dollars, p.max_micros)

            if in_pos == "long":
                sl = entry_px - stop_dist
                tp = entry_px + stop_dist * p.rr
            else:
                sl = entry_px + stop_dist
                tp = entry_px - stop_dist * p.rr

    return trades
