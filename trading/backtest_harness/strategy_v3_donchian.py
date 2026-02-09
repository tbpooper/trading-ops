"""Strategy v3 (alternate family): Donchian breakout + ATR stop + RR take-profit.

Goal: try a fundamentally different signal family than EMA/RSI pullback.

Signal:
- Compute Donchian channel over lookback N on bar j (exclude current bar):
  - upper = max(high[j-N:j])
  - lower = min(low[j-N:j])
- Long if close[j] > upper (breakout)
- Short if close[j] < lower

Risk:
- ATR stop: stop_dist = ATR * atr_mult
- TP: stop_dist * rr

Governors:
- max trades/day
- max losses/day
- daily loss stop

Session window enforced (wrap midnight supported).

Execution:
- signal on bar j, enter at next bar open (i)
- conservative SL-first if both touched

Returns closed trades.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

from trading.backtest_harness.tv_csv import Candle
from trading.backtest_harness.strategy_v0 import atr, dollars_from_points, TICK, TICK_VALUE

Side = Literal["long", "short"]


@dataclass(frozen=True)
class Session:
    enabled: bool = True
    start_min_utc: int = 12 * 60 + 30  # 7:30am ET
    end_min_utc: int = 3 * 60  # 10pm ET (wrap)


@dataclass(frozen=True)
class Governor:
    max_trades_per_day: Optional[int] = 3
    max_losses_per_day: Optional[int] = 2
    daily_loss_stop: Optional[float] = 250.0
    cooldown_bars_after_loss: int = 4


@dataclass(frozen=True)
class Params:
    donchian_len: int = 20
    atr_len: int = 14
    atr_mult: float = 1.5
    rr: float = 1.5

    risk_per_trade_dollars: float = 200.0
    max_micros: int = 20

    session: Session = Session()
    governor: Governor = Governor()


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
    import datetime

    d = datetime.datetime.utcfromtimestamp(ts)
    return d.hour * 60 + d.minute


def day_key_utc(ts: int) -> str:
    import datetime

    return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")


def in_session(ts: int, s: Session) -> bool:
    if not s.enabled:
        return True
    m = minutes_utc(ts)
    if s.start_min_utc <= s.end_min_utc:
        return s.start_min_utc <= m <= s.end_min_utc
    return m >= s.start_min_utc or m <= s.end_min_utc


def calc_qty_from_stop(stop_dist_points: float, risk_per_trade_dollars: float, max_micros: int) -> int:
    stop_ticks = max(1.0, stop_dist_points / TICK)
    qty = int(risk_per_trade_dollars // (stop_ticks * TICK_VALUE))
    qty = max(1, qty)
    qty = min(max_micros, qty)
    return qty


def generate_trades(candles: List[Candle], p: Params) -> List[Trade]:
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    closes = [c.close for c in candles]
    opens = [c.open for c in candles]

    a = atr(candles, p.atr_len)

    trades: List[Trade] = []

    in_pos: Optional[Side] = None
    qty = 0
    entry_i = -1
    entry_px = 0.0
    sl = 0.0
    tp = 0.0

    cur_day = None
    trades_today = 0
    losses_today = 0
    day_pnl = 0.0
    cooldown = 0
    stop_for_day = False

    for i in range(1, len(candles)):
        d = day_key_utc(candles[i].ts)
        if cur_day != d:
            cur_day = d
            trades_today = 0
            losses_today = 0
            day_pnl = 0.0
            cooldown = 0
            stop_for_day = False

        if cooldown > 0:
            cooldown -= 1

        # manage position
        if in_pos is not None:
            hi = highs[i]
            lo = lows[i]
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

                day_pnl += pnl
                if pnl < 0:
                    losses_today += 1
                    cooldown = max(cooldown, p.governor.cooldown_bars_after_loss)
                    if p.governor.max_losses_per_day is not None and losses_today >= p.governor.max_losses_per_day:
                        stop_for_day = True

                if p.governor.daily_loss_stop is not None and day_pnl <= -abs(p.governor.daily_loss_stop):
                    stop_for_day = True

                in_pos = None

        # entry
        if in_pos is None:
            if stop_for_day or cooldown > 0:
                continue
            if p.governor.max_trades_per_day is not None and trades_today >= p.governor.max_trades_per_day:
                continue

            j = i - 1
            if a[j] is None:
                continue
            if not in_session(candles[j].ts, p.session):
                continue
            if j - p.donchian_len < 1:
                continue

            upper = max(highs[j - p.donchian_len : j])
            lower = min(lows[j - p.donchian_len : j])

            long_sig = closes[j] > upper
            short_sig = closes[j] < lower

            if not (long_sig or short_sig):
                continue

            in_pos = "long" if long_sig else "short"
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

            trades_today += 1

    return trades
