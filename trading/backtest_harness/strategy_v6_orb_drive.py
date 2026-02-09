"""Strategy v6: ORB / open-drive playbook (single-order simple).

Purpose: try a different edge family than v2/v5 by concentrating on the opening-range
breakout periods where MNQ can trend and reach targets quickly.

Mechanics (5m bars, UTC):
- Session window: user allowed 7:30am–10pm ET ~= 12:30–03:00 UTC (wrap)
- Define ORB from session_start for `orb_minutes` (e.g., 30 minutes).
- After ORB is formed:
  - Long if close breaks above ORB high + buffer
  - Short if close breaks below ORB low - buffer
- Optional trend filter: EMA fast vs slow + spread threshold.

Execution:
- signal on bar j, enter next bar open (i)

Exits:
- ATR stop
- trailing stop (ATR)
- time stop (max_hold_bars)
- optional TP cap (ATR)

Governors:
- max trades/day
- max losses/day
- daily loss stop

No partials.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

from trading.backtest_harness.tv_csv import Candle
from trading.backtest_harness.strategy_v0 import ema, atr, dollars_from_points, TICK, TICK_VALUE

Side = Literal["long", "short"]


def minutes_utc(ts: int) -> int:
    import datetime

    d = datetime.datetime.utcfromtimestamp(ts)
    return d.hour * 60 + d.minute


def day_key_utc(ts: int) -> str:
    import datetime

    return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")


def in_session(ts: int, start_min_utc: int, end_min_utc: int) -> bool:
    m = minutes_utc(ts)
    if start_min_utc <= end_min_utc:
        return start_min_utc <= m <= end_min_utc
    return m >= start_min_utc or m <= end_min_utc


@dataclass(frozen=True)
class TrendFilter:
    enabled: bool = True
    ema_fast: int = 10
    ema_slow: int = 40
    min_spread_points: float = 2.0


@dataclass(frozen=True)
class Exits:
    atr_len: int = 14
    stop_atr_mult: float = 1.0
    trail_atr_mult: float = 0.75
    tp_atr_mult: Optional[float] = 3.0
    max_hold_bars: int = 72  # 6 hours on 5m


@dataclass(frozen=True)
class Governor:
    max_trades_per_day: int = 1
    max_losses_per_day: int = 1
    daily_loss_stop: float = 250.0
    cooldown_bars_after_loss: int = 12


@dataclass(frozen=True)
class Params:
    # trading window
    session_start_min_utc: int = 12 * 60 + 30
    session_end_min_utc: int = 3 * 60

    # ORB
    orb_minutes: int = 30
    buffer_points: float = 1.0

    # risk/sizing
    risk_per_trade_dollars: float = 200.0
    max_micros: int = 20

    trend: TrendFilter = TrendFilter()
    exits: Exits = Exits()
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

    ef = ema(closes, p.trend.ema_fast)
    es = ema(closes, p.trend.ema_slow)
    ax = atr(candles, p.exits.atr_len)

    trades: List[Trade] = []

    in_pos: Optional[Side] = None
    qty = 0
    entry_i = -1
    entry_px = 0.0
    sl = 0.0
    tp = 0.0
    trail = 0.0

    cur_day = None
    trades_today = 0
    losses_today = 0
    day_pnl = 0.0
    cooldown = 0
    stop_for_day = False

    # ORB state per day
    orb_high: Optional[float] = None
    orb_low: Optional[float] = None
    orb_done = False

    for i in range(2, len(candles)):
        d = day_key_utc(candles[i].ts)
        if d != cur_day:
            cur_day = d
            trades_today = 0
            losses_today = 0
            day_pnl = 0.0
            cooldown = 0
            stop_for_day = False

            orb_high = None
            orb_low = None
            orb_done = False

        if cooldown > 0:
            cooldown -= 1

        # manage open position
        if in_pos is not None:
            hi = highs[i]
            lo = lows[i]

            if ax[i] is not None:
                tr = float(ax[i]) * p.exits.trail_atr_mult
                if in_pos == "long":
                    trail = max(trail, hi - tr)
                    sl_eff = max(sl, trail)
                else:
                    trail = min(trail, lo + tr)
                    sl_eff = min(sl, trail)
            else:
                sl_eff = sl

            hit_sl = (lo <= sl_eff) if in_pos == "long" else (hi >= sl_eff)
            hit_tp = (hi >= tp) if in_pos == "long" else (lo <= tp)

            exit_px = None
            if hit_sl and hit_tp:
                exit_px = sl_eff
            elif hit_sl:
                exit_px = sl_eff
            elif hit_tp:
                exit_px = tp

            if exit_px is None and (i - entry_i) >= p.exits.max_hold_bars:
                exit_px = closes[i]

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
                    if losses_today >= p.governor.max_losses_per_day:
                        stop_for_day = True

                if day_pnl <= -abs(p.governor.daily_loss_stop):
                    stop_for_day = True

                in_pos = None

        if in_pos is not None:
            continue

        # flat: update ORB from signal bar j
        j = i - 1
        if not in_session(candles[j].ts, p.session_start_min_utc, p.session_end_min_utc):
            continue

        m = minutes_utc(candles[j].ts)
        orb_end = p.session_start_min_utc + p.orb_minutes
        if p.session_start_min_utc <= orb_end:
            in_orb = p.session_start_min_utc <= m < orb_end
            after_orb = m >= orb_end
        else:
            # wrap midnight ORB (rare) - ignore
            in_orb = False
            after_orb = True

        if in_orb:
            orb_high = highs[j] if orb_high is None else max(orb_high, highs[j])
            orb_low = lows[j] if orb_low is None else min(orb_low, lows[j])

        if after_orb and orb_high is not None and orb_low is not None:
            orb_done = True

        # entry gating
        if stop_for_day or cooldown > 0:
            continue
        if trades_today >= p.governor.max_trades_per_day:
            continue
        if not orb_done:
            continue
        if ax[j] is None:
            continue

        # trend filter
        if p.trend.enabled:
            if ef[j] is None or es[j] is None:
                continue
            spread = abs(float(ef[j]) - float(es[j]))
            if spread < p.trend.min_spread_points:
                continue
            trend_up = float(ef[j]) > float(es[j])
            trend_dn = float(ef[j]) < float(es[j])
        else:
            trend_up = True
            trend_dn = True

        buf = p.buffer_points
        long_sig = trend_up and closes[j] > float(orb_high) + buf
        short_sig = trend_dn and closes[j] < float(orb_low) - buf
        if not (long_sig or short_sig):
            continue

        in_pos = "long" if long_sig else "short"
        entry_i = i
        entry_px = opens[i]

        stop_dist = float(ax[j]) * p.exits.stop_atr_mult
        qty = calc_qty_from_stop(stop_dist, p.risk_per_trade_dollars, p.max_micros)

        if in_pos == "long":
            sl = entry_px - stop_dist
            trail = sl
            tp = entry_px + (float(ax[j]) * p.exits.tp_atr_mult) if p.exits.tp_atr_mult is not None else entry_px + stop_dist * 4.0
        else:
            sl = entry_px + stop_dist
            trail = sl
            tp = entry_px - (float(ax[j]) * p.exits.tp_atr_mult) if p.exits.tp_atr_mult is not None else entry_px - stop_dist * 4.0

        trades_today += 1

    return trades
