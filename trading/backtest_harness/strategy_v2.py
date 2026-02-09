"""Strategy v2 — adds regime filters + trade frequency governor.

This is a more *selective* evolution of v1 intended to improve Lucid eval pass-rate:
- Regime filter: require EMA spread (trend strength) and optional ATR band
- Optional time windows (UTC minutes-of-day)
- Trade governor:
  - max trades per day
  - cooldown bars after a loss
  - stop trading for the UTC day after hitting a daily loss limit
- Optional breakout confirmation to avoid mean-reversion chop
- Optional breakeven move once trade reaches a threshold

Execution model:
- signal on bar j, enter at next bar open (i)
- SL/TP checked on bar high/low; if both touched -> SL first (conservative)

Note: This stays single-position and produces closed trades for eval simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

from trading.backtest_harness.tv_csv import Candle
from trading.backtest_harness.strategy_v0 import ema, rsi, atr, dollars_from_points, TICK, TICK_VALUE

Side = Literal["long", "short"]


@dataclass(frozen=True)
class TimeWindow:
    start_min_utc: int
    end_min_utc: int


@dataclass(frozen=True)
class Filters:
    windows: Optional[Sequence[TimeWindow]] = None
    atr_min_points: Optional[float] = None
    atr_max_points: Optional[float] = None
    min_ema_spread_points: Optional[float] = None


@dataclass(frozen=True)
class Governor:
    # trade frequency controls (per UTC day)
    max_trades_per_day: Optional[int] = 4
    cooldown_bars_after_loss: int = 6

    # optional: stop trading for day if day pnl <= -daily_loss_stop
    daily_loss_stop: Optional[float] = 200.0

    # optional: stop trading for day after N losing trades
    max_losses_per_day: Optional[int] = 2


@dataclass(frozen=True)
class Breakout:
    enabled: bool = True
    lookback: int = 20
    range_atr_mult: float = 0.4  # require candle range >= ATR * mult


@dataclass(frozen=True)
class ORB:
    """Opening Range Breakout (ORB) style filter.

    If enabled, we define an opening range each day from session_start_min_utc for
    `range_minutes` and only allow trades once price breaks that range.

    This is intended to reduce chop and concentrate trades into high-momentum periods.
    """

    enabled: bool = False
    session_start_min_utc: int = 14 * 60 + 30  # 14:30 UTC ~ 9:30 ET
    range_minutes: int = 30  # 30-min opening range
    one_trade_per_day: bool = True


@dataclass(frozen=True)
class Session:
    """Hard time-of-day gate.

    Used to prevent the strategy from trading overnight (user cannot place trades).
    Default: RTH 14:30–21:00 UTC.
    """

    enabled: bool = True
    start_min_utc: int = 14 * 60 + 30
    end_min_utc: int = 21 * 60


@dataclass(frozen=True)
class Exits:
    # Move SL to breakeven after reaching +X*R (existing behavior)
    move_sl_to_be_at_r_multiple: Optional[float] = 0.6

    # Optional partial take-profit: close `partial_pct` of position at +partial_r_multiple*R,
    # then continue managing the remainder with the original TP.
    partial_enabled: bool = False
    partial_pct: float = 0.5
    partial_r_multiple: float = 0.8

    # Structure-based TP option
    # If enabled, TP is set to the most recent swing (donchian) extreme instead of fixed RR.
    # For long: tp = max(highs[j-lookback:j])
    # For short: tp = min(lows[j-lookback:j])
    structure_tp_enabled: bool = False
    structure_tp_lookback: int = 50


@dataclass(frozen=True)
class Params:
    ema_fast: int = 10
    ema_slow: int = 30
    rsi_len: int = 14
    rsi_min_long: float = 50
    rsi_max_short: float = 50
    atr_len: int = 14
    atr_mult: float = 2.0
    rr: float = 0.75

    # risk/sizing
    risk_per_trade_dollars: float = 200.0
    max_micros: int = 20

    filters: Filters = Filters()
    governor: Governor = Governor()
    breakout: Breakout = Breakout()
    orb: ORB = ORB()
    session: Session = Session()
    exits: Exits = Exits()


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


def in_any_window(ts: int, windows: Optional[Sequence[TimeWindow]]) -> bool:
    if not windows:
        return True
    m = minutes_utc(ts)
    for w in windows:
        if w.start_min_utc <= w.end_min_utc:
            if w.start_min_utc <= m <= w.end_min_utc:
                return True
        else:
            # wraps midnight
            if m >= w.start_min_utc or m <= w.end_min_utc:
                return True
    return False


def calc_qty_from_stop(stop_dist_points: float, risk_per_trade_dollars: float, max_micros: int) -> int:
    stop_ticks = max(1.0, stop_dist_points / TICK)
    qty = int(risk_per_trade_dollars // (stop_ticks * TICK_VALUE))
    qty = max(1, qty)
    qty = min(max_micros, qty)
    return qty


def generate_trades(candles: List[Candle], p: Params) -> List[Trade]:
    closes = [c.close for c in candles]
    opens = [c.open for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]

    ef = ema(closes, p.ema_fast)
    es = ema(closes, p.ema_slow)
    r = rsi(closes, p.rsi_len)
    a = atr(candles, p.atr_len)

    trades: List[Trade] = []

    in_pos: Optional[Side] = None
    qty = 0
    entry_i = -1
    entry_px = 0.0
    sl = 0.0
    tp = 0.0
    init_risk = 0.0
    moved_to_be = False
    partial_done = False
    partial_tp = 0.0

    # governor state (per day)
    cur_day = None
    trades_today = 0
    day_pnl = 0.0
    losses_today = 0
    cooldown = 0
    stop_for_day = False

    # ORB state (per day)
    orb_high: Optional[float] = None
    orb_low: Optional[float] = None
    orb_done = False

    for i in range(1, len(candles)):
        d = day_key_utc(candles[i].ts)
        if cur_day != d:
            cur_day = d
            trades_today = 0
            day_pnl = 0.0
            losses_today = 0
            cooldown = 0
            stop_for_day = False

            orb_high = None
            orb_low = None
            orb_done = False

        if cooldown > 0:
            cooldown -= 1

        # manage open position on bar i
        if in_pos is not None:
            hi = candles[i].high
            lo = candles[i].low

            # optional: partial take-profit
            if p.exits.partial_enabled and (not partial_done) and init_risk > 0:
                partial_trigger = entry_px + init_risk * p.exits.partial_r_multiple if in_pos == "long" else entry_px - init_risk * p.exits.partial_r_multiple
                reached = (hi >= partial_trigger) if in_pos == "long" else (lo <= partial_trigger)
                if reached:
                    # close a fraction at partial_trigger
                    close_qty = max(1, int(round(qty * p.exits.partial_pct)))
                    close_qty = min(close_qty, qty)
                    pnl_points = (partial_trigger - entry_px) if in_pos == "long" else (entry_px - partial_trigger)
                    pnl = dollars_from_points(pnl_points, qty_mnq=close_qty)
                    trades.append(
                        Trade(
                            side=in_pos,
                            qty=close_qty,
                            entry_ts=candles[entry_i].ts,
                            exit_ts=candles[i].ts,
                            entry=entry_px,
                            exit=partial_trigger,
                            sl=sl,
                            tp=tp,
                            pnl_dollars=pnl,
                        )
                    )
                    day_pnl += pnl
                    qty -= close_qty
                    partial_done = True
                    # if fully closed, flatten
                    if qty <= 0:
                        in_pos = None
                        continue

            # optional: move SL to breakeven after reaching threshold
            if (not moved_to_be) and p.exits.move_sl_to_be_at_r_multiple is not None:
                if init_risk > 0:
                    be_trigger = entry_px + init_risk * p.exits.move_sl_to_be_at_r_multiple if in_pos == "long" else entry_px - init_risk * p.exits.move_sl_to_be_at_r_multiple
                    reached = (hi >= be_trigger) if in_pos == "long" else (lo <= be_trigger)
                    if reached:
                        sl = max(sl, entry_px) if in_pos == "long" else min(sl, entry_px)
                        moved_to_be = True

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

                # governor updates
                day_pnl += pnl
                if pnl < 0:
                    losses_today += 1
                    cooldown = max(cooldown, p.governor.cooldown_bars_after_loss)
                    if p.governor.max_losses_per_day is not None and losses_today >= p.governor.max_losses_per_day:
                        stop_for_day = True

                if p.governor.daily_loss_stop is not None and day_pnl <= -abs(p.governor.daily_loss_stop):
                    stop_for_day = True

                in_pos = None

        # if flat, check signal on j=i-1 and enter at i open
        if in_pos is None:
            # Update ORB range using the signal bar (j=i-1) so it's ready before entries.
            j = i - 1
            m = minutes_utc(candles[j].ts)
            if p.orb.enabled:
                start = p.orb.session_start_min_utc
                end = start + p.orb.range_minutes
                if start <= m < end:
                    orb_high = highs[j] if orb_high is None else max(orb_high, highs[j])
                    orb_low = lows[j] if orb_low is None else min(orb_low, lows[j])
                if m >= end and orb_high is not None and orb_low is not None:
                    orb_done = True

            if stop_for_day:
                continue
            if p.orb.enabled and p.orb.one_trade_per_day and trades_today >= 1:
                continue
            if p.governor.max_trades_per_day is not None and trades_today >= p.governor.max_trades_per_day:
                continue
            if cooldown > 0:
                continue

            if ef[j] is None or es[j] is None or r[j] is None or a[j] is None:
                continue

            # filters on signal bar
            if p.session.enabled:
                m2 = minutes_utc(candles[j].ts)
                if p.session.start_min_utc <= p.session.end_min_utc:
                    if not (p.session.start_min_utc <= m2 <= p.session.end_min_utc):
                        continue
                else:
                    # wraps midnight
                    if not (m2 >= p.session.start_min_utc or m2 <= p.session.end_min_utc):
                        continue

            if not in_any_window(candles[j].ts, p.filters.windows):
                continue
            if p.filters.atr_min_points is not None and a[j] < p.filters.atr_min_points:
                continue
            if p.filters.atr_max_points is not None and a[j] > p.filters.atr_max_points:
                continue
            if p.filters.min_ema_spread_points is not None:
                if abs(ef[j] - es[j]) < p.filters.min_ema_spread_points:
                    continue

            trend_up = ef[j] > es[j]
            trend_dn = ef[j] < es[j]

            # Two entry styles:
            # - Pullback: trend + close crosses against fast EMA (mean-reversion within trend)
            # - Breakout: trend + close breaks lookback extreme (momentum)
            long_pullback = trend_up and closes[j] < ef[j] and r[j] >= p.rsi_min_long
            short_pullback = trend_dn and closes[j] > ef[j] and r[j] <= p.rsi_max_short

            if p.breakout.enabled:
                lb = p.breakout.lookback
                if j - lb < 1:
                    continue
                prev_high = max(highs[j - lb : j])
                prev_low = min(lows[j - lb : j])
                rng = highs[j] - lows[j]
                if rng < a[j] * p.breakout.range_atr_mult:
                    continue

                long_breakout = trend_up and closes[j] >= prev_high and r[j] >= p.rsi_min_long
                short_breakout = trend_dn and closes[j] <= prev_low and r[j] <= p.rsi_max_short

                long_sig = long_pullback or long_breakout
                short_sig = short_pullback or short_breakout
            else:
                long_sig = long_pullback
                short_sig = short_pullback

            # ORB gate: only take signals after ORB is defined and broken.
            if p.orb.enabled:
                start = p.orb.session_start_min_utc
                end = start + p.orb.range_minutes
                if not orb_done or orb_high is None or orb_low is None:
                    continue
                if m < end:
                    continue
                long_sig = long_sig and closes[j] > orb_high
                short_sig = short_sig and closes[j] < orb_low

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

            moved_to_be = False
            partial_done = False
            init_risk = stop_dist

            if in_pos == "long":
                sl = entry_px - stop_dist
                if p.exits.structure_tp_enabled:
                    lb2 = p.exits.structure_tp_lookback
                    start = max(0, j - lb2)
                    struct_tp = max(highs[start:j]) if j > start else entry_px + stop_dist * p.rr
                    tp = max(entry_px + TICK, float(struct_tp))
                else:
                    tp = entry_px + stop_dist * p.rr
            else:
                sl = entry_px + stop_dist
                if p.exits.structure_tp_enabled:
                    lb2 = p.exits.structure_tp_lookback
                    start = max(0, j - lb2)
                    struct_tp = min(lows[start:j]) if j > start else entry_px - stop_dist * p.rr
                    tp = min(entry_px - TICK, float(struct_tp))
                else:
                    tp = entry_px - stop_dist * p.rr

            trades_today += 1

    return trades
