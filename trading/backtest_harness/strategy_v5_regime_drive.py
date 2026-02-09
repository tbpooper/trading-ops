"""Strategy v5: Regime-first + session playbook + set-and-forget exits.

v5.2 changes:
- Adds optional conditional size-up on strong regime (still capped by max_micros)

Still:
- No partials.
- Optional breakout->retest->confirmation entry.
- Trailing + time stop exits.

Note: This is a research harness, not production fills.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

from trading.backtest_harness.tv_csv import Candle
from trading.backtest_harness.strategy_v0 import ema, atr, dollars_from_points, TICK, TICK_VALUE

Side = Literal["long", "short"]


@dataclass(frozen=True)
class Window:
    start_min_utc: int
    end_min_utc: int


def minutes_utc(ts: int) -> int:
    import datetime

    d = datetime.datetime.utcfromtimestamp(ts)
    return d.hour * 60 + d.minute


def in_windows(ts: int, windows: Sequence[Window]) -> bool:
    if not windows:
        return True
    m = minutes_utc(ts)
    for w in windows:
        if w.start_min_utc <= w.end_min_utc:
            if w.start_min_utc <= m <= w.end_min_utc:
                return True
        else:
            if m >= w.start_min_utc or m <= w.end_min_utc:
                return True
    return False


def day_key_utc(ts: int) -> str:
    import datetime

    return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class Regime:
    ema_fast: int = 10
    ema_slow: int = 40
    min_spread_points: float = 3.0
    atr_len: int = 14
    atr_min_points: float = 10.0

    # size-up when spread >= sizeup_spread_points
    sizeup_enabled: bool = False
    sizeup_spread_points: float = 6.0
    sizeup_mult: float = 1.5


@dataclass(frozen=True)
class Entry:
    lookback: int = 12

    use_retest: bool = True
    retest_deadline_bars: int = 6
    retest_epsilon_points: float = 5.0


@dataclass(frozen=True)
class Exits:
    atr_len: int = 14
    stop_atr_mult: float = 1.5
    trail_atr_mult: float = 1.0
    max_hold_bars: int = 36
    tp_atr_mult: Optional[float] = None


@dataclass(frozen=True)
class Governor:
    max_trades_per_day: int = 2
    max_losses_per_day: int = 2
    daily_loss_stop: float = 250.0
    cooldown_bars_after_loss: int = 6


@dataclass(frozen=True)
class Params:
    risk_per_trade_dollars: float = 150.0
    max_micros: int = 20

    # default windows: full allowed window split to avoid midnight wrap issues
    windows: Sequence[Window] = (
        Window(12 * 60 + 30, 3 * 60),
    )

    regime: Regime = Regime()
    entry: Entry = Entry()
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

    ef = ema(closes, p.regime.ema_fast)
    es = ema(closes, p.regime.ema_slow)
    a = atr(candles, p.regime.atr_len)

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

    state = "idle"
    level = 0.0
    deadline = 0

    for i in range(2, len(candles)):
        d = day_key_utc(candles[i].ts)
        if d != cur_day:
            cur_day = d
            trades_today = 0
            losses_today = 0
            day_pnl = 0.0
            cooldown = 0
            stop_for_day = False
            state = "idle"

        if cooldown > 0:
            cooldown -= 1

        # manage open
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

        if stop_for_day or cooldown > 0:
            continue
        if trades_today >= p.governor.max_trades_per_day:
            continue

        j = i - 1
        if not in_windows(candles[j].ts, p.windows):
            continue

        if ef[j] is None or es[j] is None or a[j] is None or ax[j] is None:
            continue
        if float(a[j]) < p.regime.atr_min_points:
            continue

        spread = abs(float(ef[j]) - float(es[j]))
        if spread < p.regime.min_spread_points:
            continue

        trend_up = float(ef[j]) > float(es[j])
        trend_dn = float(ef[j]) < float(es[j])

        lb = p.entry.lookback
        if j - lb < 1:
            continue
        rng_high = max(highs[j - lb : j])
        rng_low = min(lows[j - lb : j])
        eps = p.entry.retest_epsilon_points

        long_sig = False
        short_sig = False

        if not p.entry.use_retest:
            long_sig = trend_up and closes[j] > rng_high
            short_sig = trend_dn and closes[j] < rng_low
        else:
            if state == "idle":
                if trend_up and closes[j] > rng_high:
                    state = "wait_retest_long"
                    level = rng_high
                    deadline = j + p.entry.retest_deadline_bars
                elif trend_dn and closes[j] < rng_low:
                    state = "wait_retest_short"
                    level = rng_low
                    deadline = j + p.entry.retest_deadline_bars

            if state == "wait_retest_long":
                if j > deadline:
                    state = "idle"
                else:
                    if (lows[j] <= level + eps) and (closes[j] > level):
                        long_sig = True
                        state = "idle"
            elif state == "wait_retest_short":
                if j > deadline:
                    state = "idle"
                else:
                    if (highs[j] >= level - eps) and (closes[j] < level):
                        short_sig = True
                        state = "idle"

        if not (long_sig or short_sig):
            continue

        in_pos = "long" if long_sig else "short"
        entry_i = i
        entry_px = opens[i]

        stop_dist = float(ax[j]) * p.exits.stop_atr_mult

        risk = p.risk_per_trade_dollars
        if p.regime.sizeup_enabled and spread >= p.regime.sizeup_spread_points:
            risk = risk * p.regime.sizeup_mult

        qty = calc_qty_from_stop(stop_dist, float(risk), p.max_micros)

        if in_pos == "long":
            sl = entry_px - stop_dist
            trail = sl
            if p.exits.tp_atr_mult is None:
                tp = entry_px + stop_dist * 3.0
            else:
                tp = entry_px + float(ax[j]) * p.exits.tp_atr_mult
        else:
            sl = entry_px + stop_dist
            trail = sl
            if p.exits.tp_atr_mult is None:
                tp = entry_px - stop_dist * 3.0
            else:
                tp = entry_px - float(ax[j]) * p.exits.tp_atr_mult

        trades_today += 1

    return trades
