"""Strategy v9: Open snapback mean-reversion.

Goal: raise calendar-days pass<=5 by capturing high-probability reversals around the open.

Idea:
- Trade only during open window (9:30–10:15 ET by default).
- Identify excursion away from fast EMA by k*ATR, then require a reversal bar and
  take a snapback entry toward EMA.
- Fast exits: ATR stop, modest ATR TP, tight time stop.
- Strong governor: stop after 1 loss optional; daily target ladder.

Research harness only.
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


def in_session(ts: int, start_min_utc: int, end_min_utc: int) -> bool:
    m = minutes_utc(ts)
    if start_min_utc <= end_min_utc:
        return start_min_utc <= m <= end_min_utc
    return m >= start_min_utc or m <= end_min_utc


def day_key_utc(ts: int) -> str:
    import datetime

    return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class Regime:
    ema_fast: int = 10
    ema_slow: int = 40
    atr_len: int = 14

    atr_min_points: float = 10.0


@dataclass(frozen=True)
class Snapback:
    dev_atr_mult: float = 0.8
    require_reversal_bar: bool = True
    # optional trend filter: only fade when slow trend is flat-ish
    use_spread_filter: bool = True
    max_spread_points: float = 2.0


@dataclass(frozen=True)
class Exits:
    atr_len: int = 14
    stop_atr_mult: float = 0.6
    trail_atr_mult: float = 0.4
    tp_atr_mult: float = 1.5
    max_hold_bars: int = 24


@dataclass(frozen=True)
class Governor:
    max_trades_per_day: int = 4
    max_losses_per_day: int = 2
    daily_loss_stop: float = 300.0
    cooldown_bars_after_loss: int = 4

    daily_profit_target_base: float = 200.0
    daily_profit_target_press: float = 625.0
    stop_after_first_loss: bool = False


@dataclass(frozen=True)
class Params:
    # open window (9:30–10:15 ET => 14:30–15:15 UTC standard time)
    open_start_min_utc: int = 14 * 60 + 30
    open_end_min_utc: int = 15 * 60 + 15

    risk_per_trade_dollars: float = 150.0
    max_micros: int = 20

    regime: Regime = Regime()
    snap: Snapback = Snapback()
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


def calc_qty(stop_dist_points: float, risk_per_trade_dollars: float, max_micros: int) -> int:
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
    press_mode = False
    first_trade_done = False

    for i in range(2, len(candles)):
        d = day_key_utc(candles[i].ts)
        if d != cur_day:
            cur_day = d
            trades_today = 0
            losses_today = 0
            day_pnl = 0.0
            cooldown = 0
            stop_for_day = False
            press_mode = False
            first_trade_done = False

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
                if not first_trade_done:
                    first_trade_done = True
                    if pnl > 0:
                        press_mode = True

                if pnl < 0:
                    losses_today += 1
                    cooldown = max(cooldown, p.governor.cooldown_bars_after_loss)
                    if p.governor.stop_after_first_loss:
                        stop_for_day = True
                    if losses_today >= p.governor.max_losses_per_day:
                        stop_for_day = True

                if day_pnl <= -abs(p.governor.daily_loss_stop):
                    stop_for_day = True

                target = p.governor.daily_profit_target_press if press_mode else p.governor.daily_profit_target_base
                if day_pnl >= abs(target):
                    stop_for_day = True

                in_pos = None

        if in_pos is not None:
            continue

        if stop_for_day or cooldown > 0:
            continue
        if trades_today >= p.governor.max_trades_per_day:
            continue

        j = i - 1
        if not in_session(candles[j].ts, p.open_start_min_utc, p.open_end_min_utc):
            continue

        if ef[j] is None or es[j] is None or a[j] is None or ax[j] is None:
            continue

        if float(a[j]) < p.regime.atr_min_points:
            continue

        if p.snap.use_spread_filter:
            spread = abs(float(ef[j]) - float(es[j]))
            if spread > p.snap.max_spread_points:
                continue

        dev = float(ax[j]) * p.snap.dev_atr_mult

        long_sig = False
        short_sig = False

        # fade excursion away from EMA fast
        if closes[j] <= float(ef[j]) - dev:
            long_sig = True
        elif closes[j] >= float(ef[j]) + dev:
            short_sig = True

        if p.snap.require_reversal_bar and (long_sig or short_sig):
            # require reversal body direction
            if long_sig and closes[j] < opens[j]:
                long_sig = False
            if short_sig and closes[j] > opens[j]:
                short_sig = False

        if not (long_sig or short_sig):
            continue

        in_pos = "long" if long_sig else "short"
        entry_i = i
        entry_px = opens[i]

        stop_dist = float(ax[j]) * p.exits.stop_atr_mult
        qty = calc_qty(stop_dist, p.risk_per_trade_dollars, p.max_micros)

        if in_pos == "long":
            sl = entry_px - stop_dist
            trail = sl
            tp = entry_px + float(ax[j]) * p.exits.tp_atr_mult
        else:
            sl = entry_px + stop_dist
            trail = sl
            tp = entry_px - float(ax[j]) * p.exits.tp_atr_mult

        trades_today += 1

    return trades
