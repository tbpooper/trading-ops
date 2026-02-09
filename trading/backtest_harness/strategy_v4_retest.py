"""Strategy v4: Breakout -> retest confirmation with dynamic TP.

Uses 1m candles for execution and 15m candles for regime filter.

Idea:
- Regime (15m): require EMA fast > EMA slow and EMA spread >= min_spread (trend) and ATR >= atr_min.
- Level (1m): use recent swing/Donchian high/low over L as breakout level.
- Breakout phase: close crosses above level.
- Retest phase: price pulls back to within epsilon of level (or crosses back) within N bars.
- Confirmation: bullish/bearish close back in breakout direction -> enter.

Exits:
- stop: ATR*atr_mult from entry
- take profit: dynamic target = entry + tp_atr_mult*ATR OR next structure (donchian band)
- time stop: exit after max_hold_bars at market (close)
- move SL to BE after be_atr_mult*ATR

Governors:
- max trades/day
- max losses/day
- daily loss stop

This is experimental and intended to change the edge vs v2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from trading.backtest_harness.tv_csv_multi import Candle
from trading.backtest_harness.strategy_v0 import ema, atr, dollars_from_points, TICK, TICK_VALUE

Side = Literal["long", "short"]


@dataclass(frozen=True)
class Session:
    enabled: bool = True
    start_min_utc: int = 12 * 60 + 30
    end_min_utc: int = 3 * 60


@dataclass(frozen=True)
class Governor:
    max_trades_per_day: Optional[int] = 3
    max_losses_per_day: Optional[int] = 2
    daily_loss_stop: Optional[float] = 250.0
    cooldown_bars_after_loss: int = 10


@dataclass(frozen=True)
class Regime:
    ema_fast: int = 20
    ema_slow: int = 50
    min_spread_points: float = 3.0
    atr_len: int = 14
    atr_min_points: float = 10.0


@dataclass(frozen=True)
class Entry:
    level_lookback: int = 60
    retest_deadline_bars: int = 45
    retest_epsilon_points: float = 5.0


@dataclass(frozen=True)
class Exits:
    atr_len: int = 14
    atr_mult_stop: float = 2.0
    tp_atr_mult: float = 3.0
    be_atr_mult: Optional[float] = 1.0
    max_hold_bars: int = 240


@dataclass(frozen=True)
class Params:
    risk_per_trade_dollars: float = 200.0
    max_micros: int = 20

    session: Session = Session()
    governor: Governor = Governor()
    regime: Regime = Regime()
    entry: Entry = Entry()
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


def in_session(ts: int, s: Session) -> bool:
    if not s.enabled:
        return True
    m = minutes_utc(ts)
    if s.start_min_utc <= s.end_min_utc:
        return s.start_min_utc <= m <= s.end_min_utc
    return m >= s.start_min_utc or m <= s.end_min_utc


def day_key_utc(ts: int) -> str:
    import datetime

    return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")


def calc_qty_from_stop(stop_dist_points: float, risk_per_trade_dollars: float, max_micros: int) -> int:
    stop_ticks = max(1.0, stop_dist_points / TICK)
    qty = int(risk_per_trade_dollars // (stop_ticks * TICK_VALUE))
    qty = max(1, qty)
    qty = min(max_micros, qty)
    return qty


def align_regime(c1: List[Candle], c15: List[Candle]) -> List[int]:
    """For each 1m candle index, return the index of the most recent 15m candle."""
    out = []
    j = 0
    for i in range(len(c1)):
        ts = c1[i].ts
        while j + 1 < len(c15) and c15[j + 1].ts <= ts:
            j += 1
        out.append(j)
    return out


def generate_trades(c1: List[Candle], c15: List[Candle], p: Params) -> List[Trade]:
    closes15 = [c.close for c in c15]
    ef15 = ema(closes15, p.regime.ema_fast)
    es15 = ema(closes15, p.regime.ema_slow)
    atr15 = atr(c15, p.regime.atr_len)

    atr1 = atr(c1, p.exits.atr_len)

    map15 = align_regime(c1, c15)

    trades: List[Trade] = []

    in_pos: Optional[Side] = None
    qty = 0
    entry_i = -1
    entry_px = 0.0
    sl = 0.0
    tp = 0.0
    be_moved = False

    # governor state per day
    cur_day = None
    trades_today = 0
    losses_today = 0
    day_pnl = 0.0
    cooldown = 0
    stop_for_day = False

    # state machine for setup
    state = "idle"  # idle | breakout_long | breakout_short
    level = None
    deadline_i = None

    for i in range(2, len(c1)):
        d = day_key_utc(c1[i].ts)
        if cur_day != d:
            cur_day = d
            trades_today = 0
            losses_today = 0
            day_pnl = 0.0
            cooldown = 0
            stop_for_day = False
            state = "idle"
            level = None
            deadline_i = None

        if cooldown > 0:
            cooldown -= 1

        # manage open position
        if in_pos is not None:
            hi = c1[i].high
            lo = c1[i].low

            # move SL to BE once price moves in favor by threshold
            if (not be_moved) and p.exits.be_atr_mult is not None and atr1[i] is not None:
                thr = float(atr1[i]) * p.exits.be_atr_mult
                if in_pos == "long" and hi >= entry_px + thr:
                    sl = max(sl, entry_px)
                    be_moved = True
                if in_pos == "short" and lo <= entry_px - thr:
                    sl = min(sl, entry_px)
                    be_moved = True

            hit_sl = (lo <= sl) if in_pos == "long" else (hi >= sl)
            hit_tp = (hi >= tp) if in_pos == "long" else (lo <= tp)

            exit_px = None
            if hit_sl and hit_tp:
                exit_px = sl
            elif hit_sl:
                exit_px = sl
            elif hit_tp:
                exit_px = tp

            # time stop
            if exit_px is None and (i - entry_i) >= p.exits.max_hold_bars:
                exit_px = c1[i].close

            if exit_px is not None:
                pnl_points = (exit_px - entry_px) if in_pos == "long" else (entry_px - exit_px)
                pnl = dollars_from_points(pnl_points, qty_mnq=qty)
                trades.append(
                    Trade(
                        side=in_pos,
                        qty=qty,
                        entry_ts=c1[entry_i].ts,
                        exit_ts=c1[i].ts,
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
                be_moved = False

        if in_pos is not None:
            continue

        # flat: maybe stop
        if stop_for_day or cooldown > 0:
            continue
        if p.governor.max_trades_per_day is not None and trades_today >= p.governor.max_trades_per_day:
            continue

        # session
        if not in_session(c1[i - 1].ts, p.session):
            continue

        # regime filter using mapped 15m bar
        j15 = map15[i - 1]
        if ef15[j15] is None or es15[j15] is None or atr15[j15] is None:
            continue
        if float(atr15[j15]) < p.regime.atr_min_points:
            continue
        spread = abs(float(ef15[j15]) - float(es15[j15]))
        if spread < p.regime.min_spread_points:
            continue

        trend_up = float(ef15[j15]) > float(es15[j15])
        trend_dn = float(ef15[j15]) < float(es15[j15])

        # compute breakout level from 1m history
        L = p.entry.level_lookback
        j = i - 1
        if j - L < 2:
            continue
        prev_high = max(x.high for x in c1[j - L : j])
        prev_low = min(x.low for x in c1[j - L : j])

        # state transitions
        close_j = c1[j].close
        eps = p.entry.retest_epsilon_points

        if state == "idle":
            if trend_up and close_j > prev_high:
                state = "breakout_long"
                level = prev_high
                deadline_i = j + p.entry.retest_deadline_bars
            elif trend_dn and close_j < prev_low:
                state = "breakout_short"
                level = prev_low
                deadline_i = j + p.entry.retest_deadline_bars

        if state == "breakout_long":
            if deadline_i is not None and j > deadline_i:
                state = "idle"
            else:
                # retest touch
                if c1[j].low <= float(level) + eps:
                    # confirmation close back above level
                    if close_j > float(level):
                        # enter long next bar open
                        if atr1[j] is None:
                            state = "idle"
                        else:
                            entry_i = i
                            entry_px = c1[i].open
                            stop_dist = float(atr1[j]) * p.exits.atr_mult_stop
                            qty = calc_qty_from_stop(stop_dist, p.risk_per_trade_dollars, p.max_micros)
                            sl = entry_px - stop_dist
                            tp = entry_px + float(atr1[j]) * p.exits.tp_atr_mult
                            in_pos = "long"
                            trades_today += 1
                            state = "idle"

        if state == "breakout_short":
            if deadline_i is not None and j > deadline_i:
                state = "idle"
            else:
                if c1[j].high >= float(level) - eps:
                    if close_j < float(level):
                        if atr1[j] is None:
                            state = "idle"
                        else:
                            entry_i = i
                            entry_px = c1[i].open
                            stop_dist = float(atr1[j]) * p.exits.atr_mult_stop
                            qty = calc_qty_from_stop(stop_dist, p.risk_per_trade_dollars, p.max_micros)
                            sl = entry_px + stop_dist
                            tp = entry_px - float(atr1[j]) * p.exits.tp_atr_mult
                            in_pos = "short"
                            trades_today += 1
                            state = "idle"

    return trades
