"""Strategy v8.1: Open-engine ORB + Pullback continuation with daily target ladder.

Goal: step-change calendar-days pass<=5 by prioritizing the 9:30–10:15 ET open
(move-capture window) while staying conservative outside the open.

Key ideas:
- Regime filter: ATR >= min + EMA spread >= min
- OPEN ENGINE (default 14:30–15:15 UTC):
  - ORB breakout (aggressive settings)
  - Optional pullback continuation (higher frequency)
  - If first open trade is a winner, allow "press mode" daily target
- OUTSIDE OPEN: stricter filters + ORB only (optional), typically lower frequency
- Exits: ATR stop + ATR TP + ATR trail + max hold
- Governor: max trades/losses/day, daily loss stop, cooldown, stop-after-1-loss option

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
    min_spread_points: float = 2.0


@dataclass(frozen=True)
class ORB:
    lookback: int = 12
    buffer_points: float = 0.5


@dataclass(frozen=True)
class Pullback:
    enabled: bool = True
    min_bars_since_cross: int = 1


@dataclass(frozen=True)
class Exits:
    atr_len: int = 14
    stop_atr_mult: float = 0.75
    trail_atr_mult: float = 0.5
    tp_atr_mult: float = 2.0
    max_hold_bars: int = 72


@dataclass(frozen=True)
class Governor:
    max_trades_per_day: int = 4
    max_losses_per_day: int = 2
    daily_loss_stop: float = 300.0
    cooldown_bars_after_loss: int = 6

    # Daily target ladder
    daily_profit_target_base: float = 200.0
    daily_profit_target_press: float = 625.0

    # If True, shut down the day after the first losing trade
    stop_after_first_loss: bool = False


@dataclass(frozen=True)
class Params:
    # Broad allowed session (awake window). Defaults match 7:30am–10pm ET.
    session_start_min_utc: int = 12 * 60 + 30
    session_end_min_utc: int = 3 * 60

    # Open engine window (9:30–10:15 ET => 14:30–15:15 UTC in standard time)
    open_start_min_utc: int = 14 * 60 + 30
    open_end_min_utc: int = 15 * 60 + 15

    risk_per_trade_dollars: float = 150.0
    max_micros: int = 20

    regime: Regime = Regime()

    # Open engine knobs
    open_min_spread_points: float = 1.0
    open_orb: ORB = ORB(lookback=9, buffer_points=0.0)
    open_pullback: Pullback = Pullback(enabled=True, min_bars_since_cross=1)

    # Open-quality confirmation (reduce fakeouts)
    open_confirm_enabled: bool = True
    # for ORB: require close in top/bottom X% of the candle range
    open_confirm_frac: float = 0.6
    # for pullback: require close beyond fast EMA by >= margin*ATR(exits)
    open_pullback_ema_margin_atr: float = 0.1

    # Rest-of-day knobs
    day_orb: ORB = ORB(lookback=18, buffer_points=1.0)
    day_pullback: Pullback = Pullback(enabled=False, min_bars_since_cross=2)

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

    last_cross_i = -999999
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
                    # enable press mode only if first trade is a win AND it happened during open engine
                    if pnl > 0 and in_session(candles[entry_i].ts, p.open_start_min_utc, p.open_end_min_utc):
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

                # daily target ladder
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
        if not in_session(candles[j].ts, p.session_start_min_utc, p.session_end_min_utc):
            continue

        if ef[j] is None or es[j] is None or a[j] is None or ax[j] is None:
            continue

        atrp = float(a[j])
        if atrp < p.regime.atr_min_points:
            continue

        in_open = in_session(candles[j].ts, p.open_start_min_utc, p.open_end_min_utc)

        spread = abs(float(ef[j]) - float(es[j]))
        min_spread = p.open_min_spread_points if in_open else p.regime.min_spread_points
        if spread < min_spread:
            continue

        trend_up = float(ef[j]) > float(es[j])
        trend_dn = float(ef[j]) < float(es[j])

        # track pullback cross of fast EMA
        if closes[j - 1] is not None and ef[j - 1] is not None:
            prev_above = closes[j - 1] > float(ef[j - 1])
            now_above = closes[j] > float(ef[j])
            if prev_above != now_above:
                last_cross_i = j

        orb = p.open_orb if in_open else p.day_orb
        pull = p.open_pullback if in_open else p.day_pullback

        long_sig = False
        short_sig = False

        # Entry A: ORB breakout
        lb = orb.lookback
        if j - lb >= 1:
            rng_high = max(highs[j - lb : j])
            rng_low = min(lows[j - lb : j])
            buf = orb.buffer_points

            if trend_up and closes[j] > rng_high + buf:
                long_sig = True
            if trend_dn and closes[j] < rng_low - buf:
                short_sig = True

            # Open confirmation: avoid weak closes on fake breakouts
            if in_open and p.open_confirm_enabled:
                r = max(1e-9, highs[j] - lows[j])
                frac = p.open_confirm_frac
                if long_sig and closes[j] < lows[j] + frac * r:
                    long_sig = False
                if short_sig and closes[j] > highs[j] - frac * r:
                    short_sig = False

        # Entry B: pullback continuation
        if pull.enabled and not (long_sig or short_sig):
            if (j - last_cross_i) >= pull.min_bars_since_cross:
                if trend_up and closes[j - 1] < float(ef[j - 1]) and closes[j] > float(ef[j]):
                    long_sig = True
                if trend_dn and closes[j - 1] > float(ef[j - 1]) and closes[j] < float(ef[j]):
                    short_sig = True

            # Open confirmation for pullbacks: require some distance beyond EMA
            if in_open and p.open_confirm_enabled and (long_sig or short_sig) and ax[j] is not None:
                margin = float(ax[j]) * p.open_pullback_ema_margin_atr
                if long_sig and (closes[j] - float(ef[j])) < margin:
                    long_sig = False
                if short_sig and (float(ef[j]) - closes[j]) < margin:
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
