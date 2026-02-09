"""Strategy v7: Regime switch (drive vs chop vs no-trade).

Goal: step-change pass<=5 by not forcing one behavior on all days.

Regimes (computed on the fly using EMAs + ATR):
- DRIVE: trend strength high (ema spread >= threshold) and ATR >= threshold.
  Entry: breakout/continuation (similar to v5/v2)
- CHOP: atr ok but trend strength low.
  Entry: mean reversion back to fast EMA after excursion (simple fade)
- NO_TRADE: low ATR (skip)

This is research-only and intentionally simple. No partials.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

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

    atr_min_points: float = 8.0
    drive_spread_points: float = 2.0


@dataclass(frozen=True)
class Drive:
    lookback: int = 12
    buffer_points: float = 0.5


@dataclass(frozen=True)
class Chop:
    # mean reversion entry: if price deviates from fast EMA by >= k*ATR, fade back
    dev_atr_mult: float = 0.8
    # require RSI filter? (skip for now)


@dataclass(frozen=True)
class Exits:
    atr_len: int = 14
    stop_atr_mult: float = 1.0
    trail_atr_mult: float = 0.75
    tp_atr_mult: Optional[float] = 2.5
    max_hold_bars: int = 60


@dataclass(frozen=True)
class Governor:
    max_trades_per_day: int = 3
    max_losses_per_day: int = 2
    daily_loss_stop: float = 250.0
    cooldown_bars_after_loss: int = 6


@dataclass(frozen=True)
class Params:
    session_start_min_utc: int = 12 * 60 + 30
    session_end_min_utc: int = 3 * 60

    risk_per_trade_dollars: float = 150.0
    max_micros: int = 20

    regime: Regime = Regime()
    drive: Drive = Drive()
    chop: Chop = Chop()
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

    for i in range(2, len(candles)):
        d = day_key_utc(candles[i].ts)
        if d != cur_day:
            cur_day = d
            trades_today = 0
            losses_today = 0
            day_pnl = 0.0
            cooldown = 0
            stop_for_day = False

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
        if not in_session(candles[j].ts, p.session_start_min_utc, p.session_end_min_utc):
            continue

        if ef[j] is None or es[j] is None or a[j] is None or ax[j] is None:
            continue

        atrp = float(a[j])
        spread = abs(float(ef[j]) - float(es[j]))
        if atrp < p.regime.atr_min_points:
            continue  # NO_TRADE

        drive = spread >= p.regime.drive_spread_points

        long_sig = False
        short_sig = False

        if drive:
            lb = p.drive.lookback
            if j - lb >= 1:
                rng_high = max(highs[j - lb : j])
                rng_low = min(lows[j - lb : j])
                buf = p.drive.buffer_points
                long_sig = closes[j] > rng_high + buf and float(ef[j]) > float(es[j])
                short_sig = closes[j] < rng_low - buf and float(ef[j]) < float(es[j])
        else:
            # CHOP: fade excursion away from fast EMA by >= dev_atr_mult*ATR
            dev = float(ax[j]) * p.chop.dev_atr_mult
            if closes[j] >= float(ef[j]) + dev:
                short_sig = True
            elif closes[j] <= float(ef[j]) - dev:
                long_sig = True

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
            tp = entry_px + (float(ax[j]) * p.exits.tp_atr_mult) if p.exits.tp_atr_mult is not None else entry_px + stop_dist * 3.0
        else:
            sl = entry_px + stop_dist
            trail = sl
            tp = entry_px - (float(ax[j]) * p.exits.tp_atr_mult) if p.exits.tp_atr_mult is not None else entry_px - stop_dist * 3.0

        trades_today += 1

    return trades
