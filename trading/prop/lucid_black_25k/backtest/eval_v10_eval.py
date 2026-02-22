"""Eval v10 evaluator (pure python, GitHub Actions friendly).

LucidBlack 25k evaluation rules (Thomas confirmed):
- Profit target: $1,250
- Consistency cap: 60% (largest single-day profit / total profit <= 60%)
- Max Loss Limit (MLL): $1,000
- Drawdown type: EOD trailing/lock
- No daily loss limit
- EOD session close: 5:00pm ET (handled externally; we approximate via daily closes)

v10 change: split trading windows (fixed daily schedule):
- 9:30–10:15 ET
- 10:45–12:00 ET

This module provides:
- compute_day_pnl_for_sessions(candles, params, sessions)
- score_eval_pass(day_pnl_by_date)

No external deps.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

from trading.backtest_harness.strategy_v9_open_snapback import Params, Regime, Snapback, Exits, Governor, generate_trades
from trading.backtest_harness.tv_csv import Candle


@dataclass(frozen=True)
class Eval25kSpec:
    profit_target: float = 1250.0
    consistency_cap: float = 0.60

    start_balance: float = 25000.0
    max_loss_limit: float = 1000.0

    # EOD trail/lock (25k)
    initial_trail_balance: float = 26100.0
    locked_mll_balance: float = 25100.0


def _day_key_utc(ts: int) -> str:
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")


def _parse_day_key(s: str) -> date:
    y, m, d = s.split("-")
    return date(int(y), int(m), int(d))


def _is_weekday(d: date) -> bool:
    return d.weekday() < 5


def _iter_bdays(start: date, end: date) -> List[date]:
    out: List[date] = []
    cur = start
    while cur <= end:
        if _is_weekday(cur):
            out.append(cur)
        cur += timedelta(days=1)
    return out


def compute_day_pnl_for_sessions(candles: List[Candle], params: dict, sessions_utc: List[Tuple[int, int]]) -> Dict[str, float]:
    """Aggregate realized PnL by UTC date key across multiple disjoint sessions."""

    agg: Dict[str, float] = {}

    for os, oe in sessions_utc:
        p = Params(
            open_start_min_utc=int(os),
            open_end_min_utc=int(oe),
            risk_per_trade_dollars=float(params["risk"]),
            max_micros=int(params.get("max_micros", 20)),
            regime=Regime(atr_min_points=float(params["atr_min"])),
            snap=Snapback(
                dev_atr_mult=float(params["dev"]),
                require_reversal_bar=bool(params.get("require_reversal_bar", True)),
                use_spread_filter=bool(params.get("use_spread", False)),
                max_spread_points=float(params.get("max_spread", 2.0)),
            ),
            exits=Exits(
                stop_atr_mult=float(params["stop"]),
                trail_atr_mult=float(params["trail"]),
                tp_atr_mult=float(params["tp"]),
                max_hold_bars=int(params["hold"]),
            ),
            governor=Governor(
                max_trades_per_day=int(params["mt"]),
                max_losses_per_day=int(params.get("ml", 99)),
                daily_loss_stop=float(params.get("dls", 1e9)),
                cooldown_bars_after_loss=int(params.get("cool", 0)),
                daily_profit_target_base=float(params.get("dpt_base", 1e9)),
                daily_profit_target_press=float(params.get("dpt_press", 1e9)),
                stop_after_first_loss=bool(params.get("stop1", False)),
            ),
        )

        trades = generate_trades(candles, p)
        for t in trades:
            dk = _day_key_utc(t.exit_ts)
            agg[dk] = agg.get(dk, 0.0) + float(t.pnl_dollars)

    # Fill missing business days with 0
    if not agg:
        return {}

    keys = sorted(agg.keys())
    start = _parse_day_key(keys[0])
    end = _parse_day_key(keys[-1])
    for d in _iter_bdays(start, end):
        dk = d.strftime("%Y-%m-%d")
        agg.setdefault(dk, 0.0)

    return dict(sorted(agg.items()))


def _eval_cycle(day_keys: List[str], day_pnl: Dict[str, float], start_i: int, max_days: int, spec: Eval25kSpec) -> Tuple[bool, Optional[int], bool]:
    total = 0.0
    largest = float("-inf")

    close_bal = spec.start_balance
    high_close = close_bal
    mll = high_close - spec.max_loss_limit

    for k in range(max_days):
        i = start_i + k
        if i >= len(day_keys):
            break

        dk = day_keys[i]
        pnl = float(day_pnl.get(dk, 0.0))

        total += pnl
        largest = max(largest, pnl)

        close_bal += pnl
        high_close = max(high_close, close_bal)

        if high_close >= spec.initial_trail_balance:
            mll = spec.locked_mll_balance
        else:
            mll = high_close - spec.max_loss_limit

        if close_bal <= mll:
            return False, None, True

        if total >= spec.profit_target and total > 0:
            if (largest / total) <= spec.consistency_cap:
                return True, k + 1, False

    return False, None, False


def score_eval_pass(day_pnl_by_date: Dict[str, float], horizons=(4, 5, 7, 10), max_scan_days: int = 15) -> dict:
    spec = Eval25kSpec()

    day_keys = list(day_pnl_by_date.keys())
    totals = 0
    breaches = 0
    timeouts = 0
    pass_counts = {h: 0 for h in horizons}

    for start_i in range(0, len(day_keys)):
        if start_i + max_scan_days >= len(day_keys):
            break

        totals += 1
        passed, days, breached = _eval_cycle(day_keys, day_pnl_by_date, start_i, max_scan_days, spec)
        if breached:
            breaches += 1
            continue
        if not passed:
            timeouts += 1
            continue
        assert days is not None
        for h in horizons:
            if days <= h:
                pass_counts[h] += 1

    def rate(n: int, d: int) -> float:
        return (n / d) if d else 0.0

    out = {
        "samples": totals,
        "breach_rate": rate(breaches, totals),
        "timeout_rate": rate(timeouts, totals),
    }
    for h in horizons:
        out[f"pass_leq{h}_rate"] = rate(pass_counts[h], totals)
    return out
