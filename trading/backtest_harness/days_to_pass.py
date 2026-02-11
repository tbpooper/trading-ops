"""Compute distribution of days-to-pass (or fail) for Lucid eval simulation.

For each possible start day in the dataset, simulate forward day-by-day until:
- target hit (pass)
- MLL breach / consistency breach (fail)
- max_days reached (timeout)

This is intended to answer: "If we didn't stop at 5 days, how many days would it take?"

Note: day boundaries are UTC (consistent with the rest of the harness).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from trading.backtest_harness.tv_csv import Candle
from trading.prop.lucid_black_25k.risk_governor import Lucid25kRules, step_eod_drawdown, is_consistency_ok


def day_key(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def build_day_index(candles: Sequence[Candle]) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
    start_end: Dict[str, List[int]] = {}
    for i, c in enumerate(candles):
        d = day_key(c.ts)
        if d not in start_end:
            start_end[d] = [i, i]
        else:
            start_end[d][1] = i
    days = sorted(start_end.keys())
    index = {d: (start_end[d][0], start_end[d][1]) for d in days}
    return days, index


@dataclass
class PathResult:
    outcome: str  # target_hit | mll_breach | consistency_breach | timeout
    days: int
    total_profit: float
    largest_day: float


def simulate_path(
    trades: Iterable,
    days_order: Sequence[str],
    rules: Optional[Lucid25kRules] = None,
    daily_profit_cap: float = 750.0,
    daily_loss_cap: Optional[float] = 300.0,
    max_days: int = 30,
) -> PathResult:
    """Simulate an eval path over *calendar trading days*.

    Important: days_order must include days even when there are no trades.
    This matches real eval time: if you buy on Monday and don't trade until
    Wednesday, Monday+Tuesday still count as days elapsed.

    (We treat "calendar" as the dataset's trading days; weekends/holidays are
    absent from the candle series.)
    """

    rules = rules or Lucid25kRules()
    tlist = sorted(list(trades), key=lambda t: t.exit_ts)

    # bucket trades by exit-day
    by_day: Dict[str, List] = defaultdict(list)
    for t in tlist:
        by_day[day_key(t.exit_ts)].append(t)

    bal = rules.start_balance
    total = 0.0
    largest = 0.0
    highest_close = rules.start_balance

    # simulate day by day, up to max_days or available days
    for di, d in enumerate(list(days_order)[:max_days], start=1):
        day_profit = 0.0
        for t in by_day.get(d, []):
            if day_profit >= daily_profit_cap:
                break
            if daily_loss_cap is not None and day_profit <= -abs(daily_loss_cap):
                break

            day_profit += float(t.pnl_dollars)
            if day_profit > daily_profit_cap:
                day_profit = daily_profit_cap
                break

        total += day_profit
        largest = max(largest, day_profit)
        bal += day_profit

        highest_close, floor, breached = step_eod_drawdown(rules, highest_close, bal)
        if breached:
            return PathResult("mll_breach", di, total, largest)

        if largest > rules.eval_consistency_cap * rules.eval_profit_target:
            return PathResult("consistency_breach", di, total, largest)

        if total >= rules.eval_profit_target:
            if not is_consistency_ok(total, largest, rules.eval_consistency_cap):
                return PathResult("consistency_breach", di, total, largest)
            return PathResult("target_hit", di, total, largest)

    return PathResult("timeout", min(max_days, len(days_order)), total, largest)


def days_to_pass_distribution(
    candles: Sequence[Candle],
    gen_trades: Callable[[List[Candle]], Sequence],
    rules: Optional[Lucid25kRules] = None,
    daily_profit_cap: float = 750.0,
    daily_loss_cap: Optional[float] = 300.0,
    max_days: int = 30,
) -> Dict:
    rules = rules or Lucid25kRules()
    days, idx = build_day_index(candles)

    pass_days = Counter()
    outcomes = Counter()
    # track average days for passes and fails
    sum_days = defaultdict(int)
    n_days = defaultdict(int)

    for s in range(len(days)):
        start_i = idx[days[s]][0]
        sub = list(candles[start_i:])
        trades = list(gen_trades(sub))
        res = simulate_path(
            trades,
            days_order=days[s : s + max_days],
            rules=rules,
            daily_profit_cap=daily_profit_cap,
            daily_loss_cap=daily_loss_cap,
            max_days=max_days,
        )
        outcomes[res.outcome] += 1
        sum_days[res.outcome] += res.days
        n_days[res.outcome] += 1
        if res.outcome == "target_hit":
            pass_days[res.days] += 1

    out = {
        "windows": len(days),
        "max_days": max_days,
        "outcomes": dict(outcomes),
        "avg_days": {k: (sum_days[k] / n_days[k] if n_days[k] else None) for k in outcomes.keys()},
        "pass_days_hist": dict(sorted(pass_days.items())),
        "pass_rate": (outcomes["target_hit"] / len(days) if days else 0.0),
    }
    return out
