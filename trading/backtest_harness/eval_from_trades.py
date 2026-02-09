"""Lucid eval simulation directly from a chronological trade stream.

This allows us to model *gating* rules that depend on intra-day progression:
- stop trading after daily profit cap (e.g. $750)
- stop trading after daily max loss (risk control)

We still enforce EOD trailing MLL based on closes.

Assumptions:
- Trades do not overlap (true for our single-position strategy model)
- We bucket by UTC day boundary.

This is a simplification but closer than aggregating all trades then evaluating.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple

from trading.prop.lucid_black_25k.risk_governor import Lucid25kRules, step_eod_drawdown, is_consistency_ok


@dataclass
class EvalSimResult:
    passed: bool
    reason: str
    days: int
    total_profit: float
    largest_day: float
    highest_close: float
    mll_floor: float


def day_key(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d')


def simulate_eval_from_trades(
    trades: Iterable,
    rules: Optional[Lucid25kRules] = None,
    max_days: int = 5,
    daily_profit_cap: float = 750.0,
    daily_loss_cap: Optional[float] = None,
) -> EvalSimResult:
    rules = rules or Lucid25kRules()

    # group trades by exit day in chronological order
    tlist = sorted(list(trades), key=lambda t: t.exit_ts)

    total = 0.0
    largest = 0.0
    highest_close = rules.start_balance
    mll_floor = highest_close - rules.max_loss_limit

    # Determine ordered days present
    days_order: List[str] = []
    for t in tlist:
        d = day_key(t.exit_ts)
        if not days_order or days_order[-1] != d:
            days_order.append(d)

    days_order = days_order[:max_days]

    bal = rules.start_balance

    for di, d in enumerate(days_order, start=1):
        day_profit = 0.0
        for t in [x for x in tlist if day_key(x.exit_ts) == d]:
            # gating: stop if already hit cap
            if day_profit >= daily_profit_cap:
                break
            if daily_loss_cap is not None and day_profit <= -abs(daily_loss_cap):
                break

            day_profit += float(t.pnl_dollars)

            if day_profit > daily_profit_cap:
                # clamp to cap (conservative; assumes we stop once crossed)
                day_profit = daily_profit_cap
                break

        total += day_profit
        largest = max(largest, day_profit)
        bal += day_profit

        highest_close, mll_floor, breached = step_eod_drawdown(rules, highest_close, bal)
        if breached:
            return EvalSimResult(False, 'mll_breach', di, total, largest, highest_close, mll_floor)

        # eval daily cap implied consistency
        if largest > rules.eval_consistency_cap * rules.eval_profit_target:
            return EvalSimResult(False, 'consistency_breach', di, total, largest, highest_close, mll_floor)

        if total >= rules.eval_profit_target:
            # final ratio check
            if not is_consistency_ok(total, largest, rules.eval_consistency_cap):
                return EvalSimResult(False, 'consistency_breach', di, total, largest, highest_close, mll_floor)
            return EvalSimResult(True, 'target_hit', di, total, largest, highest_close, mll_floor)

    return EvalSimResult(False, 'time_limit', len(days_order), total, largest, highest_close, mll_floor)
