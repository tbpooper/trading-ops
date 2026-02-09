"""LucidBlack 25k eval attempt simulator (v0).

Inputs are synthetic (daily realized PnL + daily closing balance), allowing us to
validate Lucid rule enforcement before integrating real trade-by-trade PnL.

This is intentionally minimal but produces Lucid-style pass/fail reasons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

from trading.prop.lucid_black_25k.risk_governor import (
    Lucid25kRules,
    is_consistency_ok,
    step_eod_drawdown,
)


FailReason = Literal[
    "target_hit",
    "time_limit",
    "consistency_breach",
    "mll_breach",
]


@dataclass
class AttemptResult:
    passed: bool
    reason: FailReason
    days: int
    total_profit: float
    largest_day: float
    highest_close: float
    mll_floor: float


def simulate_eval_attempt_daily(
    day_profits: List[float],
    day_closes: Optional[List[float]] = None,
    rules: Optional[Lucid25kRules] = None,
    max_days: int = 5,
) -> AttemptResult:
    """Simulate an eval attempt (<= max_days) using daily realized PnL and EOD closes.

    - day_profits: realized PnL per trading day.
    - day_closes: end-of-day account balances (closing). If omitted, we assume
      close = prior_close + day_profit.

    Rules enforced each day:
    - EOD drawdown/MLL breach based on trailing floor
    - Consistency (largest day / total profit <= 60%)
    - Profit target hit
    """

    rules = rules or Lucid25kRules()
    n = min(len(day_profits), max_days)

    closes: List[float] = []
    if day_closes is None:
        bal = rules.start_balance
        for i in range(n):
            bal += day_profits[i]
            closes.append(bal)
    else:
        closes = day_closes[:n]

    total = 0.0
    largest = 0.0
    highest_close = rules.start_balance
    mll_floor = highest_close - rules.max_loss_limit

    for i in range(n):
        p = day_profits[i]
        c = closes[i]

        total += p
        largest = max(largest, p)

        highest_close, mll_floor, breached = step_eod_drawdown(rules, highest_close, c)
        if breached:
            return AttemptResult(False, "mll_breach", i + 1, total, largest, highest_close, mll_floor)

        # Eval consistency handling:
        # In practice, Lucid evaluates consistency against the *profit target* context.
        # The strict ratio check largest_day/total_profit at early stages would
        # falsely fail almost every run (day1 profit => ratio~1.0). So for eval we
        # enforce the implied daily cap: largest day must be <= cap * profit_target.
        eval_day_cap = rules.eval_consistency_cap * rules.eval_profit_target
        if largest > eval_day_cap:
            return AttemptResult(
                False,
                "consistency_breach",
                i + 1,
                total,
                largest,
                highest_close,
                mll_floor,
            )

        if total >= rules.eval_profit_target:
            # Final consistency check at/after target (conservative)
            if not is_consistency_ok(total, largest, rules.eval_consistency_cap):
                return AttemptResult(
                    False,
                    "consistency_breach",
                    i + 1,
                    total,
                    largest,
                    highest_close,
                    mll_floor,
                )
            return AttemptResult(True, "target_hit", i + 1, total, largest, highest_close, mll_floor)

    return AttemptResult(False, "time_limit", n, total, largest, highest_close, mll_floor)
