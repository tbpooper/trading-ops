"""Backtest/simulation skeleton for Lucid rules.

This will evolve into:
- candle loader
- signal generator
- fills / slippage model
- day-by-day rule evaluation (profit target, MLL/EOD drawdown, consistency)

For now we focus on rule plumbing, not alpha.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from trading.prop.lucid_black_25k.risk_governor import Lucid25kRules, Mode, is_consistency_ok


@dataclass(frozen=True)
class DayResult:
    day_profit: float


@dataclass
class SimResult:
    passed: bool
    reason: str
    days: int
    total_profit: float
    largest_day: float


def simulate_eval_attempt(
    day_profits: List[float],
    rules: Optional[Lucid25kRules] = None,
    max_days: int = 5,
) -> SimResult:
    """Simulate an eval attempt using *realized* per-day PnL.

    This ignores intraday path + EOD drawdown for now. It's a first scaffold to
    validate consistency + profit-target pacing.
    """

    rules = rules or Lucid25kRules()

    total = 0.0
    largest = 0.0
    for i, p in enumerate(day_profits[:max_days], start=1):
        total += p
        largest = max(largest, p)

        # Consistency constraint should hold at all times (conservative).
        if total > 0 and not is_consistency_ok(total, largest, rules.eval_consistency_cap):
            return SimResult(False, "consistency_breach", i, total, largest)

        if total >= rules.eval_profit_target:
            return SimResult(True, "target_hit", i, total, largest)

    return SimResult(False, "time_limit", min(len(day_profits), max_days), total, largest)
