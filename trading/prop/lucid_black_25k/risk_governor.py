"""LucidBlack 25K — Risk Governor (Eval + Funded)

Goal: centralize Lucid rule math as pure functions so we can reuse it in:
- local backtest harness
- alert/auto-trade gatekeeping
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class Mode(str, Enum):
    EVAL_PASS = "EVAL_PASS"
    FUNDED_PAYOUT = "FUNDED_PAYOUT"


@dataclass(frozen=True)
class Lucid25kRules:
    max_loss_limit: float = 1000.0

    start_balance: float = 25000.0
    initial_trail_balance: float = 26100.0
    locked_mll_balance: float = 25100.0

    eval_profit_target: float = 1250.0
    eval_consistency_cap: float = 0.60

    payout_profit_goal: float = 1500.0
    funded_consistency_cap: float = 0.40

    max_micros_absolute: int = 20
    max_minis_absolute: int = 2

    funded_micros_tier1_cap: int = 10
    funded_micros_tier2_cap: int = 20
    funded_tier2_threshold_profit: float = 1000.0


def consistency_ratio(total_profit: float, largest_day_profit: float) -> float:
    if total_profit <= 0:
        return float("inf")
    return largest_day_profit / total_profit


def is_consistency_ok(total_profit: float, largest_day_profit: float, cap: float) -> bool:
    return consistency_ratio(total_profit, largest_day_profit) <= cap


def eod_trailing_floor(rules: Lucid25kRules, highest_close: float) -> float:
    raw = highest_close - rules.max_loss_limit
    return min(raw, rules.locked_mll_balance)


def update_highest_close(prev_highest_close: float, eod_close: float) -> float:
    return max(prev_highest_close, eod_close)


def step_eod_drawdown(
    rules: Lucid25kRules,
    prev_highest_close: float,
    eod_close: float,
) -> Tuple[float, float, bool]:
    new_highest = update_highest_close(prev_highest_close, eod_close)
    floor = eod_trailing_floor(rules, new_highest)
    breached = eod_close <= floor
    return new_highest, floor, breached
