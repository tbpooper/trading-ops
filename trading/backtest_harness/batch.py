"""Batch runners for Lucid eval simulations.

This file provides a Monte Carlo style runner using a toy daily-PnL generator.
It is NOT alpha; it exists to validate the rule evaluation + reporting pipeline.

Once we have MNQ candles, we will replace the generator with strategy-derived trades.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import random

from trading.backtest_harness.eval_attempt import AttemptResult, simulate_eval_attempt_daily
from trading.prop.lucid_black_25k.risk_governor import Lucid25kRules


@dataclass(frozen=True)
class DailyPnlModel:
    """Toy model: each day is win or loss with fixed magnitudes."""

    p_win: float
    win: float
    loss: float


@dataclass
class BatchSummary:
    n: int
    pass_rate: float
    avg_days: float
    reasons: Dict[str, int]


def sample_attempt_day_profits(model: DailyPnlModel, max_days: int, rng: random.Random) -> List[float]:
    out: List[float] = []
    for _ in range(max_days):
        out.append(model.win if rng.random() < model.p_win else model.loss)
    return out


def run_eval_batch(
    n: int,
    model: DailyPnlModel,
    rules: Optional[Lucid25kRules] = None,
    max_days: int = 5,
    seed: int = 1,
) -> BatchSummary:
    rules = rules or Lucid25kRules()
    rng = random.Random(seed)

    passed = 0
    total_days = 0
    reasons: Dict[str, int] = {}

    for _ in range(n):
        profits = sample_attempt_day_profits(model, max_days=max_days, rng=rng)
        r: AttemptResult = simulate_eval_attempt_daily(profits, rules=rules, max_days=max_days)
        total_days += r.days
        reasons[r.reason] = reasons.get(r.reason, 0) + 1
        if r.passed:
            passed += 1

    return BatchSummary(
        n=n,
        pass_rate=passed / n if n else 0.0,
        avg_days=total_days / n if n else 0.0,
        reasons=reasons,
    )
