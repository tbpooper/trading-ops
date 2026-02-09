"""Rolling 5-day Lucid eval pass-rate evaluator.

Given:
- candles
- a strategy that returns closed trades

We compute rolling windows of 5 *UTC days* and measure pass/fail.

Note: UTC day boundaries are a simplification. Later we can switch to CME session days.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Sequence, Tuple

from trading.backtest_harness.tv_csv import Candle
from trading.prop.lucid_black_25k.risk_governor import Lucid25kRules
from trading.backtest_harness.eval_attempt import simulate_eval_attempt_daily, AttemptResult
from trading.backtest_harness.eval_from_trades import simulate_eval_from_trades


def day_key(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d')


@dataclass
class RollingSummary:
    windows: int
    passed: int
    pass_rate: float
    reasons: Dict[str, int]


def build_day_index(candles: Sequence[Candle]) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
    # day -> (start_idx, end_idx)
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


def rolling_eval_5day(
    candles: Sequence[Candle],
    gen_trades: Callable[[List[Candle]], Sequence],
    rules: Lucid25kRules | None = None,
    daily_profit_cap: float = 750.0,
    daily_loss_cap: float | None = None,
    use_trade_stream: bool = True,
) -> RollingSummary:
    rules = rules or Lucid25kRules()

    days, idx = build_day_index(candles)
    windows = 0
    passed = 0
    reasons: Dict[str, int] = defaultdict(int)

    for s in range(0, len(days) - 5 + 1):
        win_days = days[s : s + 5]
        start_i = idx[win_days[0]][0]
        end_i = idx[win_days[-1]][1]
        sub = list(candles[start_i : end_i + 1])

        trades = list(gen_trades(sub))

        if use_trade_stream:
            r2 = simulate_eval_from_trades(
                trades,
                rules=rules,
                max_days=5,
                daily_profit_cap=daily_profit_cap,
                daily_loss_cap=daily_loss_cap,
            )
            windows += 1
            reasons[r2.reason] += 1
            if r2.passed:
                passed += 1
            continue

        # fallback: aggregate daily (less realistic)
        daily_pnl: Dict[str, float] = defaultdict(float)
        for t in trades:
            k = day_key(t.exit_ts)
            daily_pnl[k] += float(t.pnl_dollars)

        profits: List[float] = []
        closes: List[float] = []
        bal = rules.start_balance
        for d in win_days:
            p = daily_pnl.get(d, 0.0)
            profits.append(p)
            bal += p
            closes.append(bal)

        r: AttemptResult = simulate_eval_attempt_daily(profits, closes, rules=rules, max_days=5)
        windows += 1
        reasons[r.reason] += 1
        if r.passed:
            passed += 1

    return RollingSummary(windows=windows, passed=passed, pass_rate=(passed / windows if windows else 0.0), reasons=dict(reasons))
