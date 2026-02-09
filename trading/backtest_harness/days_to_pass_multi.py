"""Days-to-pass distribution for multi-timeframe strategies.

Unlike days_to_pass_distribution (single candle series), we need to slice multiple
candle series consistently by start-day.

We use the 1m series as the primary timeline and derive start points from it.
For each start day, we slice 1m from that day onward and slice the higher TF
series to <= last ts of the 1m slice.

UTC day boundaries.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

from trading.backtest_harness.days_to_pass import build_day_index
from trading.backtest_harness.days_to_pass import simulate_path


def days_to_pass_distribution_multi(
    c_primary: Sequence,
    c_aux: Dict[str, Sequence],
    gen_trades: Callable[[List, Dict[str, List]], Sequence],
    daily_profit_cap: float = 750.0,
    daily_loss_cap: float | None = 300.0,
    max_days: int = 30,
) -> Dict:
    # build day index on primary
    days, idx = build_day_index(c_primary)

    outcomes = {"target_hit": 0, "mll_breach": 0, "consistency_breach": 0, "timeout": 0}
    pass_days_hist: Dict[int, int] = {}

    for s in range(len(days)):
        start_i = idx[days[s]][0]
        sub_primary = list(c_primary[start_i:])
        if not sub_primary:
            continue
        last_ts = sub_primary[-1].ts
        sub_aux: Dict[str, List] = {}
        for k, series in c_aux.items():
            # keep only candles up to last primary ts
            sub_aux[k] = [c for c in series if c.ts <= last_ts]

        trades = list(gen_trades(sub_primary, sub_aux))
        res = simulate_path(
            trades,
            daily_profit_cap=daily_profit_cap,
            daily_loss_cap=daily_loss_cap,
            max_days=max_days,
        )
        outcomes[res.outcome] = outcomes.get(res.outcome, 0) + 1
        if res.outcome == "target_hit":
            pass_days_hist[res.days] = pass_days_hist.get(res.days, 0) + 1

    windows = len(days)
    pass_rate = outcomes.get("target_hit", 0) / windows if windows else 0.0

    return {
        "windows": windows,
        "max_days": max_days,
        "outcomes": outcomes,
        "pass_days_hist": dict(sorted(pass_days_hist.items())),
        "pass_rate": pass_rate,
    }
