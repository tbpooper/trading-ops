"""Convert trade list into daily realized PnL + EOD closes for Lucid simulation.

We need:
- day_profits: realized PnL per day
- day_closes: end-of-day account balances (closing)

We assume all trading happens on the same account, starting at rules.start_balance.
EOD close is prior close + day profit (since we don't model fees/slippage yet).

Date bucketing uses UTC day boundaries because TradingView export timestamps are UTC.
This is fine for initial testing; we can later switch to CME session boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from trading.backtest_harness.strategy_v0 import Trade
from trading.prop.lucid_black_25k.risk_governor import Lucid25kRules


def utc_day_key(ts: int) -> str:
    d = datetime.fromtimestamp(ts, tz=timezone.utc)
    return d.strftime('%Y-%m-%d')


def aggregate_daily(trades: List[Trade]) -> Dict[str, float]:
    by: Dict[str, float] = {}
    for t in trades:
        k = utc_day_key(t.exit_ts)
        by[k] = by.get(k, 0.0) + t.pnl_dollars
    return dict(sorted(by.items()))


def to_day_profits_and_closes(trades: List[Trade], rules: Lucid25kRules | None = None) -> Tuple[List[float], List[float], List[str]]:
    rules = rules or Lucid25kRules()
    daily = aggregate_daily(trades)
    keys = list(daily.keys())
    profits = [daily[k] for k in keys]

    closes: List[float] = []
    bal = rules.start_balance
    for p in profits:
        bal += p
        closes.append(bal)

    return profits, closes, keys
