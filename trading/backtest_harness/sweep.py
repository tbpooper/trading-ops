"""Parameter sweep utilities.

For now we sweep the toy DailyPnlModel to validate the sweep/reporting pipeline.
Later this will sweep strategy parameters on real MNQ candle data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from trading.backtest_harness.batch import DailyPnlModel, run_eval_batch


@dataclass(frozen=True)
class SweepRow:
    p_win: float
    win: float
    loss: float
    pass_rate: float
    avg_days: float


def sweep_daily_model(
    n: int = 2000,
    p_win_vals: List[float] = [0.50, 0.55, 0.60, 0.65],
    win_vals: List[float] = [200, 250, 300, 350],
    loss_vals: List[float] = [-150, -200, -250],
    seed: int = 42,
) -> List[SweepRow]:
    rows: List[SweepRow] = []
    for p in p_win_vals:
        for w in win_vals:
            for l in loss_vals:
                s = run_eval_batch(n=n, model=DailyPnlModel(p_win=p, win=w, loss=l), seed=seed)
                rows.append(SweepRow(p, w, l, s.pass_rate, s.avg_days))

    rows.sort(key=lambda r: (r.pass_rate, -r.avg_days), reverse=True)
    return rows


def format_top(rows: List[SweepRow], k: int = 10) -> str:
    lines = ["top configs (toy daily-PnL):"]
    for r in rows[:k]:
        lines.append(
            f"p_win={r.p_win:.2f} win={r.win:.0f} loss={r.loss:.0f} -> pass_rate={r.pass_rate:.3f} avg_days={r.avg_days:.2f}"
        )
    return "\n".join(lines)
