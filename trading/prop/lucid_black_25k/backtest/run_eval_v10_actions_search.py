"""GitHub Actions runner: bounded search for Eval v10 (pass<=4 focus).

Reads:
- DATA_CSV: path to TradingView CSV
- MAX_SECONDS: time budget

Writes:
- artifacts/eval_v10_best.json

No external deps.
"""

from __future__ import annotations

import json
import os
import random
import time

from trading.backtest_harness.tv_csv import load_tradingview_ohlc_csv
from trading.prop.lucid_black_25k.backtest.eval_v10_eval import compute_day_pnl_for_sessions, score_eval_pass


SPLIT_WINDOWS_UTC = [(14 * 60 + 30, 15 * 60 + 15), (15 * 60 + 45, 17 * 60)]


def main() -> None:
    data_csv = os.environ.get("DATA_CSV", "trading/data/inbound/mnq1_5m_tv_unix.csv")
    max_seconds = float(os.environ.get("MAX_SECONDS", "1500"))
    seed = int(os.environ.get("SEED", "777"))

    random.seed(seed)

    candles = load_tradingview_ohlc_csv(data_csv)

    # Base params (eval rules: no DLL, but keep governor fields inert)
    base = {
        "risk": 250.0,
        "mt": 4,
        "atr_min": 8.0,
        "dev": 0.4,
        "stop": 0.5,
        "trail": 0.1,
        "tp": 1.2,
        "hold": 18,
        "use_spread": False,
        "max_spread": 2.0,
        "require_reversal_bar": True,
        "max_micros": 20,
        "ml": 99,
        "dls": 1e9,
        "cool": 0,
        "dpt_base": 1e9,
        "dpt_press": 1e9,
        "stop1": False,
    }

    # Search space (tight)
    space = {
        "risk": [150.0, 200.0, 250.0, 300.0],
        "mt": [3, 4, 5, 6],
        "dev": [0.24, 0.28, 0.32, 0.36, 0.4, 0.44],
        "atr_min": [6.0, 8.0, 10.0, 12.0],
        "stop": [0.45, 0.5, 0.55, 0.6],
        "trail": [0.05, 0.1, 0.15, 0.2],
        "tp": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        "hold": [14, 18, 22],
    }

    best = None
    best_score = None
    best_metrics = None

    t0 = time.time()
    iters = 0

    while time.time() - t0 < max_seconds:
        iters += 1
        p = dict(base)
        for k, vals in space.items():
            p[k] = random.choice(vals)

        day_pnl = compute_day_pnl_for_sessions(candles, p, SPLIT_WINDOWS_UTC)
        metrics = score_eval_pass(day_pnl)

        score = (
            metrics["pass_leq4_rate"],
            -metrics["breach_rate"],
            -metrics["timeout_rate"],
            metrics["pass_leq5_rate"],
        )

        if best_score is None or score > best_score:
            best_score = score
            best = p
            best_metrics = metrics

    out = {
        "seed": seed,
        "iters": iters,
        "best_score": best_score,
        "best_params": best,
        "best_metrics": best_metrics,
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/eval_v10_best.json", "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("BEST", best_metrics)


if __name__ == "__main__":
    main()
