"""GitHub Actions runner: bounded search that writes artifacts.

Loads MNQ 5m TradingView-export CSV and runs a random search over one strategy family.
Outputs the best config found under safety constraint.

Artifacts written to ./artifacts:
- best.json (best config + metrics)
- summary.json (run summary)
"""

from __future__ import annotations

import json
import os
import random
import time

from trading.backtest_harness.days_to_pass import days_to_pass_distribution
from trading.backtest_harness.strategy_v2 import (
    Breakout,
    Filters,
    Governor,
    Params as V2Params,
    generate_trades as v2_trades,
)
from trading.backtest_harness.tv_csv import load_tradingview_ohlc_csv


def run_v2(candles, rng: random.Random, max_seconds: float):
    space = {
        "spread": [1.0, 1.5, 2.0, 2.5, 3.0],
        "max_trades": [2, 3, 4],
        "cool": [2, 4, 6, 8],
        "dls": [200.0, 250.0, 300.0],
        "rr": [0.75, 1.0, 1.25],          # v2 rr default-ish range
        "atr_mult": [1.5, 2.0, 2.5],
        "risk": [100, 150, 200],
        "bo_lb": [10, 15, 20],
        "max_losses": [2, 3],
    }

    best = None
    iters = 0
    t0 = time.time()

    while time.time() - t0 < max_seconds:
        iters += 1
        cfg = {k: rng.choice(v) for k, v in space.items()}

        p = V2Params(
            rr=float(cfg["rr"]),
            atr_mult=float(cfg["atr_mult"]),
            risk_per_trade_dollars=float(cfg["risk"]),
            filters=Filters(
                min_ema_spread_points=float(cfg["spread"]),
            ),
            governor=Governor(
                max_trades_per_day=int(cfg["max_trades"]),
                cooldown_bars_after_loss=int(cfg["cool"]),
                daily_loss_stop=float(cfg["dls"]),
                max_losses_per_day=int(cfg["max_losses"]),
            ),
            breakout=Breakout(
                enabled=True,
                lookback=int(cfg["bo_lb"]),
            ),
        )

        out = days_to_pass_distribution(
            candles,
            lambda sub: v2_trades(sub, p),
            daily_profit_cap=750.0,
            daily_loss_cap=300.0,
            max_days=30,
        )

        w = out["windows"]
        pass5 = sum(v for d, v in out["pass_days_hist"].items() if d <= 5) / w
        mll = out["outcomes"].get("mll_breach", 0) / w
        timeout = out["outcomes"].get("timeout", 0) / w

        if mll > 0.05:
            continue

        score = (pass5, -timeout)
        cand = {
            "strategy": "v2",
            "score": score,
            "pass5_rate": pass5,
            "mll_rate": mll,
            "timeout_rate": timeout,
            "cfg": cfg,
            "outcomes": out["outcomes"],
            "pass_days_hist": out["pass_days_hist"],
        }

        if best is None or tuple(cand["score"]) > tuple(best["score"]):
            best = cand

    return best, iters, round(time.time() - t0, 2)


def main():
    os.makedirs("artifacts", exist_ok=True)

    data_csv = os.environ.get("DATA_CSV", "trading/data/inbound/mnq1_5m_tv_unix.csv")
    max_seconds = float(os.environ.get("MAX_SECONDS", "1200"))
    seed = int(os.environ.get("SEED", "1337"))

    candles = load_tradingview_ohlc_csv(data_csv)
    rng = random.Random(seed)

    best, iters, elapsed = run_v2(candles, rng, max_seconds=max_seconds)

    summary = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_csv": data_csv,
        "seed": seed,
        "max_seconds": max_seconds,
        "iters": iters,
        "elaps
