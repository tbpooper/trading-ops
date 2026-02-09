"""Overnight search loop for Strategy v2.

Optimizes for:
- maximize pass rate by day <= 5
- minimize MLL breach rate
under Lucid 25k eval constraints.

Writes best results periodically to a JSON file.
"""

from __future__ import annotations

# Allow running as a script even when cwd isn't repo root.
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import json
import random
import time

from trading.backtest_harness.tv_csv import load_tradingview_ohlc_csv
from trading.backtest_harness.days_to_pass import days_to_pass_distribution
from trading.backtest_harness.strategy_v2 import Params, Filters, Governor, Breakout, ORB, Session, Exits, generate_trades


def metric(out: dict) -> tuple[float, float, float, float]:
    windows = out["windows"]
    pass5 = sum(v for d, v in out["pass_days_hist"].items() if int(d) <= 5)
    pass5r = pass5 / windows
    mll = out["outcomes"].get("mll_breach", 0) / windows
    timeout = out["outcomes"].get("timeout", 0) / windows
    overall = out["pass_rate"]
    # Objective: prioritize passing within 5 days.
    # We'll still track MLL/timeout for constraints/ranking.
    score = pass5r - 0.2 * timeout + 0.05 * overall
    return score, pass5r, mll, timeout


def main():
    candles = load_tradingview_ohlc_csv("trading/data/inbound/mnq1_5m_tv_unix.csv")

    # 7:30am–10pm ET ~= 12:30–03:00 UTC (wrap midnight)
    sess = Session(enabled=True, start_min_utc=12 * 60 + 30, end_min_utc=3 * 60)

    choices = {
        "spread": [None, 0.25, 0.5, 1.0, 2.0, 3.0],
        "max_trades": [2, 3, 4, 6],
        "cool": [0, 2, 4, 6, 8, 12],
        "dls": [150.0, 200.0, 250.0, 300.0],
        "rr": [1.25, 1.5, 2.0, 2.5],
        "atr_mult": [0.75, 1.0, 1.25, 1.5, 2.0],
        "risk": [75, 100, 150, 200, 250, 300],
        "bo_lb": [10, 15, 20, 30],
        "max_losses": [1, 2, 3],
    }

    best = None
    out_path = "trading/backtest_harness/best_v2_overnight.json"
    checkpoint_path = "trading/backtest_harness/best_v2_overnight_ckpt.json"

    n = 0
    t0 = time.time()
    while True:
        n += 1
        cfg = {k: random.choice(v) for k, v in choices.items()}
        p = Params(
            risk_per_trade_dollars=cfg["risk"],
            atr_mult=cfg["atr_mult"],
            rr=cfg["rr"],
            filters=Filters(min_ema_spread_points=cfg["spread"]),
            governor=Governor(
                max_trades_per_day=cfg["max_trades"],
                cooldown_bars_after_loss=cfg["cool"],
                daily_loss_stop=cfg["dls"],
                max_losses_per_day=cfg["max_losses"],
            ),
            breakout=Breakout(enabled=True, lookback=cfg["bo_lb"], range_atr_mult=0.0),
            orb=ORB(enabled=False),
            session=sess,
            exits=Exits(move_sl_to_be_at_r_multiple=None),
        )

        out = days_to_pass_distribution(
            candles,
            lambda sub: generate_trades(sub, p),
            daily_profit_cap=750.0,
            daily_loss_cap=300.0,
            max_days=30,
        )

        score, pass5r, mll, timeout = metric(out)

        # Hard safety constraint to avoid "fast but blowup" configs.
        # Tuneable: we start with 0.15 (<= ~7/47 blowups).
        if mll > 0.15:
            continue

        cand = (score, pass5r, mll, timeout, cfg, out)

        # Always checkpoint current result so a reboot doesn't erase all progress.
        ckpt = {
            "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "iterations": n,
            "elapsed_s": round(time.time() - t0, 1),
            "current": {
                "score": score,
                "pass5_rate": pass5r,
                "mll_rate": mll,
                "timeout_rate": timeout,
                "cfg": cfg,
                "outcomes": out["outcomes"],
                "pass_days_hist": out["pass_days_hist"],
            },
        }
        with open(checkpoint_path, "w") as f:
            json.dump(ckpt, f, indent=2)

        if best is None or cand[:4] > best[:4]:
            best = cand
            payload = {
                "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "iterations": n,
                "elapsed_s": round(time.time() - t0, 1),
                "score": best[0],
                "pass5_rate": best[1],
                "mll_rate": best[2],
                "timeout_rate": best[3],
                "cfg": best[4],
                "outcomes": best[5]["outcomes"],
                "pass_days_hist": best[5]["pass_days_hist"],
            }
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)

        if n % 10 == 0:
            # Light progress logging to stdout.
            print(
                f"n={n} current: score={score:.3f} p5={pass5r:.3f} mll={mll:.3f} to={timeout:.3f} | best: score={best[0]:.3f} p5={best[1]:.3f} mll={best[2]:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
