"""Random search for Strategy v4 (retest + multi-TF) aiming to maximize pass<=5.

Uses 1m primary series with 15m regime. (5m reserved for future structure levels).
Saves best + checkpoint.

NOTE: This is a brute-force heuristic search. Keep parameter space modest to avoid
machine instability.
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

from trading.backtest_harness.tv_csv_multi import load_tv_csv
from trading.backtest_harness.days_to_pass_multi import days_to_pass_distribution_multi
from trading.backtest_harness.strategy_v4_retest import Params, Session, Governor, Regime, Entry, Exits, generate_trades


def pass5_rate(out: dict) -> float:
    w = out["windows"]
    p5 = sum(v for d, v in out["pass_days_hist"].items() if int(d) <= 5)
    return p5 / w if w else 0.0


def main():
    c1 = load_tv_csv("trading/data/inbound/CME_MINI_MNQ1_1.csv")
    c5 = load_tv_csv("trading/data/inbound/CME_MINI_MNQ1_5.csv")
    c15 = load_tv_csv("trading/data/inbound/CME_MINI_MNQ1_15.csv")

    sess = Session(enabled=True, start_min_utc=12 * 60 + 30, end_min_utc=3 * 60)

    space = {
        # governors
        "max_trades": [1, 2, 3],
        "max_losses": [1, 2],
        "daily_loss": [150.0, 200.0, 250.0, 300.0],
        "cooldown": [5, 10, 20, 30],

        # regime (15m)
        "ema_fast": [10, 20, 30],
        "ema_slow": [40, 50, 60],
        "min_spread": [1.5, 2.0, 3.0, 4.0],
        "atr_min": [5.0, 8.0, 10.0, 12.0],

        # entry
        "level_lb": [30, 60, 120, 240],
        "deadline": [30, 60, 120, 180],
        "eps": [2.0, 3.0, 5.0, 8.0],

        # exits
        "stop_atr": [1.0, 1.25, 1.5, 2.0],
        "tp_atr": [1.5, 2.0, 2.5, 3.0, 4.0],
        "be_atr": [None, 0.6, 0.8, 1.0],
        "max_hold": [120, 240, 360, 480],

        # risk
        "risk": [50.0, 75.0, 100.0, 150.0, 200.0],
    }

    out_best = "trading/backtest_harness/best_v4.json"
    out_ckpt = "trading/backtest_harness/best_v4_ckpt.json"

    best = None
    n = 0
    t0 = time.time()

    def pick():
        c = {k: random.choice(v) for k, v in space.items()}
        if c["ema_fast"] >= c["ema_slow"]:
            c["ema_fast"], c["ema_slow"] = min(c["ema_fast"], c["ema_slow"]), max(c["ema_fast"], c["ema_slow"])
        return c

    while True:
        n += 1
        cfg = pick()

        p = Params(
            risk_per_trade_dollars=float(cfg["risk"]),
            session=sess,
            governor=Governor(
                max_trades_per_day=int(cfg["max_trades"]),
                max_losses_per_day=int(cfg["max_losses"]),
                daily_loss_stop=float(cfg["daily_loss"]),
                cooldown_bars_after_loss=int(cfg["cooldown"]),
            ),
            regime=Regime(
                ema_fast=int(cfg["ema_fast"]),
                ema_slow=int(cfg["ema_slow"]),
                min_spread_points=float(cfg["min_spread"]),
                atr_len=14,
                atr_min_points=float(cfg["atr_min"]),
            ),
            entry=Entry(
                level_lookback=int(cfg["level_lb"]),
                retest_deadline_bars=int(cfg["deadline"]),
                retest_epsilon_points=float(cfg["eps"]),
            ),
            exits=Exits(
                atr_len=14,
                atr_mult_stop=float(cfg["stop_atr"]),
                tp_atr_mult=float(cfg["tp_atr"]),
                be_atr_mult=cfg["be_atr"],
                max_hold_bars=int(cfg["max_hold"]),
            ),
        )

        def gen(sub1, aux):
            sub15 = aux["15m"]
            if len(sub15) < 200:
                return []
            return generate_trades(sub1, sub15, p)

        out = days_to_pass_distribution_multi(
            c1,
            {"5m": c5, "15m": c15},
            gen,
            daily_profit_cap=750.0,
            daily_loss_cap=300.0,
            max_days=30,
        )

        w = out["windows"]
        p5 = pass5_rate(out)
        mll = out["outcomes"].get("mll_breach", 0) / w if w else 0.0
        timeout = out["outcomes"].get("timeout", 0) / w if w else 0.0

        # constraint: keep blowups low
        if mll > 0.05:
            continue

        score = (p5, -timeout)
        cand = (score, p5, mll, timeout, cfg, out)

        # checkpoint current
        with open(out_ckpt, "w") as f:
            json.dump(
                {
                    "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "iterations": n,
                    "elapsed_s": round(time.time() - t0, 1),
                    "current": {
                        "score": score,
                        "pass5_rate": p5,
                        "mll_rate": mll,
                        "timeout_rate": timeout,
                        "cfg": cfg,
                        "outcomes": out["outcomes"],
                        "pass_days_hist": out["pass_days_hist"],
                    },
                },
                f,
                indent=2,
            )

        if best is None or cand[0] > best[0]:
            best = cand
            with open(out_best, "w") as f:
                json.dump(
                    {
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
                    },
                    f,
                    indent=2,
                )

        if n % 10 == 0 and best is not None:
            print(
                f"n={n} current p5={p5:.3f} mll={mll:.3f} to={timeout:.3f} | best p5={best[1]:.3f} mll={best[2]:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
