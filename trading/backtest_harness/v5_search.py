"""Search loop for Strategy v5 (regime-first playbook) with aggressive checkpointing.

Objective:
- maximize pass<=5
- keep MLL fail <= 5%

Writes:
- best_v5.json
- best_v5_ckpt.json (every iteration)

Run from repo root or anywhere.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from trading.backtest_harness.tv_csv import load_tradingview_ohlc_csv
from trading.backtest_harness.days_to_pass import days_to_pass_distribution
from trading.backtest_harness.strategy_v5_regime_drive import Params, Window, Regime, Entry, Exits, Governor, generate_trades


def pass5_rate(out: dict) -> float:
    w = out["windows"]
    p5 = sum(v for d, v in out["pass_days_hist"].items() if int(d) <= 5)
    return p5 / w if w else 0.0


def main():
    candles = load_tradingview_ohlc_csv("trading/data/inbound/mnq1_5m_tv_unix.csv")

    # Trading window 7:30am–10pm ET (12:30–03:00 UTC) implemented as 2 windows
    windows = (
        Window(12 * 60 + 30, 15 * 60),
        Window(18 * 60, 3 * 60),
    )

    space = {
        "min_spread": [0.5, 1.0, 1.5, 2.0, 3.0],
        "atr_min": [3.0, 5.0, 8.0, 10.0, 12.0],
        "lookback": [6, 9, 12, 18],
        "use_retest": [True, False],
        "retest_deadline": [3, 6, 9],
        "eps": [2.0, 3.0, 5.0, 8.0],
        "stop_atr": [0.75, 1.0, 1.25, 1.5, 2.0],
        "trail_atr": [0.5, 0.75, 1.0, 1.25, 1.5],
        "max_hold": [24, 36, 48, 72],
        "tp_atr": [None, 2.0, 3.0, 4.0],
        "max_trades": [1, 2, 3],
        "max_losses": [1, 2],
        "daily_loss": [150.0, 200.0, 250.0, 300.0],
        "cool": [2, 4, 6, 10],
        "risk": [75.0, 100.0, 150.0, 200.0],
    }

    out_best = "trading/backtest_harness/best_v5.json"
    out_ckpt = "trading/backtest_harness/best_v5_ckpt.json"

    best = None
    n = 0
    t0 = time.time()

    while True:
        n += 1
        cfg = {k: random.choice(v) for k, v in space.items()}

        p = Params(
            risk_per_trade_dollars=float(cfg["risk"]),
            windows=windows,
            regime=Regime(min_spread_points=float(cfg["min_spread"]), atr_min_points=float(cfg["atr_min"])),
            entry=Entry(
                lookback=int(cfg["lookback"]),
                use_retest=bool(cfg["use_retest"]),
                retest_deadline_bars=int(cfg["retest_deadline"]),
                retest_epsilon_points=float(cfg["eps"]),
            ),
            exits=Exits(
                stop_atr_mult=float(cfg["stop_atr"]),
                trail_atr_mult=float(cfg["trail_atr"]),
                max_hold_bars=int(cfg["max_hold"]),
                tp_atr_mult=cfg["tp_atr"],
            ),
            governor=Governor(
                max_trades_per_day=int(cfg["max_trades"]),
                max_losses_per_day=int(cfg["max_losses"]),
                daily_loss_stop=float(cfg["daily_loss"]),
                cooldown_bars_after_loss=int(cfg["cool"]),
            ),
        )

        out = days_to_pass_distribution(
            candles,
            lambda sub: generate_trades(sub, p),
            daily_profit_cap=750.0,
            daily_loss_cap=300.0,
            max_days=30,
        )

        w = out["windows"]
        p5 = pass5_rate(out)
        mll = out["outcomes"].get("mll_breach", 0) / w if w else 0.0
        timeout = out["outcomes"].get("timeout", 0) / w if w else 0.0

        # hard constraint: fail <= 5%
        ok = (mll <= 0.05)

        # checkpoint every iteration
        with open(out_ckpt, "w") as f:
            json.dump(
                {
                    "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "iterations": n,
                    "elapsed_s": round(time.time() - t0, 1),
                    "ok": ok,
                    "current": {
                        "pass5_rate": p5,
                        "mll_rate": mll,
                        "timeout_rate": timeout,
                        "cfg": cfg,
                        "outcomes": out["outcomes"],
                        "pass_days_hist": out["pass_days_hist"],
                    },
                    "best": {
                        "pass5_rate": (best[1] if best else None),
                        "mll_rate": (best[2] if best else None),
                    },
                },
                f,
                indent=2,
            )

        if not ok:
            continue

        score = (p5, -timeout)
        cand = (score, p5, mll, timeout, cfg, out)

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

        if n % 50 == 0 and best is not None:
            print(
                f"n={n} best p5={best[1]:.3f} mll={best[2]:.3f} timeout={best[3]:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
