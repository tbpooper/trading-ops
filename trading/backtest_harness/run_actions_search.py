"""GitHub Actions runner: bounded search with strategy-family rotation.

Writes artifacts to ./artifacts:
- summary.json
- best.json (best candidate found)

Env:
- DATA_CSV: path to CSV
- MAX_SECONDS: time budget
- SEED: RNG seed
- FAMILY: v2|v5|v6|v7 (default: v2)

Design: dependency-free (stdlib only).
"""

from __future__ import annotations

import json
import os
import random
import time

from trading.backtest_harness.days_to_pass import days_to_pass_distribution
from trading.backtest_harness.tv_csv import load_tradingview_ohlc_csv


def score_from_out(out: dict) -> dict:
    w = out["windows"]
    pass5 = sum(v for d, v in out["pass_days_hist"].items() if int(d) <= 5) / w
    mll = out["outcomes"].get("mll_breach", 0) / w
    timeout = out["outcomes"].get("timeout", 0) / w
    return {"pass5_rate": pass5, "mll_rate": mll, "timeout_rate": timeout}


def run_family_v2(candles, rng: random.Random, max_seconds: float):
    from trading.backtest_harness.strategy_v2 import Breakout, Filters, Governor, Params as V2Params, generate_trades

    space = {
        "spread": [0.5, 1.0, 1.5, 2.0, 2.5],
        "max_trades": [2, 3, 4],
        "cool": [2, 4, 6, 8],
        "dls": [200.0, 250.0, 300.0],
        "rr": [0.75, 1.0, 1.25, 1.5],
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
            filters=Filters(min_ema_spread_points=float(cfg["spread"])),
            governor=Governor(
                max_trades_per_day=int(cfg["max_trades"]),
                cooldown_bars_after_loss=int(cfg["cool"]),
                daily_loss_stop=float(cfg["dls"]),
                max_losses_per_day=int(cfg["max_losses"]),
            ),
            breakout=Breakout(enabled=True, lookback=int(cfg["bo_lb"])),
        )
        out = days_to_pass_distribution(
            candles,
            lambda sub: generate_trades(sub, p),
            daily_profit_cap=750.0,
            daily_loss_cap=300.0,
            max_days=30,
        )
        s = score_from_out(out)
        if s["mll_rate"] > 0.05:
            continue
        cand = {
            "family": "v2",
            "cfg": cfg,
            "outcomes": out["outcomes"],
            "pass_days_hist": out["pass_days_hist"],
            **s,
            "score": [s["pass5_rate"], -s["timeout_rate"]],
        }
        if best is None or tuple(cand["score"]) > tuple(best["score"]):
            best = cand
    return best, iters, round(time.time() - t0, 2)


def run_family_v5(candles, rng: random.Random, max_seconds: float):
    from trading.backtest_harness.strategy_v5_regime_drive import Params, Window, Regime, Entry, Exits, Governor, generate_trades

    windows = (Window(12 * 60 + 30, 3 * 60),)
    space = {
        "min_spread": [0.5, 1.0, 1.5, 2.0],
        "atr_min": [5.0, 8.0, 10.0, 12.0],
        "lookback": [6, 9, 12, 18],
        "use_retest": [False, True],
        "retest_deadline": [3, 6, 9],
        "eps": [2.0, 3.0, 5.0, 8.0],
        "stop_atr": [0.75, 1.0, 1.25],
        "trail_atr": [0.5, 0.75, 1.0],
        "max_hold": [36, 48, 72],
        "tp_atr": [None, 2.0, 3.0, 4.0],
        "max_trades": [2, 3, 4],
        "max_losses": [1, 2],
        "daily_loss": [200.0, 250.0, 300.0],
        "cool": [0, 2, 4, 6],
        "risk": [100.0, 150.0, 200.0],
    }

    best = None
    iters = 0
    t0 = time.time()
    while time.time() - t0 < max_seconds:
        iters += 1
        cfg = {k: rng.choice(v) for k, v in space.items()}
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
        s = score_from_out(out)
        if s["mll_rate"] > 0.05:
            continue
        cand = {
            "family": "v5",
            "cfg": cfg,
            "outcomes": out["outcomes"],
            "pass_days_hist": out["pass_days_hist"],
            **s,
            "score": [s["pass5_rate"], -s["timeout_rate"]],
        }
        if best is None or tuple(cand["score"]) > tuple(best["score"]):
            best = cand
    return best, iters, round(time.time() - t0, 2)


def run_family_v6(candles, rng: random.Random, max_seconds: float):
    from trading.backtest_harness.strategy_v6_orb_drive import Params, TrendFilter, Exits, Governor, generate_trades

    space = {
        "orb_minutes": [15, 30, 45],
        "buffer": [0.5, 1.0, 1.5, 2.0],
        "trend": [True, False],
        "min_spread": [0.0, 1.0, 2.0, 3.0],
        "stop": [0.75, 1.0, 1.25],
        "trail": [0.5, 0.75, 1.0],
        "tp": [2.0, 3.0, 4.0, None],
        "hold": [36, 48, 72],
        "risk": [100.0, 150.0, 200.0],
        "mt": [1, 2],
        "ml": [1, 2],
        "dls": [200.0, 250.0, 300.0],
    }

    best = None
    iters = 0
    t0 = time.time()
    while time.time() - t0 < max_seconds:
        iters += 1
        cfg = {k: rng.choice(v) for k, v in space.items()}
        p = Params(
            orb_minutes=int(cfg["orb_minutes"]),
            buffer_points=float(cfg["buffer"]),
            risk_per_trade_dollars=float(cfg["risk"]),
            trend=TrendFilter(enabled=bool(cfg["trend"]), min_spread_points=float(cfg["min_spread"])),
            exits=Exits(stop_atr_mult=float(cfg["stop"]), trail_atr_mult=float(cfg["trail"]), tp_atr_mult=cfg["tp"], max_hold_bars=int(cfg["hold"])),
            governor=Governor(max_trades_per_day=int(cfg["mt"]), max_losses_per_day=int(cfg["ml"]), daily_loss_stop=float(cfg["dls"])),
        )
        out = days_to_pass_distribution(
            candles,
            lambda sub: generate_trades(sub, p),
            daily_profit_cap=750.0,
            daily_loss_cap=300.0,
            max_days=30,
        )
        s = score_from_out(out)
        if s["mll_rate"] > 0.05:
            continue
        cand = {
            "family": "v6",
            "cfg": cfg,
            "outcomes": out["outcomes"],
            "pass_days_hist": out["pass_days_hist"],
            **s,
            "score": [s["pass5_rate"], -s["timeout_rate"]],
        }
        if best is None or tuple(cand["score"]) > tuple(best["score"]):
            best = cand
    return best, iters, round(time.time() - t0, 2)


def run_family_v7(candles, rng: random.Random, max_seconds: float):
    from trading.backtest_harness.strategy_v7_regime_switch import Params, Regime, Drive, Chop, Exits, Governor, generate_trades

    space = {
        "atr_min": [5.0, 8.0, 10.0, 12.0],
        "drive_spread": [1.0, 2.0, 3.0],
        "drive_lb": [9, 12, 18],
        "drive_buf": [0.0, 0.5, 1.0],
        "chop_dev": [0.6, 0.8, 1.0, 1.2],
        "stop": [0.75, 1.0, 1.25],
        "trail": [0.5, 0.75, 1.0],
        "tp": [2.0, 2.5, 3.0, None],
        "hold": [36, 48, 60, 72],
        "risk": [100.0, 150.0, 200.0],
        "mt": [2, 3, 4],
        "ml": [1, 2],
        "dls": [200.0, 250.0, 300.0],
        "cool": [0, 2, 4, 6],
    }

    best = None
    iters = 0
    t0 = time.time()
    while time.time() - t0 < max_seconds:
        iters += 1
        cfg = {k: rng.choice(v) for k, v in space.items()}
        p = Params(
            risk_per_trade_dollars=float(cfg["risk"]),
            regime=Regime(atr_min_points=float(cfg["atr_min"]), drive_spread_points=float(cfg["drive_spread"])),
            drive=Drive(lookback=int(cfg["drive_lb"]), buffer_points=float(cfg["drive_buf"])),
            chop=Chop(dev_atr_mult=float(cfg["chop_dev"])),
            exits=Exits(stop_atr_mult=float(cfg["stop"]), trail_atr_mult=float(cfg["trail"]), tp_atr_mult=cfg["tp"], max_hold_bars=int(cfg["hold"])),
            governor=Governor(max_trades_per_day=int(cfg["mt"]), max_losses_per_day=int(cfg["ml"]), daily_loss_stop=float(cfg["dls"]), cooldown_bars_after_loss=int(cfg["cool"])),
        )
        out = days_to_pass_distribution(
            candles,
            lambda sub: generate_trades(sub, p),
            daily_profit_cap=750.0,
            daily_loss_cap=300.0,
            max_days=30,
        )
        s = score_from_out(out)
        if s["mll_rate"] > 0.05:
            continue
        cand = {
            "family": "v7",
            "cfg": cfg,
            "outcomes": out["outcomes"],
            "pass_days_hist": out["pass_days_hist"],
            **s,
            "score": [s["pass5_rate"], -s["timeout_rate"]],
        }
        if best is None or tuple(cand["score"]) > tuple(best["score"]):
            best = cand

    return best, iters, round(time.time() - t0, 2)


def main():
    os.makedirs("artifacts", exist_ok=True)

    data_csv = os.environ.get("DATA_CSV", "trading/data/inbound/mnq1_5m_tv_unix.csv")
    max_seconds = float(os.environ.get("MAX_SECONDS", "1200"))
    seed = int(os.environ.get("SEED", "1337"))
    family = os.environ.get("FAMILY", "v2").strip().lower()

    candles = load_tradingview_ohlc_csv(data_csv)
    rng = random.Random(seed)

    if family == "v5":
        best, iters, elapsed = run_family_v5(candles, rng, max_seconds)
    elif family == "v6":
        best, iters, elapsed = run_family_v6(candles, rng, max_seconds)
    elif family == "v7":
        best, iters, elapsed = run_family_v7(candles, rng, max_seconds)
    else:
        best, iters, elapsed = run_family_v2(candles, rng, max_seconds)

    summary = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "family": family,
        "data_csv": data_csv,
        "seed": seed,
        "max_seconds": max_seconds,
        "iters": iters,
        "elapsed_s": elapsed,
        "found": best is not None,
    }

    with open("artifacts/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if best is not None:
        best_out = dict(best)
        best_out["updatedAt"] = summary["ts"]
        best_out["iters"] = iters
        best_out["elapsed_s"] = elapsed
        with open("artifacts/best.json", "w") as f:
            json.dump(best_out, f, indent=2)


if __name__ == "__main__":
    main()
