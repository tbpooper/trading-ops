#!/usr/bin/env python3

"""Sweep Strategy v1 (sizing + gating) over a small grid and report top configs.

This uses rolling 5-day windows on MNQ 5m CSV.
"""

from itertools import product

from trading.backtest_harness.tv_csv import load_tradingview_ohlc_csv
from trading.backtest_harness.strategy_v1 import Params, Filters, generate_trades
from trading.backtest_harness.rolling_eval import rolling_eval_5day


def main():
    candles = load_tradingview_ohlc_csv('trading/data/inbound/mnq1_5m_tv_unix.csv')

    ema_pairs = [(10, 30), (20, 30), (20, 50)]
    atr_mults = [1.5, 2.0]
    rrs = [0.75, 1.0, 1.5]
    rsi_longs = [45, 50]
    rsi_shorts = [55, 50]
    risks = [150, 200]

    results = []
    for (ef, es), am, rr, rmin, rmax, risk in product(ema_pairs, atr_mults, rrs, rsi_longs, rsi_shorts, risks):
        if ef >= es:
            continue
        p = Params(
            ema_fast=ef,
            ema_slow=es,
            rsi_min_long=rmin,
            rsi_max_short=rmax,
            atr_mult=am,
            rr=rr,
            risk_per_trade_dollars=risk,
            max_micros=20,
            filters=Filters(),
        )
        s = rolling_eval_5day(
            candles,
            lambda sub: generate_trades(sub, p),
            daily_profit_cap=750.0,
            daily_loss_cap=300.0,
            use_trade_stream=True,
        )
        mll = s.reasons.get('mll_breach', 0)
        tl = s.reasons.get('time_limit', 0)
        results.append((s.pass_rate, -mll, -tl, p, s))

    results.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))

    print('Top 12 configs (rolling 5-day windows):')
    for i, (rate, negmll, negtl, p, s) in enumerate(results[:12], 1):
        print(
            i,
            f'pass_rate={rate:.3f} ({s.passed}/{s.windows})',
            f'mll={s.reasons.get("mll_breach",0)}',
            f'time_limit={s.reasons.get("time_limit",0)}',
            f'ema=({p.ema_fast},{p.ema_slow})',
            f'rsi=({p.rsi_min_long},{p.rsi_max_short})',
            f'atr_mult={p.atr_mult}',
            f'rr={p.rr}',
            f'risk=${p.risk_per_trade_dollars:.0f}',
        )


if __name__ == '__main__':
    main()
