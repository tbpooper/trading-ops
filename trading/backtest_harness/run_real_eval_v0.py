#!/usr/bin/env python3

"""Run strategy v0 on a TradingView CSV and score a Lucid eval attempt.

This is a first end-to-end run: candles -> trades -> daily pnl -> Lucid eval attempt.

Note: this is NOT a full walk-forward optimization yet.
"""

from trading.backtest_harness.tv_csv import load_tradingview_ohlc_csv
from trading.backtest_harness.strategy_v0 import StrategyParams, generate_trades
from trading.backtest_harness.trades_to_days import to_day_profits_and_closes
from trading.backtest_harness.eval_attempt import simulate_eval_attempt_daily


def main(path: str):
    candles = load_tradingview_ohlc_csv(path)
    params = StrategyParams()
    trades = generate_trades(candles, params, qty_mnq=1)
    profits, closes, keys = to_day_profits_and_closes(trades)

    # Evaluate on first N trading days in the dataset
    r = simulate_eval_attempt_daily(profits, closes)

    print('file:', path)
    print('candles:', len(candles))
    print('trades:', len(trades))
    if keys:
        print('days span:', keys[0], '->', keys[-1], '(', len(keys), 'days )')
    print('eval attempt result:', r)


if __name__ == '__main__':
    main('trading/data/inbound/mnq1_5m_tv_unix.csv')
