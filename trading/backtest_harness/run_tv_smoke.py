#!/usr/bin/env python3

from trading.backtest_harness.tv_csv import load_tradingview_ohlc_csv, infer_bar_seconds


def main():
    path = 'trading/data/inbound/mnq1_5m_tv_unix.csv'
    candles = load_tradingview_ohlc_csv(path)
    print('loaded', len(candles), 'candles')
    print('first', candles[0])
    print('last', candles[-1])
    print('bar_seconds', infer_bar_seconds(candles))


if __name__ == '__main__':
    main()
