#!/usr/bin/env python3

from trading.backtest_harness.sweep import sweep_daily_model, format_top


def main():
    rows = sweep_daily_model(n=3000)
    print(format_top(rows, k=15))


if __name__ == '__main__':
    main()
