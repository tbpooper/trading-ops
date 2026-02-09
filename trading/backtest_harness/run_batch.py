#!/usr/bin/env python3

from trading.backtest_harness.batch import DailyPnlModel, run_eval_batch


def main():
    # Example toy settings: aim for small steady gains.
    model = DailyPnlModel(p_win=0.60, win=300, loss=-200)
    s = run_eval_batch(n=2000, model=model, seed=42)

    print('Lucid 25k eval (toy daily-PnL model)')
    print(f'n={s.n}')
    print(f'pass_rate={s.pass_rate:.3f}')
    print(f'avg_days={s.avg_days:.2f}')
    print('reasons:')
    for k, v in sorted(s.reasons.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
