# Lucid Backtest Harness (WIP)

Goal: simulate strategy results under LucidBlack rules:
- Profit targets
- EOD drawdown / Max Loss Limit (MLL)
- Consistency (largest day / total profit)
- Size caps + funded scaling tiers

This harness is intentionally minimal at first:
- It validates rule math with synthetic sequences
- Then we can wire in real market data (MNQ) and strategy signals

## Run tests

```bash
python3 -m pytest -q
```

(If pytest isn't installed, we can add it or run the modules directly.)
