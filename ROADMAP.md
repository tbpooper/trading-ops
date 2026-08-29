# Trading Ops Roadmap

## Phase 0 — Foundation (now)
- [x] Repo scaffold (journal / strategies / checklists)
- [ ] PR template + review process

## Phase 1 — Trading journal automation
- [ ] CSV → SQLite importer
- [ ] Daily summary (PnL, R, winrate, biggest mistake tag)
- [ ] Screenshot folder conventions

## Phase 2 — Strategy rule cards
- [ ] One YAML per setup
- [ ] Validator (ensures stop/invalidations/risk fields exist)
- [ ] “Is this an A+ setup?” pre-trade prompt generator

## Phase 3 — Backtesting (only after rules are precise)
- [ ] Backtest harness + metrics
- [ ] Paper-trade simulation mode

## Safety
- No auto-order execution. Alerts/analytics only until explicitly approved.
