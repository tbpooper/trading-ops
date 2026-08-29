# Trading Ops (Bats Mark I)

Goal: reduce decision fatigue, enforce risk discipline, and make post-trade review automatic.

This repo will hold:
- A simple trading journal (CSV/SQLite) + import templates
- A strategy "rule card" format (YAML) so you can encode your setups
- A checklist generator for pre-market and opening range
- A lightweight backtest harness (later) once your rules are clearly specified

## Safety / scope
- Decision support + analytics only.
- No auto-order execution.
- Not financial advice.

## Quick start
1) Pick 1 setup and write it as a rule card in `strategies/`.
2) Log every trade in `journal/trades.template.csv`.
3) Use `checklists/opening-checklist.md` before 9:30.

## Roadmap
See `ROADMAP.md`.
