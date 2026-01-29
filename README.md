# Trading Ops (Bats Mark I)

Goal: reduce decision fatigue, enforce risk discipline, and make post-trade review automatic.

This repo will hold:
- A simple trading journal (CSV/SQLite) + import templates
- A strategy "rule card" format (YAML) so you can encode your setups
- A checklist generator for pre-market and opening range
- A lightweight backtest harness (later) once your rules are clearly specified

## Safety / scope
This is decision support and analytics — not auto-trading, not financial advice.

## Next steps
1. Define your setups (names + entry/exit rules) in `strategies/`.
2. Start logging trades with `journal/` templates.
3. Generate daily review prompts.
