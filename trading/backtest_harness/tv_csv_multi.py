"""Load TradingView OHLC CSVs with various timeframes.

We already have tv_csv loader that expects a specific unix format file.
These TradingView exports (e.g., "CME_MINI_MNQ1!, 1.csv") typically include
columns like: time, open, high, low, close, volume.

This module provides a tolerant loader that:
- parses time column (epoch ms or ISO) into unix seconds
- returns candles sorted by timestamp

Used for 1m/5m/15m multi-timeframe strategy testing.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List


@dataclass(frozen=True)
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def _parse_time(s: str) -> int:
    s = s.strip().strip('"')
    if s.isdigit():
        # could be seconds or ms
        v = int(s)
        if v > 10_000_000_000:
            return v // 1000
        return v
    # try ISO formats
    # TradingView often uses: 2026-02-05T14:30:00Z
    try:
        dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
        return int(dt.timestamp())
    except Exception:
        pass
    # try: YYYY-MM-DD HH:MM:SS
    for fmt in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except Exception:
            continue
    raise ValueError(f"Unrecognized time value: {s}")


def load_tv_csv(path: str) -> List[Candle]:
    out: List[Candle] = []
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        cols = {c.lower(): c for c in (r.fieldnames or [])}
        # flexible headers
        tcol = cols.get('time') or cols.get('timestamp') or cols.get('date')
        if not tcol:
            raise ValueError(f"No time column found in {path}: {r.fieldnames}")
        for row in r:
            ts = _parse_time(row[tcol])
            out.append(
                Candle(
                    ts=ts,
                    open=float(row[cols.get('open','open')]),
                    high=float(row[cols.get('high','high')]),
                    low=float(row[cols.get('low','low')]),
                    close=float(row[cols.get('close','close')]),
                    volume=float(row.get(cols.get('volume','volume'), 0) or 0),
                )
            )
    out.sort(key=lambda c: c.ts)
    return out
