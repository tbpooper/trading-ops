"""TradingView CSV loader utilities.

Supports the 'Download chart data' CSV from TradingView table view.

Expected columns (as seen in MNQ export):
- time (unix seconds)
- open, high, low, close
Optional: volume

We normalize into a list of Candle objects, sorted by timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
import csv


@dataclass(frozen=True)
class Candle:
    ts: int  # unix seconds
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


def load_tradingview_ohlc_csv(path: str | Path) -> List[Candle]:
    p = Path(path)
    with p.open('r', newline='') as f:
        r = csv.DictReader(f)
        candles: List[Candle] = []
        for row in r:
            ts = int(float(row['time']))
            o = float(row['open'])
            h = float(row['high'])
            l = float(row['low'])
            c = float(row['close'])
            v = row.get('volume')
            candles.append(Candle(ts=ts, open=o, high=h, low=l, close=c, volume=float(v) if v not in (None, '', 'null') else None))

    candles.sort(key=lambda x: x.ts)
    return candles


def infer_bar_seconds(candles: List[Candle]) -> int:
    if len(candles) < 2:
        return 0
    # Most common delta
    from collections import Counter
    deltas = [candles[i].ts - candles[i-1].ts for i in range(1, min(len(candles), 2000))]
    d = Counter(deltas).most_common(1)[0][0]
    return int(d)
