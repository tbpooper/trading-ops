"""Helpers to compute speed metrics (pass-by-day buckets) and fastpass score.

We optimize for passing within 5 days, but it helps to track how far out passes occur.
"""

from __future__ import annotations

from typing import Dict, Tuple


def pass_buckets(pass_days_hist: Dict[int, int] | Dict[str, int], windows: int) -> Dict[str, Tuple[int, float]]:
    hist = {int(k): int(v) for k, v in pass_days_hist.items()}

    def cnt(lo: int, hi: int) -> int:
        return sum(v for d, v in hist.items() if lo <= d <= hi)

    buckets = {
        "1-5": cnt(1, 5),
        "6-10": cnt(6, 10),
        "11-15": cnt(11, 15),
        "16-20": cnt(16, 20),
        "21-25": cnt(21, 25),
        "26-30": cnt(26, 30),
    }
    return {k: (v, (v / windows if windows else 0.0)) for k, v in buckets.items()}


def fastpass_score(pass_days_hist: Dict[int, int] | Dict[str, int], windows: int) -> float:
    """Heuristic score rewarding earlier passes more."""
    hist = {int(k): int(v) for k, v in pass_days_hist.items()}
    score = 0.0
    for day, n in hist.items():
        if day <= 5:
            w = 1.0
        elif day <= 10:
            w = 0.5
        elif day <= 15:
            w = 0.25
        elif day <= 20:
            w = 0.1
        else:
            w = 0.0
        score += w * n
    return score / windows if windows else 0.0
