"""Aggregate matrix artifacts into a single best result.

Expected layout:
- artifacts/<artifact-name>/best.json
- artifacts/<artifact-name>/summary.json

Writes:
- results/latest_best.json
- results/latest_table.json

No external deps.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def load_json(p: Path):
    return json.loads(p.read_text())


def main():
    root = Path("artifacts")
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for best_path in root.glob("**/best.json"):
        try:
            best = load_json(best_path)
        except Exception:
            continue
        summary_path = best_path.with_name("summary.json")
        summary = load_json(summary_path) if summary_path.exists() else {}

        rows.append(
            {
                "artifact": str(best_path.parent),
                "family": best.get("family") or best.get("strategy") or summary.get("family"),
                "seed": summary.get("seed"),
                "pass5_rate": best.get("pass5_rate"),
                "mll_rate": best.get("mll_rate"),
                "timeout_rate": best.get("timeout_rate"),
                "score": best.get("score"),
                "cfg": best.get("cfg"),
                "outcomes": best.get("outcomes"),
                "updatedAt": best.get("updatedAt") or summary.get("ts"),
            }
        )

    # filter valid
    rows = [r for r in rows if isinstance(r.get("pass5_rate"), (int, float))]

    # best: max pass5, then min timeout
    def key(r):
        return (float(r.get("pass5_rate", 0.0)), -float(r.get("timeout_rate", 1e9)))

    best = max(rows, key=key) if rows else None

    (out_dir / "latest_table.json").write_text(json.dumps({"rows": rows}, indent=2))
    (out_dir / "latest_best.json").write_text(json.dumps({"best": best}, indent=2))

    # Also print a single-line summary for logs
    if best:
        print(
            f"BEST family={best.get('family')} seed={best.get('seed')} pass5={best.get('pass5_rate'):.4f} "
            f"mll={best.get('mll_rate'):.4f} timeout={best.get('timeout_rate'):.4f}"
        )
    else:
        print("BEST none")


if __name__ == "__main__":
    main()
