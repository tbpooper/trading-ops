from __future__ import annotations

import json
import os
import time


def main():
    os.makedirs("artifacts", exist_ok=True)

    start = time.time()
    result = {
        "status": "NOT_IMPLEMENTED",
        "note": "Workflow scaffolding ready. Next: wire data fetch + invoke search harness.",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": 0,
    }
    result["elapsed_s"] = round(time.time() - start, 2)

    with open("artifacts/status.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
