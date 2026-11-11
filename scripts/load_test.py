"""Small load test: N concurrent clients hit POST /predict for a fixed time.

Reports throughput (requests/s) and latency percentiles, writes reports/load_test.json
and exits non-zero if p95 latency is above the acceptance limit (default 300 ms).

    python scripts/load_test.py --seconds 30 --concurrency 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default=os.environ.get("ACEPREDICT_BASE_URL",
                                                         "http://localhost:8000"))
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--p95-limit-ms", type=float, default=300)
    args = ap.parse_args()
    url = args.base_url.rstrip("/")

    players = {t: [p["id"] for p in requests.get(f"{url}/players", params={"tour": t,
                                                                           "limit": 100},
                                                 timeout=10).json()]
               for t in ("atp", "wta")}
    latencies: list[float] = []
    codes: dict[int, int] = {}
    lock = threading.Lock()
    stop_at = time.time() + args.seconds

    def worker(seed: int) -> None:
        rng = random.Random(seed)
        s = requests.Session()
        while time.time() < stop_at:
            tour = rng.choice(["atp", "wta"])
            a, b = rng.sample(players[tour], 2)
            body = {"tour": tour, "player1_id": a, "player2_id": b,
                    "surface": rng.choice(["Hard", "Clay", "Grass"])}
            t0 = time.perf_counter()
            try:
                code = s.post(f"{url}/predict", json=body, timeout=10).status_code
            except requests.RequestException:
                code = 0
            dt = time.perf_counter() - t0
            with lock:
                latencies.append(dt)
                codes[code] = codes.get(code, 0) + 1

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.concurrency)]
    t_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t_start

    lat = sorted(latencies)
    pct = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))] * 1000 if lat else float("nan")  # noqa: E731
    report = {"requests": len(lat), "seconds": round(elapsed, 1),
              "throughput_rps": round(len(lat) / elapsed, 1),
              "p50_ms": round(pct(0.50), 1), "p95_ms": round(pct(0.95), 1),
              "p99_ms": round(pct(0.99), 1),
              "mean_ms": round(statistics.mean(lat) * 1000, 1) if lat else None,
              "status_codes": codes, "concurrency": args.concurrency,
              "p95_limit_ms": args.p95_limit_ms}
    report["passed"] = bool(lat) and report["p95_ms"] <= args.p95_limit_ms \
        and codes.get(200, 0) == len(lat)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "load_test.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
