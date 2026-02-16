#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Smoke performance benchmark for /api/search.

Usage:
    python scripts/perf_smoke.py
    python scripts/perf_smoke.py --base-url http://127.0.0.1:8080 --query "휴가 규정"
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Sample:
    ok: bool
    latency_ms: float
    bytes_len: int
    status_code: int
    error: str = ""


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(ordered[int(k)])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo))


def one_request(url: str, payload_bytes: bytes, timeout: float) -> Sample:
    req = urllib.request.Request(
        url=url,
        data=payload_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            elapsed = (time.perf_counter() - t0) * 1000.0
            ok = bool(200 <= resp.status < 300)
            return Sample(ok=ok, latency_ms=elapsed, bytes_len=len(body), status_code=resp.status)
    except urllib.error.HTTPError as e:
        elapsed = (time.perf_counter() - t0) * 1000.0
        body = e.read() if e.fp else b""
        return Sample(ok=False, latency_ms=elapsed, bytes_len=len(body), status_code=e.code, error=str(e))
    except Exception as e:  # noqa: BLE001
        elapsed = (time.perf_counter() - t0) * 1000.0
        return Sample(ok=False, latency_ms=elapsed, bytes_len=0, status_code=0, error=str(e))


def run_stage(url: str, payload_bytes: bytes, count: int, concurrency: int, timeout: float) -> List[Sample]:
    samples: List[Sample] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(one_request, url, payload_bytes, timeout) for _ in range(count)]
        for fut in as_completed(futures):
            samples.append(fut.result())
    return samples


def summarize(samples: List[Sample]) -> Tuple[float, float, float, float, float]:
    latencies = [s.latency_ms for s in samples if s.ok]
    avg_size = statistics.mean([s.bytes_len for s in samples]) if samples else 0.0
    err_rate = (len([s for s in samples if not s.ok]) / len(samples) * 100.0) if samples else 0.0
    return (
        percentile(latencies, 50),
        percentile(latencies, 95),
        percentile(latencies, 99),
        float(avg_size),
        float(err_rate),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Search API smoke performance benchmark")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="Server base URL")
    parser.add_argument("--endpoint", default="/api/search", help="Target API endpoint")
    parser.add_argument("--query", default="규정", help="Search query")
    parser.add_argument("--k", type=int, default=5, help="Top-k")
    parser.add_argument("--hybrid", action="store_true", default=True, help="Use hybrid search")
    parser.add_argument("--sort-by", default="relevance", help="Sort field")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup request count")
    parser.add_argument("--measure", type=int, default=200, help="Measured request count")
    parser.add_argument("--concurrency", default="1,5,10", help="Comma-separated concurrency levels")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout (sec)")
    args = parser.parse_args()

    url = args.base_url.rstrip("/") + args.endpoint
    payload = {
        "query": args.query,
        "k": args.k,
        "hybrid": bool(args.hybrid),
        "sort_by": args.sort_by,
    }
    payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    levels = [int(v.strip()) for v in args.concurrency.split(",") if v.strip()]
    if not levels:
        levels = [1, 5, 10]

    print("=== perf_smoke ===")
    print(f"url: {url}")
    print(f"payload: {payload}")
    print(f"warmup: {args.warmup}, measure: {args.measure}, levels: {levels}")
    print("")
    print("conc | p50(ms) | p95(ms) | p99(ms) | avg_bytes | error_rate(%)")
    print("-----|---------|---------|---------|-----------|--------------")

    for c in levels:
        run_stage(url, payload_bytes, args.warmup, c, args.timeout)
        samples = run_stage(url, payload_bytes, args.measure, c, args.timeout)
        p50, p95, p99, avg_size, err_rate = summarize(samples)
        print(f"{c:>4} | {p50:>7.1f} | {p95:>7.1f} | {p99:>7.1f} | {avg_size:>9.1f} | {err_rate:>12.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
