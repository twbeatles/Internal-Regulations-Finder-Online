#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IRFO 검색 recall 평가 (kcsc-mcp evaluate_search_quality 적응)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="POST /api/search golden recall 평가")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--fixture",
        default=str(Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "search_goldens.json"),
    )
    parser.add_argument("--limit", type=int, default=None, dest="k")
    return parser.parse_args()


def _matches(item: dict[str, Any], expected: dict[str, str]) -> bool:
    if not expected:
        return True
    source = str(item.get("source") or "")
    content = str(item.get("content") or "")
    article = str(item.get("article_no") or "")
    checks = {
        "source": source,
        "source_contains": source,
        "snippet": content,
        "snippet_contains": content,
        "citation_contains": article or content,
    }
    for key, expected_value in expected.items():
        actual = checks.get(key)
        if actual is None:
            continue
        if key.endswith("_contains"):
            if expected_value not in actual:
                return False
        elif actual != expected_value:
            return False
    return True


def _dcg(gains: list[float]) -> float:
    return sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    defaults = fixture.get("defaults") or {}
    queries = fixture.get("queries") or []
    reciprocal_ranks: list[float] = []
    ndcg_values: list[float] = []
    hits = 0
    zero_results = 0
    details: list[dict[str, Any]] = []

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        for entry in queries:
            k = int(args.k or entry.get("k") or defaults.get("k") or 5)
            hybrid = bool(entry.get("hybrid", defaults.get("hybrid", True)))
            response = client.post(
                "/api/search",
                json={
                    "query": entry["query"],
                    "k": k,
                    "hybrid": hybrid,
                    "filter_file_id": entry.get("filter_file_id"),
                    "filter_file": entry.get("filter_file"),
                },
            )
            response.raise_for_status()
            body = response.json()
            results = body.get("data") if body.get("success") else []
            if not isinstance(results, list):
                results = []
            zero_results += int(not results)
            expected = entry.get("expected") if isinstance(entry.get("expected"), dict) else {}
            rank = next(
                (index + 1 for index, item in enumerate(results[:k]) if _matches(item, expected)),
                0,
            )
            if rank:
                hits += 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
            gains = [1.0 if _matches(item, expected) else 0.0 for item in results[:k]]
            ideal = _dcg([1.0] * min(k, 1))
            ndcg_values.append((_dcg(gains) / ideal) if ideal else 0.0)
            details.append(
                {
                    "query": entry["query"],
                    "category": entry.get("category"),
                    "result_count": len(results),
                    "rank": rank,
                    "ndcg_at_k": ndcg_values[-1],
                    "top_sources": [item.get("source") for item in results[: min(k, 3)]],
                }
            )

    total = len(queries) or 1
    return {
        "query_count": len(queries),
        "recall_at_k": hits / total,
        "mrr_at_k": sum(reciprocal_ranks) / total,
        "ndcg_at_k": sum(ndcg_values) / total,
        "zero_result_rate": zero_results / total,
        "details": details,
    }


def main() -> int:
    try:
        report = evaluate(parse_args())
    except httpx.HTTPError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())