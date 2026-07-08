# Ported from kcsc-mcp (2026-07)
from __future__ import annotations

import re
import sys
from pathlib import Path

PRINTABLE_RE = re.compile(r"[ -~가-힣]{3,}")


def _utf16_candidates(raw: bytes) -> list[str]:
    try:
        decoded = raw.decode("utf-16-le", errors="ignore")
    except Exception:
        return []
    return [match.group(0).strip() for match in PRINTABLE_RE.finditer(decoded) if match.group(0).strip()]


def _utf8_candidates(raw: bytes) -> list[str]:
    try:
        decoded = raw.decode("utf-8", errors="ignore")
    except Exception:
        return []
    return [match.group(0).strip() for match in PRINTABLE_RE.finditer(decoded) if match.group(0).strip()]


def _dedupe(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for line in lines:
        normalized = " ".join(line.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def extract_hwp_text(path: str) -> str:
    raw = Path(path).read_bytes()
    lines = _dedupe([*_utf16_candidates(raw), *_utf8_candidates(raw)])
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: hwp_dump.py <path>", file=sys.stderr)
        return 2
    lines = extract_hwp_text(sys.argv[1])
    if not lines:
        print("Bundled HWP converter could not extract readable text", file=sys.stderr)
        return 1
    sys.stdout.write(lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())