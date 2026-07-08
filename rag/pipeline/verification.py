# -*- coding: utf-8 -*-
"""RAG 답변 grounding 휴리스틱 (kcsc-mcp verification_parts 이식, DB 없음)."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Ported from kcsc-mcp (2026-07)

KOREAN_CITATION_LABEL_PATTERN = re.compile(
    r"(제\s*\d+\s*조(?:의\s*\d+)?(?:\s*제?\s*\d+\s*항)?(?:\s*제?\s*\d+\s*호)?)"
)
SOURCE_PREFIX_PATTERN = re.compile(r"^(?:출처|source)\s*:\s*", re.IGNORECASE)
QUOTED_FRAGMENT_PATTERNS = (
    re.compile(r'"([^"\n]+)"'),
    re.compile(r"'([^'\n]+)'"),
)


def _normalize_text_compact(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", "", value).lower()


def _normalized_tokens(value: str | None) -> list[str]:
    if not value:
        return []
    return [token for token in (_normalize_text_compact(part) for part in re.split(r"\s+", value)) if token]


@dataclass
class VerificationReport:
    score: float
    supported_claims: int
    total_claims: int
    unsupported_claims: list[str]

    def to_dict(self) -> dict:
        return {
            "verification_score": self.score,
            "supported_claims": self.supported_claims,
            "total_claims": self.total_claims,
            "unsupported_claims": self.unsupported_claims,
        }


def extract_claims(answer_text: str) -> list[str]:
    claims: list[str] = []
    for raw_line in (answer_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- "):
            claims.append(line[2:].strip())
        elif line and not line.startswith(("|", "#")):
            claims.extend(
                segment.strip()
                for segment in re.split(r"(?<=[.!?。])\s+|[.。]\s*", line)
                if len(segment.strip()) > 12
            )
    if not claims and answer_text and len(answer_text.strip()) > 12:
        claims.append(answer_text.strip())
    return claims


def _claim_support_status(claim: str, citation_tokens: set[str]) -> tuple[str, float]:
    normalized_claim = _normalize_text_compact(claim)
    if not normalized_claim:
        return "not_checkable", 1.0
    tokens = [token for token in _normalized_tokens(claim) if len(token) > 1]
    if not tokens:
        return "not_checkable", 1.0
    overlap = sum(1 for token in tokens if token in citation_tokens)
    ratio = overlap / max(len(tokens), 1)
    if overlap >= max(1, min(3, len(tokens) // 2)):
        return "supported", min(1.0, max(0.35, ratio))
    if overlap > 0:
        return "partial", max(0.1, ratio)
    return "unsupported", 0.0


def build_citation_token_set(chunk_texts: list[str]) -> set[str]:
    tokens: set[str] = set()
    for text in chunk_texts:
        tokens.update(_normalized_tokens(text))
        compact = _normalize_text_compact(text)
        if compact:
            tokens.add(compact)
    return tokens


def verify_answer_against_chunks(answer: str, chunk_texts: list[str]) -> VerificationReport:
    """검색 청크 대비 답변 grounding 점수."""
    if not chunk_texts:
        return VerificationReport(score=0.0, supported_claims=0, total_claims=0, unsupported_claims=[])

    citation_tokens = build_citation_token_set(chunk_texts)
    claims = extract_claims(answer)
    if not claims:
        return VerificationReport(score=0.3, supported_claims=0, total_claims=0, unsupported_claims=[])

    supported = 0
    unsupported: list[str] = []
    scores: list[float] = []
    for claim in claims:
        status, claim_score = _claim_support_status(claim, citation_tokens)
        scores.append(claim_score)
        if status in ("supported", "partial", "not_checkable"):
            supported += 1
        else:
            unsupported.append(claim[:120])

    total = len(claims)
    avg = sum(scores) / len(scores) if scores else 0.0
    ratio = supported / total if total else 0.0
    final_score = round(min(1.0, avg * 0.6 + ratio * 0.4), 3)
    return VerificationReport(
        score=final_score,
        supported_claims=supported,
        total_claims=total,
        unsupported_claims=unsupported[:5],
    )