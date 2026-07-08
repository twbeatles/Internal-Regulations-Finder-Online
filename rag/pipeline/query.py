# -*- coding: utf-8 -*-
"""질의 전처리: 조문 fast-path, multi-query, HyDE."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.config import RagConfig
from rag.pipeline.chunking import extract_article_number


@dataclass
class ProcessedQuery:
    original: str
    queries: list[str] = field(default_factory=list)
    article_no: str | None = None
    intent: str = "factual"

    def primary(self) -> str:
        return self.queries[0] if self.queries else self.original


class QueryProcessor:
    def __init__(self, config: RagConfig | None = None):
        self.config = config or RagConfig.from_settings()

    def process(self, query: str) -> ProcessedQuery:
        query = (query or "").strip()
        result = ProcessedQuery(original=query, queries=[query])
        if not query:
            return result

        if self.config.retrieval.get("article_fast_path", True):
            result.article_no = extract_article_number(query)

        result.intent = self._classify_intent(query)
        result.queries = self._expand_queries(query)
        return result

    def _classify_intent(self, query: str) -> str:
        lowered = query.lower()
        if any(k in lowered for k in ("비교", "차이", "vs", "대비")):
            return "comparison"
        if any(k in lowered for k in ("절차", "방법", "어떻게", "신청")):
            return "procedure"
        return "factual"

    def _expand_queries(self, query: str) -> list[str]:
        queries = [query]
        if not self.config.retrieval.get("hyde", True):
            return queries

        # 규칙 기반 multi-query (LLM 없이 동작)
        tokens = [t for t in re.split(r"\s+", query) if len(t) >= 2]
        if len(tokens) >= 2:
            queries.append(" ".join(tokens[:3]))
            queries.append(f"{tokens[0]} 규정 조항")
        if self.config.retrieval.get("hyde", True):
            queries.append(f"사내 규정에 따르면 {query}에 관한 조항은 다음과 같습니다.")
        # 중복 제거
        seen: set[str] = set()
        unique: list[str] = []
        for q in queries:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(q.strip())
        return unique[:4]