# -*- coding: utf-8 -*-
"""환각 방지 및 인용 검증."""

from __future__ import annotations

import re

from rag.config import RagConfig
from rag.schemas import Citation, RAGResult, RetrievedChunk


class Guardrails:
    _CITATION_RE = re.compile(r"\[(\d+)\]")

    def __init__(self, config: RagConfig | None = None):
        self.config = config or RagConfig.from_settings()

    def apply(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
        *,
        retrieval_only: bool = False,
    ) -> RAGResult:
        disclaimer = str(self.config.guardrails.get("disclaimer", "")).strip()
        citations = self._build_citations(chunks)
        refused = False
        confidence = self._estimate_confidence(answer, chunks)

        min_conf = float(self.config.guardrails.get("min_confidence", 0.35))
        if not chunks:
            answer = "해당 규정에서 확인할 수 없습니다."
            refused = True
            confidence = 0.0
        elif confidence < min_conf and self.config.guardrails.get("require_citation", True):
            if "확인할 수 없습니다" not in answer:
                answer = "해당 규정에서 확인할 수 없습니다."
            refused = True

        if disclaimer and disclaimer not in answer:
            answer = f"{answer.rstrip()}\n\n— {disclaimer}"

        return RAGResult(
            answer=answer.strip(),
            citations=citations,
            confidence=confidence,
            refused=refused,
            retrieval_only=retrieval_only,
        )

    def _build_citations(self, chunks: list[RetrievedChunk]) -> list[Citation]:
        citations: list[Citation] = []
        for idx, chunk in enumerate(chunks, start=1):
            excerpt = chunk.content[:300] + ("..." if len(chunk.content) > 300 else "")
            citations.append(
                Citation(
                    index=idx,
                    file_id=chunk.file_id,
                    source=chunk.source,
                    article_no=chunk.article_no,
                    article_title=chunk.article_title,
                    excerpt=excerpt,
                    path=chunk.path,
                )
            )
        return citations

    def _estimate_confidence(self, answer: str, chunks: list[RetrievedChunk]) -> float:
        if not chunks:
            return 0.0
        if "확인할 수 없습니다" in answer:
            return 0.1
        cited = {int(m) for m in self._CITATION_RE.findall(answer)}
        if self.config.guardrails.get("require_citation", True) and not cited:
            return 0.2
        base = min(0.95, 0.4 + len(chunks) * 0.1)
        if cited:
            valid = sum(1 for c in cited if 1 <= c <= len(chunks))
            base += valid * 0.05
        return min(base, 1.0)