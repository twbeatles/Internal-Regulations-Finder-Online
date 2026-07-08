# -*- coding: utf-8 -*-
"""검색 결과 재정렬."""

from __future__ import annotations

import re

from rag.config import RagConfig
from rag.schemas import RetrievedChunk


class ChunkReranker:
    def __init__(self, config: RagConfig | None = None):
        self.config = config or RagConfig.from_settings()

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        top_k = int(self.config.retrieval.get("rerank_top_k", 4))
        tokens = {t.lower() for t in re.split(r"\s+", query) if len(t) >= 2}

        def score_chunk(chunk: RetrievedChunk) -> float:
            text = f"{chunk.article_no} {chunk.article_title} {chunk.content}".lower()
            overlap = sum(1 for t in tokens if t in text)
            article_bonus = 0.2 if chunk.chunk_type == "article" else 0.0
            return chunk.score + overlap * 0.05 + article_bonus

        ranked = sorted(chunks, key=score_chunk, reverse=True)
        return ranked[:top_k]