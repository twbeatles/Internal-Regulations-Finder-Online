# -*- coding: utf-8 -*-
"""하이브리드 검색 (Vector + BM25 + RRF)."""

from __future__ import annotations

from typing import Any

from app.config import AppConfig
from app.services.search import RegulationQASystem, qa_system
from rag.config import RagConfig
from rag.pipeline.query import ProcessedQuery
from rag.schemas import RetrievedChunk


class HybridRetriever:
    def __init__(self, qa: RegulationQASystem | None = None, config: RagConfig | None = None):
        self.qa = qa or qa_system
        self.config = config or RagConfig.from_settings()

    def retrieve(
        self,
        processed: ProcessedQuery,
        *,
        filter_file_id: str | None = None,
        filter_file: str | None = None,
    ) -> list[RetrievedChunk]:
        fast = self._article_fast_path(processed)
        if fast:
            return fast

        top_k = int(self.config.retrieval.get("top_k", 8))
        merged: dict[int, RetrievedChunk] = {}
        rrf_k = 60

        for rank, query in enumerate(processed.queries):
            weight = 1.0 if rank == 0 else 0.6
            res = self.qa.search(
                query,
                k=top_k,
                hybrid=bool(self.config.retrieval.get("hybrid", True)),
                sort_by="relevance",
                filter_file=filter_file,
                filter_file_id=filter_file_id,
            )
            if not res.success or not res.data:
                continue
            for item_rank, item in enumerate(res.data):
                doc_id = self._resolve_doc_id(item)
                meta = self._meta_for_doc(doc_id)
                rrf_score = weight * (1.0 / (rrf_k + item_rank + 1))
                if doc_id in merged:
                    merged[doc_id].score += rrf_score
                else:
                    merged[doc_id] = RetrievedChunk(
                        doc_id=doc_id,
                        content=str(item.get("content", "")),
                        source=str(item.get("source", meta.get("source", "?"))),
                        path=str(item.get("path", meta.get("path", ""))),
                        file_id=str(item.get("file_id", meta.get("file_id", ""))),
                        score=rrf_score,
                        article_no=str(item.get("article_no", meta.get("article_no", ""))),
                        article_title=str(item.get("article_title", meta.get("article_title", ""))),
                        chunk_type=str(item.get("chunk_type", meta.get("chunk_type", "generic"))),
                    )

        ranked = sorted(merged.values(), key=lambda c: c.score, reverse=True)
        return ranked[:top_k]

    def _article_fast_path(self, processed: ProcessedQuery) -> list[RetrievedChunk]:
        article_no = processed.article_no
        if not article_no:
            return []
        results: list[RetrievedChunk] = []
        with self.qa._lock:
            for doc_id, meta in enumerate(self.qa.doc_meta):
                meta_article = str(meta.get("article_no", ""))
                if meta_article and meta_article.replace(" ", "") == article_no.replace(" ", ""):
                    if doc_id < len(self.qa.documents):
                        results.append(
                            RetrievedChunk(
                                doc_id=doc_id,
                                content=self.qa.documents[doc_id],
                                source=str(meta.get("source", "?")),
                                path=str(meta.get("path", "")),
                                file_id=str(meta.get("file_id", "")),
                                score=1.0,
                                article_no=meta_article,
                                article_title=str(meta.get("article_title", "")),
                                chunk_type=str(meta.get("chunk_type", "article")),
                            )
                        )
        return results[: int(self.config.retrieval.get("top_k", 8))]

    def _meta_for_doc(self, doc_id: int) -> dict[str, Any]:
        with self.qa._lock:
            if 0 <= doc_id < len(self.qa.doc_meta):
                meta = self.qa.doc_meta[doc_id]
                return meta if isinstance(meta, dict) else {}
        return {}

    def _resolve_doc_id(self, item: dict[str, Any]) -> int:
        content = str(item.get("content", ""))
        path = str(item.get("path", ""))
        with self.qa._lock:
            for doc_id, meta in enumerate(self.qa.doc_meta):
                if path and meta.get("path") == path and self.qa.documents[doc_id] == content:
                    return doc_id
            for doc_id, doc in enumerate(self.qa.documents):
                if doc == content:
                    return doc_id
        return hash(content) % (10**9)