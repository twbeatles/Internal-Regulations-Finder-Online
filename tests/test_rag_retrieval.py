# -*- coding: utf-8 -*-
"""RAG 검색 파이프라인 테스트."""

from unittest.mock import MagicMock

from rag.pipeline.query import QueryProcessor
from rag.pipeline.rerank import ChunkReranker
from rag.schemas import RetrievedChunk


class TestRagRetrieval:
    def test_query_processor_expands(self):
        qp = QueryProcessor()
        result = qp.process("연차 휴가 규정")
        assert result.original == "연차 휴가 규정"
        assert len(result.queries) >= 1

    def test_reranker_orders_by_overlap(self):
        reranker = ChunkReranker()
        chunks = [
            RetrievedChunk(0, "기타 내용", "a.txt", "", "f1", 0.2, chunk_type="generic"),
            RetrievedChunk(1, "연차유급휴가 15일", "b.txt", "", "f2", 0.2, article_no="제2조", chunk_type="article"),
        ]
        ranked = reranker.rerank("연차 휴가", chunks)
        assert ranked[0].article_no == "제2조"