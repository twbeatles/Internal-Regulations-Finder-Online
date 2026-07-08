# -*- coding: utf-8 -*-
"""RAG 파이프라인 오케스트레이터."""

from __future__ import annotations

from typing import Iterator

from app.services.search import RegulationQASystem, qa_system
from rag.config import RagConfig
from rag.llm.factory import create_llm_provider
from rag.pipeline.context import ContextBuilder
from rag.pipeline.generation import AnswerGenerator
from rag.pipeline.guardrails import Guardrails
from rag.pipeline.query import QueryProcessor
from rag.pipeline.rerank import ChunkReranker
from rag.pipeline.retrieval import HybridRetriever
from rag.schemas import RAGResult


class RAGPipeline:
    def __init__(self, qa: RegulationQASystem | None = None, config: RagConfig | None = None):
        self.config = config or RagConfig.from_settings()
        self.qa = qa or qa_system
        self.query_processor = QueryProcessor(self.config)
        self.retriever = HybridRetriever(self.qa, self.config)
        self.reranker = ChunkReranker(self.config)
        self.context_builder = ContextBuilder()
        self.generator = AnswerGenerator(self.config)
        self.guardrails = Guardrails(self.config)

    def is_ready(self) -> bool:
        return bool(self.qa.documents) or bool(self.qa.bm25)

    def run(
        self,
        message: str,
        *,
        filter_file_id: str | None = None,
        filter_file: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> RAGResult:
        processed = self.query_processor.process(message)
        chunks = self.retriever.retrieve(processed, filter_file_id=filter_file_id, filter_file=filter_file)
        chunks = self.reranker.rerank(processed.primary(), chunks)
        context, used_chunks = self.context_builder.build(chunks)
        llm = create_llm_provider(self.config)
        retrieval_only = llm is None or not llm.health()
        answer = self.generator.complete(message, context, history)
        return self.guardrails.apply(answer, used_chunks, retrieval_only=retrieval_only)

    def stream(
        self,
        message: str,
        *,
        filter_file_id: str | None = None,
        filter_file: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[dict]:
        processed = self.query_processor.process(message)
        chunks = self.retriever.retrieve(processed, filter_file_id=filter_file_id, filter_file=filter_file)
        chunks = self.reranker.rerank(processed.primary(), chunks)
        context, used_chunks = self.context_builder.build(chunks)

        llm = create_llm_provider(self.config)
        retrieval_only = llm is None or not llm.health()

        for citation in self.guardrails._build_citations(used_chunks):
            yield {"event": "citation", "data": citation.to_dict()}

        if not context.strip():
            yield {"event": "token", "data": "해당 규정에서 확인할 수 없습니다."}
            result = self.guardrails.apply("해당 규정에서 확인할 수 없습니다.", used_chunks, retrieval_only=retrieval_only)
            yield {"event": "done", "data": result.to_dict()}
            return

        answer_parts: list[str] = []
        for token in self.generator.stream(message, context, history):
            answer_parts.append(token)
            yield {"event": "token", "data": token}

        answer = "".join(answer_parts)
        result = self.guardrails.apply(answer, used_chunks, retrieval_only=retrieval_only)
        yield {"event": "done", "data": result.to_dict()}