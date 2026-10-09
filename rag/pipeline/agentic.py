# -*- coding: utf-8 -*-
"""LangGraph 기반 Agentic RAG (선택)."""

from __future__ import annotations

from typing import Any

from app.utils import logger
from rag.config import RagConfig
from rag.pipeline.orchestrator import RAGPipeline
from rag.schemas import RAGResult


def run_agentic_rag(
    message: str,
    *,
    config: RagConfig | None = None,
    filter_file_id: str | None = None,
    filter_file: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> RAGResult:
    """LangGraph 사용 가능 시 라우팅, 불가 시 선형 파이프라인 폴백."""
    pipeline = RAGPipeline(config=config)
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return pipeline.run(
            message,
            filter_file_id=filter_file_id,
            filter_file=filter_file,
            history=history,
        )

    class _State(dict):
        pass

    def retrieve_node(state: _State) -> _State:
        processed = pipeline.query_processor.process(state["message"])
        chunks = pipeline.retriever.retrieve(
            processed,
            filter_file_id=state.get("filter_file_id"),
            filter_file=state.get("filter_file"),
        )
        chunks = pipeline.reranker.rerank(processed.primary(), chunks)
        context, used = pipeline.context_builder.build(chunks)
        state["context"] = context
        state["chunks"] = used
        return state

    def generate_node(state: _State) -> _State:
        answer = pipeline.generator.complete(
            state["message"],
            state.get("context", ""),
            state.get("history"),
        )
        state["answer"] = answer
        return state

    def guard_node(state: _State) -> _State:
        result = pipeline.guardrails.apply(state.get("answer", ""), state.get("chunks", []))
        state["result"] = result
        return state

    try:
        graph = StateGraph(_State)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("generate", generate_node)
        graph.add_node("guard", guard_node)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "guard")
        graph.add_edge("guard", END)
        app = graph.compile()
        final: dict[str, Any] = app.invoke(
            {
                "message": message,
                "filter_file_id": filter_file_id,
                "filter_file": filter_file,
                "history": history or [],
            }
        )
        result = final.get("result")
        if isinstance(result, RAGResult):
            return result
    except Exception as e:
        logger.warning(f"Agentic RAG 폴백: {e}")

    return pipeline.run(
        message,
        filter_file_id=filter_file_id,
        filter_file=filter_file,
        history=history,
    )