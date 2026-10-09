# -*- coding: utf-8 -*-
"""답변 생성."""

from __future__ import annotations

from typing import Iterator

from app.config import AppConfig
from rag.config import RagConfig
from rag.llm.base import LLMProvider
from rag.llm.factory import create_llm_provider


SYSTEM_PROMPT = """당신은 사내 규정 전문 AI 어시스턴트입니다.

규칙:
1. 제공된 컨텍스트에 없는 내용은 추측하지 말고 "해당 규정에서 확인할 수 없습니다"라고 답하세요.
2. 모든 주장 뒤에 반드시 [번호] 형식으로 인용하세요. (예: 연차는 15일입니다[1])
3. 조문을 인용할 때는 「파일명」 제N조(제목) 형식을 사용하세요.
4. 한국어로 간결하고 정확하게 답변하세요.
5. 비교 질문은 표 형식을 사용할 수 있습니다.
"""


class AnswerGenerator:
    def __init__(self, config: RagConfig | None = None, llm: LLMProvider | None = None):
        self.config = config or RagConfig.from_settings()
        self.llm = llm

    def _get_llm(self) -> LLMProvider | None:
        if self.llm:
            return self.llm
        self.llm = create_llm_provider(self.config)
        return self.llm

    def build_messages(self, query: str, context: str, history: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
        max_q = int(getattr(AppConfig, "MAX_RAG_MESSAGE_LENGTH", 4000) or 4000)
        safe_query = str(query or "")[:max_q]
        user_content = (
            f"다음은 사내 규정 발췌 내용입니다.\n\n{context}\n\n"
            f"질문: {safe_query}\n\n"
            "위 컨텍스트만 근거로 답변하고, 각 주장 뒤에 [번호] 인용을 붙이세요."
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            max_items = int(getattr(AppConfig, "MAX_RAG_HISTORY_ITEMS", 6) or 6)
            max_len = int(getattr(AppConfig, "MAX_RAG_HISTORY_ITEM_LENGTH", 2000) or 2000)
            for item in history[-max_items:]:
                role = item.get("role", "user")
                if role in ("user", "assistant"):
                    messages.append({"role": role, "content": str(item.get("content", ""))[:max_len]})
        messages.append({"role": "user", "content": user_content})
        return messages

    def complete(self, query: str, context: str, history: list[dict[str, str]] | None = None) -> str:
        llm = self._get_llm()
        if not llm:
            return self._retrieval_fallback(context)
        messages = self.build_messages(query, context, history)
        return llm.complete(
            messages,
            temperature=self.config.llm.get("temperature", 0.1),
            max_tokens=self.config.llm.get("max_tokens", 2048),
        )

    def stream(self, query: str, context: str, history: list[dict[str, str]] | None = None) -> Iterator[str]:
        llm = self._get_llm()
        if not llm:
            yield self._retrieval_fallback(context)
            return
        messages = self.build_messages(query, context, history)
        yield from llm.stream(
            messages,
            temperature=self.config.llm.get("temperature", 0.1),
            max_tokens=self.config.llm.get("max_tokens", 2048),
        )

    def _retrieval_fallback(self, context: str) -> str:
        if not context.strip():
            return "해당 규정에서 확인할 수 없습니다. (LLM을 사용할 수 없어 검색 결과만 제공합니다.)"
        return (
            "LLM을 사용할 수 없어 검색된 규정 발췌 내용을 안내합니다.\n\n"
            f"{context[:2000]}"
        )