# -*- coding: utf-8 -*-
"""RAG API 스키마."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Citation:
    index: int
    file_id: str
    source: str
    article_no: str = ""
    article_title: str = ""
    excerpt: str = ""
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedChunk:
    doc_id: int
    content: str
    source: str
    path: str
    file_id: str
    score: float
    article_no: str = ""
    article_title: str = ""
    chunk_type: str = "generic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RAGResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    verification_score: float = 0.0
    refused: bool = False
    retrieval_only: bool = False
    conversation_id: str = ""
    message_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "confidence": self.confidence,
            "verification_score": self.verification_score,
            "refused": self.refused,
            "retrieval_only": self.retrieval_only,
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
        }