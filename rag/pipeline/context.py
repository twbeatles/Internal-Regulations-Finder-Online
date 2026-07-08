# -*- coding: utf-8 -*-
"""컨텍스트 조립."""

from __future__ import annotations

from rag.schemas import RetrievedChunk


class ContextBuilder:
    MAX_CHARS = 6000

    def build(self, chunks: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk]]:
        if not chunks:
            return "", []

        used: list[RetrievedChunk] = []
        parts: list[str] = []
        total = 0
        for idx, chunk in enumerate(chunks, start=1):
            header = f"[{idx}] 「{chunk.source}」"
            if chunk.article_no:
                header += f" {chunk.article_no}"
                if chunk.article_title:
                    header += f"({chunk.article_title})"
            body = chunk.content.strip()
            block = f"{header}\n{body}\n"
            if total + len(block) > self.MAX_CHARS and used:
                break
            parts.append(block)
            used.append(chunk)
            total += len(block)
        return "\n".join(parts).strip(), used