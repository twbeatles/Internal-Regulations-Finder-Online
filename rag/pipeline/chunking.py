# -*- coding: utf-8 -*-
"""조문 인식 청킹."""

from __future__ import annotations

import re
from typing import Any

from app.services.document import ArticleParser, DocumentSplitter

_ARTICLE_NO_RE = re.compile(r"제\s*(\d+)\s*조(?:의\s*(\d+))?", re.IGNORECASE)


def extract_article_number(query: str) -> str | None:
    match = _ARTICLE_NO_RE.search(query or "")
    if not match:
        return None
    main = match.group(1)
    sub = match.group(2)
    return f"제{main}조" + (f"의{sub}" if sub else "")


def build_chunks_from_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    article_parser: ArticleParser | None = None,
    doc_splitter: DocumentSplitter | None = None,
) -> list[dict[str, Any]]:
    """규정 텍스트를 RAG용 청크+메타 목록으로 변환."""
    article_parser = article_parser or ArticleParser()
    doc_splitter = doc_splitter or DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    articles = article_parser.parse_articles(text)
    if articles and len(articles) >= 1:
        chunks: list[dict[str, Any]] = []
        for article in articles:
            number = str(article.get("number", "")).strip()
            title = str(article.get("title", "")).strip()
            body = str(article.get("content", "")).strip()
            article_text = f"{number} {title}\n{body}".strip()
            if not article_text:
                continue
            if len(article_text) <= chunk_size:
                chunks.append(
                    {
                        "text": article_text,
                        "article_no": number,
                        "article_title": title,
                        "chunk_type": "article",
                    }
                )
            else:
                sub_chunks = doc_splitter.split(article_text)
                for sub_id, sub_text in enumerate(sub_chunks):
                    chunks.append(
                        {
                            "text": sub_text,
                            "article_no": number,
                            "article_title": title,
                            "chunk_type": "paragraph",
                            "sub_chunk_id": sub_id,
                        }
                    )
        if chunks:
            return chunks

    plain_chunks = doc_splitter.split(text)
    return [
        {
            "text": chunk,
            "article_no": "",
            "article_title": "",
            "chunk_type": "generic",
        }
        for chunk in plain_chunks
        if chunk and chunk.strip()
    ]