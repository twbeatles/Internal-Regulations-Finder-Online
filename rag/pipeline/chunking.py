# -*- coding: utf-8 -*-
"""조문 인식 청킹."""

from __future__ import annotations

import re
from typing import Any

from app.services.document import ArticleParser, DocumentSplitter

_ARTICLE_NO_RE = re.compile(r"제\s*(\d+)\s*조(?:의\s*(\d+))?", re.IGNORECASE)
_ANNEX_HEADING_RE = re.compile(r"^(부칙|별표\s*\d+|별첨\s*\d+|첨부\s*\d+)", re.MULTILINE)
_TABLE_BLOCK_RE = re.compile(r"(?:\|[^\n]+\|\n)+")


def extract_article_number(query: str) -> str | None:
    match = _ARTICLE_NO_RE.search(query or "")
    if not match:
        return None
    main = match.group(1)
    sub = match.group(2)
    return f"제{main}조" + (f"의{sub}" if sub else "")


def _extract_table_chunks(text: str, chunk_size: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for match in _TABLE_BLOCK_RE.finditer(text):
        table_text = match.group(0).strip()
        if len(table_text) < 20:
            continue
        summary = table_text if len(table_text) <= chunk_size else table_text[:chunk_size] + "..."
        chunks.append(
            {
                "text": summary,
                "article_no": "",
                "article_title": "표",
                "chunk_type": "table",
            }
        )
    return chunks


def _extract_annex_chunks(text: str, chunk_size: int, doc_splitter: DocumentSplitter) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    parts = _ANNEX_HEADING_RE.split(text)
    headings = _ANNEX_HEADING_RE.findall(text)
    if not headings:
        return chunks
    for heading, body in zip(headings, parts[1:], strict=False):
        section = f"{heading.strip()}\n{(body or '').strip()}".strip()
        if not section:
            continue
        chunk_type = "supplementary" if heading.strip().startswith("부칙") else "annex"
        if len(section) <= chunk_size:
            chunks.append(
                {
                    "text": section,
                    "article_no": "",
                    "article_title": heading.strip(),
                    "chunk_type": chunk_type,
                }
            )
        else:
            for sub_id, sub_text in enumerate(doc_splitter.split(section)):
                chunks.append(
                    {
                        "text": sub_text,
                        "article_no": "",
                        "article_title": heading.strip(),
                        "chunk_type": chunk_type,
                        "sub_chunk_id": sub_id,
                    }
                )
    return chunks


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

    chunks: list[dict[str, Any]] = []
    chunks.extend(_extract_table_chunks(text, chunk_size))
    chunks.extend(_extract_annex_chunks(text, chunk_size, doc_splitter))

    articles = article_parser.parse_articles(text)
    if articles and len(articles) >= 1:
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