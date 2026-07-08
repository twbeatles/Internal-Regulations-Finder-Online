# -*- coding: utf-8 -*-
"""MCP 도구 구현 — Flask 앱 컨텍스트 없이 qa_system/RAG 직접 호출."""

from __future__ import annotations

import os
from typing import Any

from app.services.document import ArticleParser
from app.services.search import qa_system
from rag.pipeline.orchestrator import RAGPipeline


def regulations_search(q: str, k: int = 5, hybrid: bool = True) -> dict[str, Any]:
    result = qa_system.search(q.strip(), k=max(1, min(k, 20)), hybrid=hybrid)
    items = []
    if result.success and result.data:
        for item in result.data[:k]:
            items.append(
                {
                    "source": item.get("source"),
                    "file_id": item.get("file_id"),
                    "score": item.get("score"),
                    "content": (item.get("content") or "")[:800],
                    "path": item.get("path"),
                }
            )
    return {"success": result.success, "message": result.message, "results": items}


def regulations_rag_chat(message: str, filter_file_id: str | None = None) -> dict[str, Any]:
    pipeline = RAGPipeline()
    if not pipeline.is_ready():
        return {"success": False, "message": "인덱스 미준비", "answer": ""}
    result = pipeline.run(message.strip(), filter_file_id=filter_file_id)
    return {
        "success": not result.refused,
        "answer": result.answer,
        "confidence": result.confidence,
        "verification_score": result.verification_score,
        "citations": [c.to_dict() for c in result.citations],
        "refused": result.refused,
    }


def regulations_list_files() -> dict[str, Any]:
    files = []
    for meta in getattr(qa_system, "doc_meta", []) or []:
        files.append(
            {
                "source": meta.get("source"),
                "file_id": meta.get("file_id"),
                "path": meta.get("path"),
            }
        )
    seen: set[str] = set()
    unique = []
    for f in files:
        fid = str(f.get("file_id") or f.get("source") or "")
        if fid in seen:
            continue
        seen.add(fid)
        unique.append(f)
    return {"count": len(unique), "files": unique[:200]}


def regulations_status() -> dict[str, Any]:
    pipeline = RAGPipeline()
    return {
        "ready": pipeline.is_ready(),
        "documents": len(qa_system.documents),
        "has_vector": qa_system.vector_store is not None,
        "has_bm25": qa_system.bm25 is not None,
    }


def regulations_get_article(file_id: str, article_query: str) -> dict[str, Any]:
    target_path = ""
    target_source = ""
    for meta in getattr(qa_system, "doc_meta", []) or []:
        if str(meta.get("file_id")) == file_id:
            target_path = str(meta.get("path") or "")
            target_source = str(meta.get("source") or "")
            break
    if not target_path or not os.path.isfile(target_path):
        return {"success": False, "message": "파일을 찾을 수 없습니다"}

    from app.services.document import DocumentExtractor

    extracted = DocumentExtractor().extract_with_details(target_path)
    parser = ArticleParser()
    articles = parser.parse_articles(extracted.text)
    matches = parser.search_article(articles, article_query)
    return {
        "success": True,
        "source": target_source,
        "file_id": file_id,
        "matches": matches[:5],
    }


def regulations_reindex(admin_token: str = "") -> dict[str, Any]:
    from app.services.settings_store import get_settings_store

    settings = get_settings_store().load()
    expected = str(settings.get("mcp", {}).get("admin_token") or "").strip()
    if expected and admin_token != expected:
        return {"success": False, "message": "admin_token 불일치"}
    folder = str(settings.get("folder") or "").strip()
    if not folder or not os.path.isdir(folder):
        return {"success": False, "message": "동기화 폴더가 설정되지 않았습니다"}
    result = qa_system.initialize(folder)
    return {"success": bool(result.success), "message": result.message}