# -*- coding: utf-8 -*-
"""FAISS 벡터 인덱스 재구축."""
from __future__ import annotations

from typing import Any

import app.services.search as _search_exports

from app.services.search.index.meta import normalize_doc_meta
from app.utils import logger


class VectorIndexService:
    def rebuild(self, qa) -> None:
        if not qa.documents:
            qa.vector_store = None
            return

        if not (qa.embedding_model and _search_exports.FAISS and _search_exports.Document):
            qa.vector_store = None
            return

        try:
            normalize_doc_meta(qa)
            docs = [
                _search_exports.Document(page_content=qa.documents[idx], metadata=dict(qa.doc_meta[idx]))
                for idx in range(len(qa.documents))
            ]
            qa.vector_store = _search_exports.FAISS.from_documents(docs, qa.embedding_model) if docs else None
        except Exception as e:
            logger.warning(f"벡터 인덱스 재구축 실패: {e}")
            qa.vector_store = None


vector_index_service = VectorIndexService()