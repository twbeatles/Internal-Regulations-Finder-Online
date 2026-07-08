# -*- coding: utf-8 -*-
"""문서·청크 메타데이터 정규화 및 구성."""
from __future__ import annotations

import os
from typing import Any, Dict, List

from app.utils import FileUtils


def normalize_doc_meta(qa) -> None:
    """doc_meta를 정규화합니다. 호출자가 qa._lock을 보유해야 합니다."""
    normalized_meta: List[Dict[str, Any]] = []
    for doc_id, content in enumerate(qa.documents):
        meta = {}
        if doc_id < len(qa.doc_meta) and isinstance(qa.doc_meta[doc_id], dict):
            meta = dict(qa.doc_meta[doc_id])

        path = str(meta.get('path', '') or '')
        source = str(meta.get('source', '') or (os.path.basename(path) if path else '?'))
        normalized_meta.append({
            **meta,
            'doc_id': doc_id,
            'source': source,
            'path': path,
            'file_id': meta.get('file_id') or (FileUtils.make_file_id(path) if path else ''),
        })
    qa.doc_meta = normalized_meta


def build_chunk_meta(
    *,
    doc_id: int,
    filename: str,
    file_path: str,
    file_id: str,
    chunk_id: int,
    total_chunks: int,
    details: Dict[str, Any] | None = None,
    article_no: str = "",
    article_title: str = "",
    chunk_type: str = "generic",
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        'doc_id': doc_id,
        'source': filename,
        'path': file_path,
        'file_id': file_id,
        'chunk_id': chunk_id,
        'total_chunks': total_chunks,
        'article_no': article_no,
        'article_title': article_title,
        'chunk_type': chunk_type,
    }
    if details:
        metadata = details.get('metadata')
        diagnostics = details.get('diagnostics')
        tables = details.get('tables')
        if metadata:
            meta['document_metadata'] = metadata
        if diagnostics:
            meta['diagnostics'] = diagnostics
        if tables:
            meta['table_count'] = len(tables)
    return meta


def remember_file_details(qa, file_path: str, extracted) -> Dict[str, Any]:
    details = {
        'metadata': dict(getattr(extracted, 'metadata', {}) or {}),
        'tables': list(getattr(extracted, 'table_dicts', lambda: [])() or []),
        'diagnostics': dict(getattr(extracted, 'diagnostics', {}) or {}),
    }
    qa.file_details[file_path] = details
    return details