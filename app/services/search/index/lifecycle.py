# -*- coding: utf-8 -*-
"""인덱스 생명주기: 초기화·파일 제거."""
from __future__ import annotations

from app.services.search.index.bm25_index import bm25_index_service
from app.services.search.index.meta import normalize_doc_meta
from app.services.search.index.vector_index import vector_index_service


class IndexLifecycle:
    def clear(self, qa, *, preserve_folder: bool = True) -> None:
        qa.file_infos.clear()
        qa.file_details.clear()
        qa.documents = []
        qa.doc_meta = []
        qa.vector_store = None
        qa.bm25 = None
        qa._keyword_cache = []
        qa._search_cache.clear()
        if not preserve_folder:
            qa.current_folder = ""

    def remove_file(self, qa, target_path: str, resolved_name: str, resolved_id: str) -> bool:
        deleted_from_index = False

        if target_path in qa.file_infos:
            del qa.file_infos[target_path]
            deleted_from_index = True

        if target_path in qa.file_details:
            qa.file_details.pop(target_path, None)

        if qa.documents and qa.doc_meta:
            indices_to_remove = [
                i for i, meta in enumerate(qa.doc_meta)
                if meta.get('path') == target_path
                or meta.get('file_id') == resolved_id
                or (
                    not meta.get('path')
                    and not meta.get('file_id')
                    and meta.get('source') == resolved_name
                )
            ]
            for idx in reversed(indices_to_remove):
                if idx < len(qa.documents):
                    del qa.documents[idx]
                if idx < len(qa.doc_meta):
                    del qa.doc_meta[idx]
            if indices_to_remove:
                deleted_from_index = True

        qa._keyword_cache = []
        qa._search_cache.clear()
        normalize_doc_meta(qa)
        bm25_index_service.build(qa)
        vector_index_service.rebuild(qa)
        return deleted_from_index


index_lifecycle = IndexLifecycle()