# -*- coding: utf-8 -*-
"""RegulationQASystem 공개 API 파사드."""
from __future__ import annotations

from typing import Any, Dict, List

from app.utils import TaskResult

from app.services.search.qa_system import RegulationQASystem


class RegulationQAFacade:
    """RegulationQASystem 래퍼 — 명시적 공개 메서드만 노출."""

    def __init__(self, qa: RegulationQASystem | None = None):
        self._qa = qa or RegulationQASystem()

    @property
    def is_ready(self) -> bool:
        return self._qa.is_ready

    @property
    def is_loading(self) -> bool:
        return self._qa.is_loading

    def search(
        self,
        query: str,
        k: int = 5,
        hybrid: bool = True,
        sort_by: str = 'relevance',
        filter_file: str | None = None,
        filter_file_id: str | None = None,
    ) -> TaskResult:
        return self._qa.search(query, k, hybrid, sort_by, filter_file, filter_file_id)

    def initialize(self, folder_path: str, force_reindex: bool = False) -> TaskResult:
        return self._qa.initialize(folder_path, force_reindex=force_reindex)

    def process_documents(
        self,
        folder: str,
        files: List[str],
        progress_cb=None,
        force_reindex: bool = False,
    ) -> TaskResult:
        return self._qa.process_documents(folder, files, progress_cb, force_reindex=force_reindex)

    def process_single_file(self, file_path: str) -> TaskResult:
        return self._qa.process_single_file(file_path)

    def clear_index(self, *, preserve_folder: bool = True) -> None:
        self._qa.clear_index(preserve_folder=preserve_folder)

    def remove_file_from_index(self, target_path: str, resolved_name: str, resolved_id: str) -> bool:
        return self._qa.remove_file_from_index(target_path, resolved_name, resolved_id)

    def load_model(
        self,
        model_name: str,
        offline_mode: bool | None = None,
        local_model_path: str | None = None,
    ) -> TaskResult:
        return self._qa.load_model(model_name, offline_mode, local_model_path)

    def get_stats(self) -> Dict[str, Any]:
        return self._qa.get_stats()

    def get_file_infos(self) -> List[Dict[str, Any]]:
        return self._qa.get_file_infos()