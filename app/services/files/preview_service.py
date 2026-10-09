# -*- coding: utf-8 -*-
"""파일 미리보기 캐시 및 페이로드 생성."""
import threading
from collections import OrderedDict

from app.services.document import DocumentExtractor
from app.utils import FileUtils


class PreviewService:
    """문서 미리보기 LRU 캐시 및 응답 페이로드 빌더."""

    _CACHE_MAX_SIZE = 128

    def __init__(self, qa_system, extractor: DocumentExtractor | None = None):
        self._qa_system = qa_system
        self._extractor = extractor or DocumentExtractor()
        self._cache_lock = threading.Lock()
        self._cache: OrderedDict = OrderedDict()

    def get(self, cache_key: str):
        """캐시에서 미리보기 페이로드 조회."""
        with self._cache_lock:
            payload = self._cache.get(cache_key)
            if payload is not None:
                self._cache.move_to_end(cache_key)
            return payload

    def set(self, cache_key: str, payload: dict):
        """캐시에 미리보기 페이로드 저장 (LRU)."""
        with self._cache_lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
            while len(self._cache) >= self._CACHE_MAX_SIZE:
                self._cache.popitem(last=False)
            self._cache[cache_key] = payload

    @property
    def extractor(self) -> DocumentExtractor:
        """문서 추출기 (비-TXT 미리보기용)."""
        return self._extractor

    def build_payload(
        self,
        filename: str,
        target_path: str,
        content: str,
        length: int,
        *,
        metadata: dict | None = None,
        tables: list | None = None,
        diagnostics: dict | None = None,
    ) -> dict:
        """미리보기 API 응답 페이로드 생성."""
        text = content or ''
        preview = text[:length]
        truncated = len(text) > length
        info = self._qa_system.file_infos.get(target_path)
        file_id = FileUtils.make_file_id(target_path)
        return {
            'success': True,
            'file_id': file_id,
            'filename': filename,
            'preview': preview,
            'content': preview,  # 프론트엔드 하위호환
            'total_length': len(text),
            'truncated': truncated,
            'is_truncated': truncated,  # 프론트엔드 하위호환
            'status': getattr(getattr(info, 'status', None), 'value', '완료'),
            'chunks': getattr(info, 'chunks', 0),
            'metadata': metadata or {},
            'tables': tables or [],
            'table_count': len(tables or []),
            'diagnostics': diagnostics or {},
        }