# -*- coding: utf-8 -*-
"""파일 경로 조회 및 참조 생성."""
import os

from app.exceptions import DocumentError, DocumentNotFoundError
from app.utils import FileUtils


class FilePathResolver:
    """qa_system 인덱스 기준 파일 경로 해석."""

    def __init__(self, qa_system):
        self._qa_system = qa_system

    def find_by_filename(self, filename: str) -> str:
        """파일명으로 전체 경로 찾기

        Args:
            filename: 찾을 파일명 (basename)

        Returns:
            전체 경로

        Raises:
            DocumentNotFoundError: 파일을 찾을 수 없는 경우
        """
        # Route uses <path:filename>; normalize to basename to avoid path injection.
        filename = os.path.basename((filename or "").replace("\\", "/"))
        if not filename:
            raise DocumentNotFoundError(filename)

        matches = []
        for fp in self._qa_system.file_infos.keys():
            if os.path.basename(fp) == filename:
                matches.append(fp)
        if not matches:
            raise DocumentNotFoundError(filename)
        if len(matches) > 1:
            raise DocumentError(f"동일한 파일명이 여러 개 존재합니다: {filename}")
        return matches[0]

    def find_by_id(self, file_id: str) -> str:
        """file_id로 전체 경로 찾기."""
        file_id = str(file_id or "").strip()
        if not file_id:
            raise DocumentNotFoundError(file_id)
        for fp in self._qa_system.file_infos.keys():
            if FileUtils.make_file_id(fp) == file_id:
                return fp
        raise DocumentNotFoundError(file_id)

    def resolve(self, file_id: str | None = None, filename: str | None = None) -> tuple[str, str, str]:
        """file_id 또는 filename으로 대상 경로·이름·ID를 해석."""
        if file_id is not None:
            target_path = self.find_by_id(file_id)
        elif filename is not None:
            target_path = self.find_by_filename(filename)
        else:
            raise DocumentNotFoundError("")
        resolved_name = os.path.basename(target_path)
        resolved_id = FileUtils.make_file_id(target_path)
        return target_path, resolved_name, resolved_id

    def build_file_ref(self, path: str) -> dict:
        """프론트엔드용 파일 참조 dict 생성."""
        name = os.path.basename(path)
        file_id = FileUtils.make_file_id(path)
        return {
            "file_id": file_id,
            "name": name,
            "label": name,
            "path": path
        }