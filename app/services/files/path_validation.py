# -*- coding: utf-8 -*-
"""폴더 경로 검증 (Path Traversal 완화)."""
from __future__ import annotations

import os
from typing import Optional, Tuple


def validate_folder_path(folder_path: str) -> Tuple[Optional[str], Optional[str]]:
    """관리자 폴더 동기화용 경로 검증.

    Returns:
        (normalized_absolute_path, error_message)
        성공 시 error_message는 None, 실패 시 path는 None.
    """
    if folder_path is None or not str(folder_path).strip():
        return None, "폴더 경로가 필요합니다"

    raw = str(folder_path).strip()
    # 정규화 전 의심 패턴 (상대 탈출·중복 구분자)
    dangerous_patterns = ["..", "//"]
    if any(p in raw for p in dangerous_patterns):
        return None, "유효하지 않은 경로 형식입니다"

    try:
        normalized = os.path.normpath(os.path.realpath(raw))
    except (ValueError, OSError, TypeError):
        return None, "유효하지 않은 경로입니다"

    if not os.path.exists(normalized):
        return None, f"폴더를 찾을 수 없습니다: {normalized}"

    if not os.path.isdir(normalized):
        return None, f"디렉토리가 아닙니다: {normalized}"

    return normalized, None
