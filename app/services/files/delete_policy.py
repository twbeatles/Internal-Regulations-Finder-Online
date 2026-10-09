# -*- coding: utf-8 -*-
"""원본 파일 삭제 허용 경로 정책."""
from pathlib import Path
from typing import List

from app.config import AppConfig
from app.utils import get_app_directory


def get_delete_roots(qa_system) -> List[Path]:
    """인덱스/업로드 루트 등 원본 삭제가 허용되는 경로 목록."""
    roots: List[Path] = []
    uploads_root = (Path(get_app_directory()) / AppConfig.UPLOAD_FOLDER).resolve()
    roots.append(uploads_root)

    current_folder = getattr(qa_system, 'current_folder', '') or ''
    if current_folder:
        try:
            roots.append(Path(current_folder).resolve())
        except Exception:
            pass

    # 중복 제거
    unique = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def is_source_delete_allowed(target_path: Path, qa_system) -> bool:
    """대상 경로가 허용된 삭제 루트 하위인지 확인."""
    roots = get_delete_roots(qa_system)
    for root in roots:
        if root == target_path or root in target_path.parents:
            return True
    return False