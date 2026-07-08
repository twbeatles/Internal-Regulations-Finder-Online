# -*- coding: utf-8 -*-
"""파일 작업 동시성 제어."""
import threading

file_lock = threading.Lock()
LOCK_TIMEOUT = 30  # 파일 잠금 타임아웃 (초)


def acquire_file_lock(timeout=LOCK_TIMEOUT):
    """파일 잠금 획득 (타임아웃 지원)

    Returns:
        bool: 잠금 획득 성공 여부
    """
    return file_lock.acquire(timeout=timeout)