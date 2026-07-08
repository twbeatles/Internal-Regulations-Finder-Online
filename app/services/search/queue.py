# -*- coding: utf-8 -*-
"""동시 검색 큐."""
import threading
from typing import Dict

from app.utils import logger

class SearchQueue:
    def __init__(self, max_concurrent: int = 10):
        self._semaphore = threading.Semaphore(max_concurrent)
        self._active_count = 0
        self._lock = threading.Lock()
        self._total_processed = 0
        self._total_rejected = 0
    
    def acquire(self, timeout: float = 30.0) -> bool:
        acquired = self._semaphore.acquire(timeout=timeout)
        if acquired:
            with self._lock:
                self._active_count += 1
        else:
            with self._lock:
                self._total_rejected += 1
            logger.warning("Search queue full, request rejected")
        return acquired
    
    def release(self):
        self._semaphore.release()
        with self._lock:
            self._active_count = max(0, self._active_count - 1)
            self._total_processed += 1
    
    def get_stats(self) -> Dict:
        with self._lock:
            return {
                'active': self._active_count,
                'processed': self._total_processed,
                'rejected': self._total_rejected
            }
