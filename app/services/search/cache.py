# -*- coding: utf-8 -*-
"""검색 LRU 캐시."""
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.config import AppConfig
from app.utils import logger

class SearchCache:
    """LRU 기반 검색 캐시 (OrderedDict 사용으로 O(1) 성능)
    
    Features:
        - 적응형 TTL 기반 만료 (자주 조회되는 쿼리 TTL 연장)
        - LRU eviction
        - 캐시 히트율 통계
        - 메모리 사용량 추정
    
    Performance Optimizations (v2.6.1):
        - 기본 TTL 300s → 600s 증가
        - 적응형 TTL: 히트 횟수에 따라 최대 2배까지 TTL 연장
    """
    
    @dataclass
    class CacheEntry:
        timestamp: float
        result: Any
        hit_count: int = 0

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 600, adaptive_ttl: bool | None = None):
        from collections import OrderedDict
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl_seconds
        if adaptive_ttl is None:
            self.adaptive_ttl = getattr(AppConfig, 'ADAPTIVE_CACHE_TTL', True)
        else:
            self.adaptive_ttl = bool(adaptive_ttl)
        self.max_ttl_factor = 2.0
        self._lock = threading.Lock()
        # 통계
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    @staticmethod
    def _normalize_query(query: str) -> str:
        return ' '.join(str(query or '').lower().split())

    @staticmethod
    def _normalize_component(value: str | None) -> str:
        return str(value or '').strip().lower()

    def _make_key(
        self,
        query: str,
        k: int,
        hybrid: bool,
        sort_by: str = 'relevance',
        filter_file: str | None = None,
        filter_file_id: str | None = None,
    ) -> str:
        """캐시 키 생성 (쿼리 정규화로 히트율 향상)
        
        정규화:
        - 소문자 변환
        - 연속 공백을 단일 공백으로
        - 앞뒤 공백 제거
        """
        normalized = self._normalize_query(query)
        sort_key = self._normalize_component(sort_by or 'relevance')
        filter_name_key = self._normalize_component(filter_file)
        filter_id_key = self._normalize_component(filter_file_id)
        return f"{normalized}|{k}|{hybrid}|{sort_key}|{filter_name_key}|{filter_id_key}"

    def _effective_ttl(self, entry: "SearchCache.CacheEntry") -> float:
        if not self.adaptive_ttl:
            return float(self.ttl)
        # 첫 캐시 히트에서는 TTL을 늘리지 않아 테스트/기대 동작과 일치.
        extra_hits = max(0, int(entry.hit_count) - 1)
        factor = min(self.max_ttl_factor, 1.0 + extra_hits * 0.1)
        return float(self.ttl) * factor
    
    def get(
        self,
        query: str,
        k: int,
        hybrid: bool,
        sort_by: str = 'relevance',
        filter_file: str | None = None,
        filter_file_id: str | None = None,
    ) -> Optional[Any]:
        key = self._make_key(query, k, hybrid, sort_by, filter_file, filter_file_id)
        with self._lock:
            entry = self.cache.get(key)
            if entry is not None:
                now = time.time()
                if now - entry.timestamp < self._effective_ttl(entry):
                    entry.hit_count += 1
                    self.cache.move_to_end(key)
                    self._hits += 1
                    return entry.result
                del self.cache[key]
                self._evictions += 1
            self._misses += 1
        return None
    
    def set(
        self,
        query: str,
        k: int,
        hybrid: bool,
        result: Any,
        sort_by: str = 'relevance',
        filter_file: str | None = None,
        filter_file_id: str | None = None,
    ):
        key = self._make_key(query, k, hybrid, sort_by, filter_file, filter_file_id)
        with self._lock:
            # 이미 존재하면 히트 카운트 유지하며 갱신
            old_hit_count = 0
            if key in self.cache:
                old_entry = self.cache[key]
                old_hit_count = int(old_entry.hit_count)
                del self.cache[key]
            # 크기 초과 시 가장 오래된 항목 제거 (O(1))
            while len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
                self._evictions += 1
            self.cache[key] = SearchCache.CacheEntry(
                timestamp=time.time(),
                result=result,
                hit_count=old_hit_count
            )
    
    def clear(self):
        with self._lock:
            self.cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
    
    def size(self) -> int:
        """현재 캐시 크기"""
        with self._lock:
            return len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'evictions': self._evictions,
                'hit_rate': round(hit_rate, 2),
                'ttl_seconds': self.ttl,
                'adaptive_ttl': self.adaptive_ttl,
            }
    
    def invalidate_by_file(self, filename: str) -> int:
        """특정 파일 관련 캐시 무효화
        
        Args:
            filename: 무효화할 파일명
            
        Returns:
            무효화된 항목 수
        """
        count = 0
        with self._lock:
            # 캐시된 결과에서 해당 파일 포함 항목 제거
            keys_to_remove = []
            for key, entry in self.cache.items():
                result = entry.result
                if isinstance(result, list):
                    for item in result:
                        if isinstance(item, dict) and item.get('source') == filename:
                            keys_to_remove.append(key)
                            break
            for key in keys_to_remove:
                del self.cache[key]
                count += 1
        if count > 0:
            logger.info(f"캐시 무효화: {filename} 관련 {count}개 항목 제거")
        return count
