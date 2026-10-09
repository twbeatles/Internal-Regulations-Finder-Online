# -*- coding: utf-8 -*-
"""IP 기반 Rate Limiter."""
import threading
import time
from typing import Any, Dict

from app.utils import logger

class RateLimiter:
    """IP 기반 요청 제한기
    
    Features:
        - deque 기반 O(1) 시간 복잡도
        - 자동 만료 정리
        - 차단 통계 추적
    """
    
    def __init__(self, requests_per_minute: int = 60):
        from collections import deque
        self.requests: Dict[str, deque] = {}
        self.limit = requests_per_minute
        self._lock = threading.Lock()
        self._cleanup_interval = 60
        self._last_cleanup = time.time()
        # 통계
        self._total_allowed = 0
        self._total_blocked = 0
    
    def is_allowed(self, ip: str) -> bool:
        current_time = time.time()
        cutoff = current_time - 60
        
        with self._lock:
            # 주기적 정리
            if current_time - self._last_cleanup > self._cleanup_interval:
                self._cleanup(current_time)
                self._last_cleanup = current_time
            
            # deque 초기화
            if ip not in self.requests:
                from collections import deque
                self.requests[ip] = deque()
            
            req_deque = self.requests[ip]
            
            # 만료된 요청 제거 (deque 앞에서 O(1))
            while req_deque and req_deque[0] <= cutoff:
                req_deque.popleft()
            
            # 제한 확인
            if len(req_deque) >= self.limit:
                self._total_blocked += 1
                logger.warning(f"Rate limit exceeded for IP: {ip}")
                return False
            
            # 요청 기록
            req_deque.append(current_time)
            self._total_allowed += 1
            return True
    
    def _cleanup(self, current_time: float):
        """만료된 IP 엔트리 정리"""
        cutoff = current_time - 60
        expired_ips = [
            ip for ip, req_deque in self.requests.items() 
            if not req_deque or req_deque[-1] <= cutoff
        ]
        for ip in expired_ips:
            del self.requests[ip]
        if expired_ips:
            logger.debug(f"RateLimiter cleanup: {len(expired_ips)} IPs removed")
    
    def reset(self) -> None:
        """상태 초기화 (테스트·운영 리셋용)."""
        with self._lock:
            self.requests.clear()
            self._total_allowed = 0
            self._total_blocked = 0
            self._last_cleanup = time.time()

    def get_stats(self) -> Dict[str, Any]:
        """Rate limiter 통계"""
        with self._lock:
            return {
                'active_ips': len(self.requests),
                'limit_per_minute': self.limit,
                'total_allowed': self._total_allowed,
                'total_blocked': self._total_blocked,
                'block_rate': round(self._total_blocked / max(1, self._total_allowed + self._total_blocked) * 100, 2)
            }
