# -*- coding: utf-8 -*-
import os
import sys
import logging
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
import numpy as np

from app.config import AppConfig

def get_app_directory() -> str:
    """애플리케이션 실행 디렉토리 반환 (PyInstaller 호환)"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    # 현재 파일이 app/utils.py 이므로, 부모의 부모 디렉토리가 루트
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

def setup_logger() -> logging.Logger:
    """로거 설정
    
    구조화된 로깅 포맷:
    - 타임스탬프
    - 로그 레벨
    - 모듈/라인 정보
    - 메시지
    
    환경 변수:
    - LOG_LEVEL: 로그 레벨 설정 (DEBUG, INFO, WARNING, ERROR)
    """
    logger = logging.getLogger('RegSearchServer')
    if logger.handlers:
        return logger
    
    # 환경 변수에서 로그 레벨 읽기 (기본: INFO)
    log_level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)
    
    # 로그 디렉토리 생성
    log_dir = os.path.join(get_app_directory(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 구조화된 로그 포맷
    detailed_format = '%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(message)s'
    simple_format = '%(asctime)s | %(levelname)-8s | %(message)s'
    
    # 파일 핸들러 (상세 로그)
    fh = logging.FileHandler(
        os.path.join(log_dir, f'server_{datetime.now():%Y%m%d}.log'),
        encoding='utf-8'
    )
    fh.setFormatter(logging.Formatter(detailed_format))
    fh.setLevel(logging.DEBUG)  # 파일에는 모든 레벨 기록
    logger.addHandler(fh)
    
    # 콘솔 핸들러 (간단 로그)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(logging.Formatter(simple_format))
    logger.addHandler(ch)
    
    return logger

logger = setup_logger()

class FileStatus(Enum):
    PENDING = "대기"
    PROCESSING = "처리중"
    SUCCESS = "완료"
    FAILED = "실패"
    CACHED = "캐시"

@dataclass
class TaskResult:
    """작업 결과를 담는 데이터 클래스"""
    success: bool
    message: str
    data: Any = None
    failed_items: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """API 응답용 딕셔너리 변환"""
        result = {
            'success': self.success,
            'message': self.message
        }
        if self.data is not None:
            # 검색 결과의 경우 프론트엔드가 'results' 필드를 기대함
            # 호환성을 위해 'data'와 'results' 모두 포함
            result['data'] = self.data
            result['results'] = self.data
        if self.failed_items:
            result['failed_items'] = self.failed_items
        return result

@dataclass
class FileInfo:
    path: str
    name: str = ""
    extension: str = ""
    size: int = 0
    status: FileStatus = FileStatus.PENDING
    chunks: int = 0
    error: str = ""
    mod_time: float = 0.0  # 파일 수정 시간 (timestamp)
    version: str = ""      # 파일 버전
    
    def __post_init__(self):
        """데이터 검증 및 자동 필드 설정"""
        # name이 비어있으면 path에서 추출
        if not self.name and self.path:
            self.name = os.path.basename(self.path)
        # extension이 비어있으면 name에서 추출
        if not self.extension and self.name:
            self.extension = os.path.splitext(self.name)[1].lower()
    
    def to_dict(self) -> Dict:
        """JSON 직렬화용 딕셔너리 변환"""
        return {
            "path": self.path,
            "name": self.name,
            "extension": self.extension,
            "size": self.size,
            "status": self.status.value,
            "chunks": self.chunks,
            "error": self.error,
            "mod_time": self.mod_time,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FileInfo':
        """딕셔너리에서 FileInfo 생성 (누락된 필드 처리)"""
        # status 문자열을 FileStatus enum으로 변환
        status = data.get('status', 'PENDING')
        if isinstance(status, str):
            try:
                status = FileStatus(status)
            except ValueError:
                status = FileStatus.PENDING
        
        return cls(
            path=data.get('path', data.get('name', '')),  # path가 없으면 name 사용
            name=data.get('name', ''),
            extension=data.get('extension', ''),
            size=data.get('size', 0),
            status=status,
            chunks=data.get('chunks', 0),
            error=data.get('error', ''),
            mod_time=data.get('mod_time', 0.0),
            version=data.get('version', '')
        )

class CustomJSONEncoder(json.JSONEncoder):
    """NumPy 타입 및 집합(Set)을 처리하는 커스텀 JSON 인코더"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        return super(CustomJSONEncoder, self).default(obj)


class MemoryMonitor:
    """애플리케이션 메모리 사용량 모니터링
    
    Features:
        - 현재 프로세스 메모리 사용량
        - GC 통계
        - 메모리 경고 임계값
    """
    
    _psutil_available = None
    
    @classmethod
    def _check_psutil(cls) -> bool:
        if cls._psutil_available is None:
            try:
                import psutil
                cls._psutil_available = True
            except ImportError:
                cls._psutil_available = False
        return cls._psutil_available
    
    @classmethod
    def get_memory_usage(cls) -> Dict[str, Any]:
        """현재 메모리 사용량 반환 (MB 단위)"""
        import gc
        
        result = {
            'gc_objects': len(gc.get_objects()),
            'gc_stats': gc.get_stats() if hasattr(gc, 'get_stats') else None
        }
        
        if cls._check_psutil():
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            result.update({
                'rss_mb': round(mem_info.rss / 1024 / 1024, 2),
                'vms_mb': round(mem_info.vms / 1024 / 1024, 2),
                'percent': round(process.memory_percent(), 2)
            })
            
            # 시스템 메모리
            sys_mem = psutil.virtual_memory()
            result['system'] = {
                'total_gb': round(sys_mem.total / 1024 / 1024 / 1024, 2),
                'available_gb': round(sys_mem.available / 1024 / 1024 / 1024, 2),
                'percent_used': sys_mem.percent
            }
        else:
            result['note'] = 'psutil not installed - limited memory info'
        
        return result
    
    @classmethod
    def check_memory_warning(cls, threshold_mb: int = 1024) -> Optional[str]:
        """메모리 사용량 경고 체크
        
        Args:
            threshold_mb: 경고 임계값 (MB)
            
        Returns:
            경고 메시지 또는 None
        """
        if not cls._check_psutil():
            return None
        
        import psutil
        process = psutil.Process()
        rss_mb = process.memory_info().rss / 1024 / 1024
        
        if rss_mb > threshold_mb:
            warning = f"메모리 사용량 경고: {rss_mb:.0f}MB (임계값: {threshold_mb}MB)"
            logger.warning(warning)
            return warning
        return None
    
    @classmethod
    def force_gc(cls) -> Dict[str, Any]:
        """강제 가비지 컬렉션 수행
        
        Returns:
            GC 전후 통계
        """
        import gc
        
        before = len(gc.get_objects())
        collected = gc.collect()
        after = len(gc.get_objects())
        
        result = {
            'collected': collected,
            'objects_before': before,
            'objects_after': after,
            'freed': before - after
        }
        
        logger.info(f"GC 수행: {collected}개 수집, {before - after}개 객체 해제")
        return result

class FileUtils:
    @staticmethod
    def safe_read(path: str, encoding: str = 'utf-8') -> Tuple[Optional[str], Optional[str]]:
        try:
            with open(path, 'r', encoding=encoding, errors='ignore') as f:
                return f.read(), None
        except Exception as e:
            return None, str(e)
    
    @staticmethod
    def get_metadata(path: str) -> Optional[Dict]:
        try:
            stat = os.stat(path)
            return {'size': stat.st_size, 'mtime': stat.st_mtime}
        except OSError as e:
            logger.debug(f"파일 메타데이터 조회 실패: {path} - {e}")
            return None
    
    @staticmethod
    def format_size(size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"
    
    @staticmethod
    def allowed_file(filename: str) -> bool:
        ext = os.path.splitext(filename)[1].lower()
        return ext in AppConfig.SUPPORTED_EXTENSIONS


# ============================================================================
# API 응답 헬퍼 (표준화된 응답 포맷)
# ============================================================================

def api_success(message: str = "성공", data: Any = None, **kwargs) -> Dict:
    """성공 응답 생성
    
    Args:
        message: 성공 메시지
        data: 응답 데이터 (선택)
        **kwargs: 추가 필드
        
    Returns:
        표준화된 성공 응답 딕셔너리
    """
    response = {
        'success': True,
        'message': message
    }
    if data is not None:
        response['data'] = data
    response.update(kwargs)
    return response


def api_error(message: str, error_code: str = None, status_code: int = 400, **kwargs) -> Tuple[Dict, int]:
    """에러 응답 생성
    
    Args:
        message: 에러 메시지
        error_code: 에러 코드 (선택, 예: 'MODEL_NOT_LOADED')
        status_code: HTTP 상태 코드
        **kwargs: 추가 필드
        
    Returns:
        (에러 응답 딕셔너리, HTTP 상태 코드) 튜플
    """
    response = {
        'success': False,
        'message': message
    }
    if error_code:
        response['error_code'] = error_code
    response.update(kwargs)
    return response, status_code


def api_paginated(items: List, total: int, page: int = 1, per_page: int = 10) -> Dict:
    """페이지네이션된 응답 생성
    
    Args:
        items: 현재 페이지 아이템 리스트
        total: 전체 아이템 수
        page: 현재 페이지 번호
        per_page: 페이지당 아이템 수
        
    Returns:
        페이지네이션 정보가 포함된 응답 딕셔너리
    """
    return {
        'success': True,
        'data': items,
        'pagination': {
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page
        }
    }

