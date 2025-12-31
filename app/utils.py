# -*- coding: utf-8 -*-
import os
import sys
import logging
import json
import datetime
import traceback
from typing import Optional, Tuple, Dict, Any, List, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime
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
    """로거 설정"""
    logger = logging.getLogger('RegSearchServer')
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    
    log_dir = os.path.join(get_app_directory(), 'logs')
        
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(
        os.path.join(log_dir, f'server_{datetime.now():%Y%m%d}.log'),
        encoding='utf-8'
    )
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
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
    success: bool
    message: str
    data: Any = None
    failed_items: List[str] = field(default_factory=list)

@dataclass
class FileInfo:
    path: str
    name: str
    extension: str
    size: int
    status: FileStatus = FileStatus.PENDING
    chunks: int = 0
    error: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "extension": self.extension,
            "size": self.size,
            "status": self.status.value,
            "chunks": self.chunks,
            "error": self.error
        }

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
