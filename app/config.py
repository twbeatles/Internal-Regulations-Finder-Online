# -*- coding: utf-8 -*-
import os
from typing import Dict, List


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _env_list(name: str, default_csv: str) -> List[str]:
    raw = os.environ.get(name, default_csv)
    if isinstance(raw, list):
        values = [str(v).strip() for v in raw]
        return [v for v in values if v]
    return [part.strip() for part in str(raw).split(",") if part.strip()]

class AppConfig:
    APP_NAME = "사내 규정 검색기"
    APP_VERSION = "2.6.1"  # v2.6.1 성능 최적화 리팩토링
    APP_ENV = os.environ.get("APP_ENV", "development").strip().lower()
    
    # 서버 설정
    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = 8080
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB

    # 보안/런타임 정책
    CORS_ALLOWED_ORIGINS = _env_list(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:8080,http://127.0.0.1:8080"
    )
    SESSION_COOKIE_HTTPONLY = _env_bool("SESSION_COOKIE_HTTPONLY", True)
    SESSION_COOKIE_SAMESITE = os.environ.get("SESSION_COOKIE_SAMESITE", "Lax").strip() or "Lax"
    SESSION_COOKIE_SECURE = _env_bool("SESSION_COOKIE_SECURE", APP_ENV == "production")
    
    # 오프라인 모드 설정 (폐쇄망 지원)
    OFFLINE_MODE = False  # True면 인터넷 연결 없이 로컬 모델만 사용
    LOCAL_MODEL_PATH = ""  # 사전 다운로드된 모델 폴더 경로 (빈 문자열이면 기본 경로 사용)
    
    # 임베딩 백엔드 설정
    EMBED_BACKEND: str = "onnx_fp32"  # "torch" | "onnx_fp32" | "onnx_int8"
    EMBED_NORMALIZE: bool = True   # L2 정규화 여부

    
    # AI 모델 설정
    AVAILABLE_MODELS: Dict[str, str] = {
        "SNU SBERT (고성능)": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        "BM-K Simal (균형)": "BM-K/ko-simal-roberta-base",
        "JHGan SBERT (빠름)": "jhgan/ko-sbert-nli"
    }
    DEFAULT_MODEL = "SNU SBERT (고성능)"
    
    # 파일 설정
    UPLOAD_FOLDER = "uploads"
    SUPPORTED_EXTENSIONS = {'.txt', '.docx', '.pdf', '.xlsx', '.xls', '.hwp'}
    
    # 검색 설정
    MAX_SEARCH_RESULTS = 10
    DEFAULT_SEARCH_RESULTS = 5
    
    # 청킹 설정
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 80
    VECTOR_WEIGHT = 0.7
    BM25_WEIGHT = 0.3
    
    # 동시성 설정
    MAX_WORKERS = 8       # CPU 기반 작업자 수 (문서 처리 등)
    SERVER_THREADS = 32   # 웹 서버 스레드 수 (Waitress) - 다수 접속 처리를 위해 증가
    REQUEST_TIMEOUT = 60  # 요청 대기 타임아웃 증가
    SEARCH_CACHE_SIZE = 1000 # 캐시 크기 대폭 증가 (자주 조회되는 검색어 부하 감소)
    MAX_CONCURRENT_SEARCHES = 10 # 동시 검색 실행 제한 (AI 모델 CPU 과부하 방지)
    RATE_LIMIT_PER_MINUTE = 300 # 사용자당 분당 요청 제한 완화
    
    # ========================================================================
    # 성능 최적화 설정 (v2.6.1)
    # ========================================================================
    
    # 검색 캐시 TTL (초) - 5분→10분으로 증가
    SEARCH_CACHE_TTL = 600
    
    # 적응형 TTL 활성화 - 자주 조회되는 쿼리의 TTL 자동 연장
    ADAPTIVE_CACHE_TTL = True
    
    # 하이브리드 검색 시 Vector/BM25 병렬 실행
    PARALLEL_SEARCH = True
    
    # 응답 압축 최소 크기 (바이트) - 이 크기 이상만 gzip 압축
    COMPRESS_MIN_SIZE = 500
    
    # 검색 결과 콘텐츠 미리보기 최대 길이
    MAX_CONTENT_PREVIEW = 1500

    # ZIP 업로드 제한 (압축폭탄 방지)
    ZIP_MAX_ENTRIES = _env_int("ZIP_MAX_ENTRIES", 1000)
    ZIP_MAX_UNCOMPRESSED_BYTES = _env_int("ZIP_MAX_UNCOMPRESSED_BYTES", 200 * 1024 * 1024)
    ZIP_MAX_SINGLE_FILE_BYTES = _env_int("ZIP_MAX_SINGLE_FILE_BYTES", 50 * 1024 * 1024)

