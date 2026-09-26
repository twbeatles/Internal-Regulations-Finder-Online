# -*- coding: utf-8 -*-
import os
from typing import ClassVar


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


def _env_list(name: str, default_csv: str) -> list[str]:
    raw = os.environ.get(name, default_csv)
    if isinstance(raw, list):
        values = [str(v).strip() for v in raw]
        return [v for v in values if v]
    return [part.strip() for part in str(raw).split(",") if part.strip()]

class AppConfig:
    APP_NAME: ClassVar[str] = "사내 규정 검색기"
    APP_VERSION: ClassVar[str] = "2.6.1"  # v2.6.1 성능 최적화 리팩토링
    APP_ENV: ClassVar[str] = os.environ.get("APP_ENV", "development").strip().lower()

    # 서버 설정
    SERVER_HOST: ClassVar[str] = "0.0.0.0"
    SERVER_PORT: ClassVar[int] = 8080
    MAX_CONTENT_LENGTH: ClassVar[int] = 50 * 1024 * 1024  # 50MB

    # 보안/런타임 정책
    CORS_ALLOWED_ORIGINS: ClassVar[list[str]] = _env_list(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:8080,http://127.0.0.1:8080"
    )
    SESSION_COOKIE_HTTPONLY: ClassVar[bool] = _env_bool("SESSION_COOKIE_HTTPONLY", True)
    SESSION_COOKIE_SAMESITE: ClassVar[str] = os.environ.get("SESSION_COOKIE_SAMESITE", "Lax").strip() or "Lax"
    SESSION_COOKIE_SECURE: ClassVar[bool] = _env_bool("SESSION_COOKIE_SECURE", APP_ENV == "production")

    # 오프라인 모드 설정 (폐쇄망 지원)
    OFFLINE_MODE: ClassVar[bool] = False  # True면 인터넷 연결 없이 로컬 모델만 사용
    LOCAL_MODEL_PATH: ClassVar[str] = ""  # 사전 다운로드된 모델 폴더 경로 (빈 문자열이면 기본 경로 사용)

    # 임베딩 백엔드 설정
    EMBED_BACKEND: ClassVar[str] = "onnx_fp32"  # "torch" | "onnx_fp32" | "onnx_int8"
    EMBED_NORMALIZE: ClassVar[bool] = True   # L2 정규화 여부
    EMBED_BATCH_SIZE: ClassVar[int] = _env_int("EMBED_BATCH_SIZE", 32)  # ONNX 배치 인코딩 크기 (대량 인덱싱 피크 메모리 억제)


    # AI 모델 설정 (2026-09 전면 개편: 2021~2022년식 SBERT 3종 교체)
    # 선정 기준: 한국어 검색 품질(Ko-MTEB/MIRACL-ko) + CPU/ONNX 경량 실행 + 라이선스
    AVAILABLE_MODELS: ClassVar[dict[str, str]] = {
        "E5-Small-KO (경량·기본)": "dragonkue/multilingual-e5-small-ko",
        "E5-Small-Multi (경량·다국어)": "intfloat/multilingual-e5-small",
        "BGE-M3 (고품질·8192)": "BAAI/bge-m3",
        "KURE-v1 (한국어 최고)": "nlpai-lab/KURE-v1",
        "Qwen3-0.6B (최신·고성능)": "Qwen/Qwen3-Embedding-0.6B",
    }
    DEFAULT_MODEL: ClassVar[str] = "E5-Small-KO (경량·기본)"

    # 모델별 임베딩 프리셋 (E5 prefix 규약, 디코더형 풀링 강제 등)
    # query_prefix/passage_prefix: E5 계열 검색 품질에 필수인 입력 접두사
    # force_backend: "torch" 지정 시 전역 EMBED_BACKEND과 무관하게 torch로 로드
    #   (Qwen3-Embedding은 causal-LM + last-token 풀링이라 mean-pooling ONNX와 비호환)
    EMBED_MODEL_PRESETS: ClassVar[dict[str, dict[str, str]]] = {
        "dragonkue/multilingual-e5-small-ko": {
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
            "note": "118M급·384차원, 한국어 특화, CPU/ONNX 최적",
        },
        "intfloat/multilingual-e5-small": {
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
            "note": "118M급·384차원, 다국어 범용, CPU/ONNX 최적",
        },
        "BAAI/bge-m3": {
            "query_prefix": "",
            "passage_prefix": "",
            "note": "568M·1024차원·8192토큰, dense+sparse+multi-vector",
        },
        "nlpai-lab/KURE-v1": {
            "query_prefix": "",
            "passage_prefix": "",
            "note": "568M·1024차원, BGE-M3 한국어 튜닝, Ko 검색 최상",
        },
        "Qwen/Qwen3-Embedding-0.6B": {
            "query_prefix": "",
            "passage_prefix": "",
            "force_backend": "torch",
            "note": "0.6B·1024차원·32K, 최신 다국어 SOTA급, torch 전용",
        },
    }

    # 파일 설정
    UPLOAD_FOLDER: ClassVar[str] = "uploads"
    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {'.txt', '.docx', '.pdf', '.xlsx', '.xls', '.hwp', '.hwpx'}

    # 검색 설정
    MAX_SEARCH_RESULTS: ClassVar[int] = 10
    DEFAULT_SEARCH_RESULTS: ClassVar[int] = 5

    # 청킹 설정
    CHUNK_SIZE: ClassVar[int] = 800
    CHUNK_OVERLAP: ClassVar[int] = 80
    VECTOR_WEIGHT: ClassVar[float] = 0.7
    BM25_WEIGHT: ClassVar[float] = 0.3

    # 동시성 설정
    MAX_WORKERS: ClassVar[int] = 8       # CPU 기반 작업자 수 (문서 처리 등)
    SERVER_THREADS: ClassVar[int] = 32   # 웹 서버 스레드 수 (Waitress) - 다수 접속 처리를 위해 증가
    REQUEST_TIMEOUT: ClassVar[int] = 60  # 요청 대기 타임아웃 증가
    SEARCH_CACHE_SIZE: ClassVar[int] = 1000 # 캐시 크기 대폭 증가 (자주 조회되는 검색어 부하 감소)
    MAX_CONCURRENT_SEARCHES: ClassVar[int] = 10 # 동시 검색 실행 제한 (AI 모델 CPU 과부하 방지)
    RATE_LIMIT_PER_MINUTE: ClassVar[int] = 300 # 사용자당 분당 요청 제한 완화

    # ========================================================================
    # 성능 최적화 설정 (v2.6.1)
    # ========================================================================

    # 검색 캐시 TTL (초) - 5분→10분으로 증가
    SEARCH_CACHE_TTL: ClassVar[int] = 600

    # 적응형 TTL 활성화 - 자주 조회되는 쿼리의 TTL 자동 연장
    ADAPTIVE_CACHE_TTL: ClassVar[bool] = True

    # 하이브리드 검색 시 Vector/BM25 병렬 실행
    PARALLEL_SEARCH: ClassVar[bool] = True

    # 응답 압축 최소 크기 (바이트) - 이 크기 이상만 gzip 압축
    COMPRESS_MIN_SIZE: ClassVar[int] = 500

    # 검색 결과 콘텐츠 미리보기 최대 길이
    MAX_CONTENT_PREVIEW: ClassVar[int] = 1500

    # ZIP 업로드 제한 (압축폭탄 방지)
    ZIP_MAX_ENTRIES: ClassVar[int] = _env_int("ZIP_MAX_ENTRIES", 1000)
    ZIP_MAX_UNCOMPRESSED_BYTES: ClassVar[int] = _env_int("ZIP_MAX_UNCOMPRESSED_BYTES", 200 * 1024 * 1024)
    ZIP_MAX_SINGLE_FILE_BYTES: ClassVar[int] = _env_int("ZIP_MAX_SINGLE_FILE_BYTES", 50 * 1024 * 1024)
