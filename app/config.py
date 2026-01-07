# -*- coding: utf-8 -*-
from typing import Dict, Set

class AppConfig:
    APP_NAME = "사내 규정 검색기"
    APP_VERSION = "2.3 (웹 서버)"  # v2.3 디버깅 및 리팩토링
    
    # 서버 설정
    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = 8080
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    
    # 오프라인 모드 설정 (폐쇄망 지원)
    OFFLINE_MODE = False  # True면 인터넷 연결 없이 로컬 모델만 사용
    LOCAL_MODEL_PATH = ""  # 사전 다운로드된 모델 폴더 경로 (빈 문자열이면 기본 경로 사용)
    
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
