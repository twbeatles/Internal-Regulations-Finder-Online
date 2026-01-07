# -*- coding: utf-8 -*-
"""
사내 규정 검색기 - 상수 및 에러 메시지 정의

이 모듈은 애플리케이션 전역에서 사용되는 상수들을 중앙 집중화합니다.
"""

# ============================================================================
# 에러 메시지
# ============================================================================

class ErrorMessages:
    """한글 에러 메시지 상수"""
    
    # 모델 관련
    MODEL_NOT_LOADED = "모델이 로드되지 않았습니다"
    MODEL_LOAD_FAILED = "모델 로드 실패"
    MODEL_OFFLINE_NOT_FOUND = "오프라인 모드에서 로컬 모델을 찾을 수 없습니다"
    
    # 파일 관련
    FILE_NOT_FOUND = "파일을 찾을 수 없습니다"
    FILE_EMPTY = "파일이 비어있습니다"
    FILE_NAME_EMPTY = "파일명이 비어있습니다"
    FILE_TYPE_NOT_SUPPORTED = "지원하지 않는 파일 형식입니다"
    FILE_UPLOAD_FAILED = "파일 업로드 실패"
    FILE_EXTRACTION_FAILED = "파일에서 텍스트를 추출할 수 없습니다"
    
    # 폴더 관련
    FOLDER_NOT_FOUND = "폴더를 찾을 수 없습니다"
    FOLDER_INVALID = "유효하지 않은 경로입니다"
    FOLDER_NOT_DIRECTORY = "디렉토리가 아닙니다"
    FOLDER_NOT_INITIALIZED = "초기화된 폴더가 없습니다"
    
    # 검색 관련
    SEARCH_FAILED = "검색 실패"
    SEARCH_RATE_LIMITED = "요청이 너무 많습니다. 잠시 후 다시 시도하세요"
    SEARCH_TIMEOUT = "검색 요청 대기 시간 초과"
    SEARCH_QUEUE_FULL = "검색 큐가 가득 찼습니다"
    
    # 캐시 관련
    CACHE_LOAD_FAILED = "캐시 로드 실패"
    CACHE_SAVE_FAILED = "캐시 저장 실패"
    
    # 인증 관련
    AUTH_PASSWORD_WRONG = "비밀번호가 일치하지 않습니다"
    AUTH_REQUIRED = "인증이 필요합니다"
    
    # 기타
    INTERNAL_ERROR = "내부 오류가 발생했습니다"
    LANGCHAIN_LOAD_FAILED = "LangChain 로드 실패"
    PYTORCH_LOAD_FAILED = "PyTorch 로드 실패"


class SuccessMessages:
    """한글 성공 메시지 상수"""
    
    MODEL_LOADED = "모델 로드 완료"
    FILE_UPLOADED = "파일 업로드 성공"
    FILE_DELETED = "파일이 삭제되었습니다"
    FILE_PROCESSED = "파일 처리 완료"
    SEARCH_COMPLETED = "검색 완료"
    SERVER_INITIALIZED = "서버 초기화 완료"
    SYNC_STARTED = "동기화가 시작되었습니다"
    SYNC_STOPPED = "동기화가 중지되었습니다"


class InfoMessages:
    """정보 메시지 상수"""
    
    MODEL_LOADING = "AI 모델 초기화 중..."
    MODEL_DOWNLOADING = "모델 다운로드 중... (최초 실행 시 시간이 걸릴 수 있습니다)"
    SERVER_STARTING = "서버 초기화 시작..."
    DOCUMENTS_PROCESSING = "문서 처리 중..."
    INDEXING = "인덱싱 중..."
    CACHE_USING = "캐시에서 결과 반환"


# ============================================================================
# 매직 넘버 상수
# ============================================================================

class Limits:
    """제한값 상수"""
    
    # 파일 크기
    MAX_FILE_SIZE_MB = 50
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    
    # 청킹
    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 50
    MAX_CHUNK_SIZE = 2000
    MIN_CHUNK_SIZE = 100
    
    # 검색
    DEFAULT_SEARCH_RESULTS = 5
    MAX_SEARCH_RESULTS = 20
    MIN_QUERY_LENGTH = 2
    
    # 캐시
    DEFAULT_CACHE_SIZE = 100
    DEFAULT_CACHE_TTL_SECONDS = 300
    
    # 동시성
    DEFAULT_MAX_WORKERS = 8
    DEFAULT_RATE_LIMIT = 60  # per minute
    DEFAULT_SEARCH_TIMEOUT = 30.0


class Weights:
    """가중치 상수"""
    
    VECTOR_WEIGHT = 0.7
    BM25_WEIGHT = 0.3
    
    # BM25 파라미터
    BM25_K1 = 1.5
    BM25_B = 0.75


class FileExtensions:
    """지원 파일 확장자"""
    
    TEXT = '.txt'
    DOCX = '.docx'
    PDF = '.pdf'
    XLSX = '.xlsx'
    XLS = '.xls'
    HWP = '.hwp'
    ZIP = '.zip'
    
    SUPPORTED = {TEXT, DOCX, PDF, XLSX, XLS, HWP}
    ARCHIVE = {ZIP}


# ============================================================================
# 정규식 패턴 (사전 컴파일)
# ============================================================================
import re

class Patterns:
    """정규식 패턴 상수 (사전 컴파일)"""
    
    # 토큰화 패턴
    TOKENIZE = re.compile(r'[^\w\s가-힣]')
    
    # 조문 패턴
    ARTICLE_NUMBERED = re.compile(r'^제\s*(\d+)\s*조', re.MULTILINE)
    ARTICLE_KOREAN = re.compile(r'^제\s*([일이삼사오육칠팔구십백천]+)\s*조', re.MULTILINE)
    ARTICLE_SIMPLE = re.compile(r'^\[조\s*(\d+)\]', re.MULTILINE)
    
    # 장/절 패턴
    CHAPTER = re.compile(r'^제\s*(\d+)\s*장', re.MULTILINE)
    SECTION = re.compile(r'^제\s*(\d+)\s*절', re.MULTILINE)
    
    # 공백 정규화
    WHITESPACE = re.compile(r'\s+')
    
    # 파일명 정리
    FILENAME_UNSAFE = re.compile(r'[<>:"/\\|?*]')


# ============================================================================
# API 관련 상수
# ============================================================================

class ApiRoutes:
    """API 라우트 경로"""
    
    # 검색
    SEARCH = '/api/search'
    SUGGEST = '/api/suggest'
    
    # 파일
    FILES = '/api/files'
    UPLOAD = '/api/upload'
    
    # 시스템
    STATUS = '/api/status'
    MODELS = '/api/models'
    STATS = '/api/stats'
    HEALTH = '/api/health'
    
    # 동기화
    SYNC_STATUS = '/api/sync/status'
    SYNC_START = '/api/sync/start'
    SYNC_STOP = '/api/sync/stop'


class HttpStatus:
    """HTTP 상태 코드"""
    
    OK = 200
    CREATED = 201
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    TOO_MANY_REQUESTS = 429
    INTERNAL_ERROR = 500
    SERVICE_UNAVAILABLE = 503
