# -*- coding: utf-8 -*-
"""
사내 규정 검색기 - 커스텀 예외 클래스

표준화된 예외 처리를 위한 커스텀 예외 정의
"""

from typing import Optional, Any


class RegSearchError(Exception):
    """규정 검색기 기본 예외 클래스"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.details = details
    
    def __str__(self):
        if self.details:
            return f"{self.message} (상세: {self.details})"
        return self.message


# ============================================================================
# 모델 관련 예외
# ============================================================================

class ModelError(RegSearchError):
    """모델 관련 기본 예외"""
    pass


class ModelNotLoadedError(ModelError):
    """모델이 로드되지 않았을 때"""
    
    def __init__(self, message: str = "모델이 로드되지 않았습니다"):
        super().__init__(message)


class ModelLoadError(ModelError):
    """모델 로딩 실패"""
    
    def __init__(self, model_name: str, reason: str = ""):
        message = f"모델 '{model_name}' 로드 실패"
        if reason:
            message += f": {reason}"
        super().__init__(message, details={'model': model_name, 'reason': reason})


class ModelOfflineError(ModelError):
    """오프라인 모드에서 모델을 찾을 수 없음"""
    
    def __init__(self, model_path: str):
        super().__init__(
            f"오프라인 모드에서 로컬 모델을 찾을 수 없습니다: {model_path}",
            details={'path': model_path}
        )


# ============================================================================
# 문서 처리 관련 예외
# ============================================================================

class DocumentError(RegSearchError):
    """문서 처리 기본 예외"""
    pass


class DocumentNotFoundError(DocumentError):
    """문서 파일을 찾을 수 없음"""
    
    def __init__(self, path: str):
        super().__init__(f"파일을 찾을 수 없습니다: {path}", details={'path': path})


class DocumentExtractionError(DocumentError):
    """문서 텍스트 추출 실패"""
    
    def __init__(self, path: str, reason: str = ""):
        message = f"텍스트 추출 실패: {path}"
        if reason:
            message += f" ({reason})"
        super().__init__(message, details={'path': path, 'reason': reason})


class DocumentTypeError(DocumentError):
    """지원하지 않는 문서 형식"""
    
    def __init__(self, extension: str):
        super().__init__(
            f"지원하지 않는 파일 형식입니다: {extension}",
            details={'extension': extension}
        )


class DocumentEmptyError(DocumentError):
    """문서 내용이 비어있음"""
    
    def __init__(self, path: str):
        super().__init__(f"파일 내용이 비어있습니다: {path}", details={'path': path})


# ============================================================================
# 인덱싱 관련 예외
# ============================================================================

class IndexingError(RegSearchError):
    """인덱싱 기본 예외"""
    pass


class IndexBuildError(IndexingError):
    """인덱스 생성 실패"""
    
    def __init__(self, index_type: str, reason: str = ""):
        message = f"{index_type} 인덱스 생성 실패"
        if reason:
            message += f": {reason}"
        super().__init__(message, details={'type': index_type, 'reason': reason})


class CacheError(IndexingError):
    """캐시 관련 오류"""
    
    def __init__(self, operation: str, reason: str = ""):
        message = f"캐시 {operation} 실패"
        if reason:
            message += f": {reason}"
        super().__init__(message, details={'operation': operation, 'reason': reason})


# ============================================================================
# 검색 관련 예외
# ============================================================================

class SearchError(RegSearchError):
    """검색 기본 예외"""
    pass


class SearchTimeoutError(SearchError):
    """검색 타임아웃"""
    
    def __init__(self, timeout_seconds: float):
        super().__init__(
            f"검색 요청 대기 시간 초과 ({timeout_seconds}초)",
            details={'timeout': timeout_seconds}
        )


class SearchRateLimitError(SearchError):
    """요청 제한 초과"""
    
    def __init__(self, ip: str, limit: int):
        super().__init__(
            "요청이 너무 많습니다. 잠시 후 다시 시도하세요",
            details={'ip': ip, 'limit': limit}
        )


class SearchQueueFullError(SearchError):
    """검색 큐 포화"""
    
    def __init__(self, max_concurrent: int):
        super().__init__(
            "검색 큐가 가득 찼습니다. 잠시 후 다시 시도하세요",
            details={'max_concurrent': max_concurrent}
        )


# ============================================================================
# 폴더/경로 관련 예외
# ============================================================================

class FolderError(RegSearchError):
    """폴더 관련 기본 예외"""
    pass


class FolderNotFoundError(FolderError):
    """폴더를 찾을 수 없음"""
    
    def __init__(self, path: str):
        super().__init__(f"폴더를 찾을 수 없습니다: {path}", details={'path': path})


class FolderNotInitializedError(FolderError):
    """폴더가 초기화되지 않음"""
    
    def __init__(self):
        super().__init__("초기화된 폴더가 없습니다. 먼저 폴더를 선택해주세요")


# ============================================================================
# 인증 관련 예외
# ============================================================================

class AuthError(RegSearchError):
    """인증 관련 기본 예외"""
    pass


class AuthenticationError(AuthError):
    """인증 실패"""
    
    def __init__(self, reason: str = "비밀번호가 일치하지 않습니다"):
        super().__init__(reason)


class AuthorizationError(AuthError):
    """권한 부족"""
    
    def __init__(self, action: str = ""):
        message = "권한이 없습니다"
        if action:
            message += f": {action}"
        super().__init__(message)
