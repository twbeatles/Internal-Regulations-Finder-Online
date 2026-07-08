# -*- coding: utf-8 -*-
"""검색 서비스 패키지 (legacy 하이브리드 + 싱글톤)."""
from app.config import AppConfig
from app.services.document import (
    ArticleParser,
    DocumentComparator,
    DocumentExtractor,
    DocumentSplitter,
    TextHighlighter,
)
from app.services.search.bm25 import BM25Light
from app.services.search.cache import SearchCache
from app.services.search.history import SearchHistory
from app.services.search.langchain_lazy import (
    CharacterTextSplitter,
    Document,
    FAISS,
    HuggingFaceEmbeddings,
    _lazy_import_langchain,
)
from app.services.search.queue import SearchQueue
from app.services.search.rate_limiter import RateLimiter
from app.utils import FileInfo, FileStatus, FileUtils

# qa_system은 하위 호환 re-export 이후에 로드 (monkeypatch·순환 import 방지)
from app.services.search.qa_system import RegulationQASystem

rate_limiter = RateLimiter(AppConfig.RATE_LIMIT_PER_MINUTE)
search_queue = SearchQueue(AppConfig.MAX_CONCURRENT_SEARCHES)
qa_system = RegulationQASystem()

from app.services.search.qa_facade import RegulationQAFacade

qa_facade = RegulationQAFacade(qa_system)

__all__ = [
    "ArticleParser",
    "BM25Light",
    "CharacterTextSplitter",
    "Document",
    "DocumentComparator",
    "DocumentExtractor",
    "DocumentSplitter",
    "FAISS",
    "FileInfo",
    "FileStatus",
    "FileUtils",
    "HuggingFaceEmbeddings",
    "RateLimiter",
    "RegulationQAFacade",
    "RegulationQASystem",
    "SearchCache",
    "SearchHistory",
    "SearchQueue",
    "TextHighlighter",
    "_lazy_import_langchain",
    "qa_facade",
    "qa_system",
    "rate_limiter",
    "search_queue",
]
