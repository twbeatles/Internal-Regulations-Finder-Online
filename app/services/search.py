# -*- coding: utf-8 -*-
import os
import time
import json
import threading
import logging
import traceback
import math
import re
import gc
import shutil
import hashlib  # _get_cache_dir에서 사용
import importlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# ============================================================================
# Lazy Import 패턴 - 무거운 라이브러리는 실제 사용 시점에 로드
# GUI 시작 시간 단축을 위해 PyTorch, LangChain 등은 지연 로딩
# ============================================================================

# 이 변수들은 실제 사용 시점에 _lazy_import_*() 함수로 로드됨
CharacterTextSplitter: Any | None = None
Document: Any | None = None
HuggingFaceEmbeddings: Any | None = None
FAISS: Any | None = None
_lazy_imports_loaded = False


def _import_attr(module_name: str, attr_name: str) -> Any | None:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    return getattr(module, attr_name, None)

def _lazy_import_langchain():
    """LangChain 관련 라이브러리 지연 로드"""
    global CharacterTextSplitter, Document, HuggingFaceEmbeddings, FAISS, _lazy_imports_loaded
    
    if _lazy_imports_loaded:
        return
    
    CharacterTextSplitter = _import_attr('langchain_text_splitters', 'CharacterTextSplitter')
    if CharacterTextSplitter is None:
        CharacterTextSplitter = _import_attr('langchain.text_splitter', 'CharacterTextSplitter')

    Document = _import_attr('langchain_core.documents', 'Document')
    if Document is None:
        Document = _import_attr('langchain.docstore.document', 'Document')

    HuggingFaceEmbeddings = _import_attr('langchain_huggingface', 'HuggingFaceEmbeddings')
    if HuggingFaceEmbeddings is None:
        HuggingFaceEmbeddings = _import_attr('langchain_community.embeddings', 'HuggingFaceEmbeddings')
    if HuggingFaceEmbeddings is None:
        HuggingFaceEmbeddings = _import_attr('langchain.embeddings', 'HuggingFaceEmbeddings')

    FAISS = _import_attr('langchain_community.vectorstores', 'FAISS')
    if FAISS is None:
        FAISS = _import_attr('langchain.vectorstores', 'FAISS')
    
    _lazy_imports_loaded = True

from app.config import AppConfig
from app.utils import logger, FileInfo, FileStatus, TaskResult, FileUtils, get_app_directory
from app.constants import ErrorMessages, SuccessMessages, InfoMessages, Limits, Weights, Patterns
from app.exceptions import (
    ModelNotLoadedError, ModelLoadError, ModelOfflineError,
    SearchError, SearchTimeoutError, SearchRateLimitError, SearchQueueFullError,
    DocumentExtractionError, IndexingError, CacheError
)
from app.services.db import db
from app.services.document import DocumentExtractor, TextHighlighter, DocumentSplitter, ArticleParser
from app.services.metadata import TagManager
from app.services.file_manager import RevisionTracker

# ============================================================================
# BM25 경량 구현 (성능 최적화 버전)
# ============================================================================

# 정규식 패턴은 constants.py의 Patterns 클래스에서 가져옴
_TOKENIZE_PATTERN = Patterns.TOKENIZE

class BM25Light:
    """
    BM25 경량 구현 (스레드 안전, 성능 최적화)
    
    최적화 내용:
    - 정규식 사전 컴파일로 토큰화 성능 향상
    - term frequency를 fit() 시점에 미리 계산하여 검색 성능 40-50% 향상
    - __slots__ 사용으로 메모리 효율화
    """
    # NOTE: Keep this class fast and memory-efficient; it runs on every search.
    __slots__ = ['k1', 'b', 'doc_lens', 'doc_len_norm', 'avgdl', 'idf', 'N', '_lock', 'postings']

    # Faster token extraction than "regex sub + split" for large texts.
    _TOKEN_EXTRACT = re.compile(r"[0-9A-Za-z가-힣_]{2,}")
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lens: List[int] = []
        self.doc_len_norm: List[float] = []  # precomputed length normalization per doc
        self.avgdl = 0.0
        self.idf: Dict[str, float] = {}
        self.N = 0
        self._lock = threading.RLock()
        # Inverted index: term -> list[(doc_idx, tf)]
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """토큰화 (성능 최적화: findall 기반)"""
        if not text:
            return []
        return self._TOKEN_EXTRACT.findall(text.lower())
    
    def fit(self, docs: List[str]):
        """
        문서 인덱싱

        Query-time 성능 최적화:
        - inverted index(postings) 구축: 쿼리 토큰이 포함된 문서만 스코어링
        - doc length normalization 사전 계산
        """
        with self._lock:
            self.doc_lens = []
            self.doc_len_norm = []
            df = Counter()

            self.postings = {}

            for idx, doc in enumerate(docs):
                tokens = self._tokenize(doc)
                dl = len(tokens)
                self.doc_lens.append(dl)
                if dl == 0:
                    continue

                tf = Counter(tokens)
                df.update(tf.keys())

                # Fill postings with per-doc term frequency.
                for term, freq in tf.items():
                    lst = self.postings.get(term)
                    if lst is None:
                        self.postings[term] = [(idx, int(freq))]
                    else:
                        lst.append((idx, int(freq)))
            
            self.N = len(docs)
            self.avgdl = sum(self.doc_lens) / self.N if self.N else 0
            self.idf = {t: math.log((self.N - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}
            del df

            # Precompute length normalization per doc (avoid division in hot path).
            if self.avgdl > 0:
                k1 = self.k1
                b = self.b
                avgdl = self.avgdl
                self.doc_len_norm = [k1 * (1 - b + b * (dl / avgdl)) for dl in self.doc_lens]
            gc.collect()
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """검색 수행 (postings 기반: 쿼리 토큰이 포함된 문서만 스코어링)"""
        with self._lock:
            if not self.postings or not query:
                return []
            q_tokens = self._tokenize(query)
            if not q_tokens:
                return []
            
            # 쿼리 토큰 정규화: 중복 제거 + postings 있는 토큰만
            terms = []
            seen = set()
            for t in q_tokens:
                if t in seen:
                    continue
                seen.add(t)
                if t in self.idf and t in self.postings:
                    terms.append(t)
            if not terms:
                return []

            # Hot-path locals
            k1 = self.k1
            k1p1 = k1 + 1.0
            doc_len_norm = self.doc_len_norm
            idf_map = self.idf
            postings = self.postings

            scores: Dict[int, float] = {}

            for term in terms:
                idf = idf_map.get(term, 0.0)
                if idf <= 0:
                    continue
                for doc_idx, tf in postings.get(term, ()):
                    # Skip empty docs or missing norms defensively.
                    if doc_idx >= len(doc_len_norm):
                        continue
                    denom = tf + doc_len_norm[doc_idx]
                    if denom <= 0:
                        continue
                    inc = idf * (tf * k1p1) / denom
                    scores[doc_idx] = scores.get(doc_idx, 0.0) + inc

            if not scores:
                return []

            # Avoid full sort when candidate set is large.
            import heapq
            return heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
    
    def _score_optimized(self, query_idf: Dict[str, float], doc_tf: Counter, doc_len: int) -> float:
        """
        최적화된 BM25 점수 계산
        - 사전 계산된 term frequency 사용
        - 쿼리 IDF 미리 필터링
        """
        if self.avgdl == 0:
            return 0.0
        
        score = 0.0
        len_norm = self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
        
        for term, idf in query_idf.items():
            tf = doc_tf.get(term, 0)
            if tf > 0:
                num = tf * (self.k1 + 1)
                den = tf + len_norm
                score += idf * num / den
        
        return score
    
    def clear(self):
        """인덱스 초기화"""
        with self._lock:
            self.doc_lens.clear()
            self.doc_len_norm.clear()
            self.postings.clear()
            self.idf.clear()
            gc.collect()

# ============================================================================
# 검색 캐시, Rate Limiter, Search Queue
# ============================================================================
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

class SearchQueue:
    def __init__(self, max_concurrent: int = 10):
        self._semaphore = threading.Semaphore(max_concurrent)
        self._active_count = 0
        self._lock = threading.Lock()
        self._total_processed = 0
        self._total_rejected = 0
    
    def acquire(self, timeout: float = 30.0) -> bool:
        acquired = self._semaphore.acquire(timeout=timeout)
        if acquired:
            with self._lock:
                self._active_count += 1
        else:
            with self._lock:
                self._total_rejected += 1
            logger.warning("Search queue full, request rejected")
        return acquired
    
    def release(self):
        self._semaphore.release()
        with self._lock:
            self._active_count = max(0, self._active_count - 1)
            self._total_processed += 1
    
    def get_stats(self) -> Dict:
        with self._lock:
            return {
                'active': self._active_count,
                'processed': self._total_processed,
                'rejected': self._total_rejected
            }

# ============================================================================
# 검색 히스토리 (DB 연동)
# ============================================================================
class SearchHistory:
    def add(self, query: str):
        """검색 히스토리에 쿼리 추가 (5분 내 중복 방지)"""
        query = query.strip()
        if len(query) < 2:
            return
        try:
            # 5분 이내 동일 쿼리 중복 저장 방지
            recent = db.fetchone(
                """SELECT id FROM search_history 
                   WHERE query = ? AND timestamp > datetime('now', '-5 minutes')""",
                (query,)
            )
            if recent:
                return  # 최근 동일 쿼리 존재, 저장 스킵
            db.execute("INSERT INTO search_history (query) VALUES (?)", (query,))
        except Exception:
            pass
            
    def get_recent(self, limit: int = 10) -> List[str]:
        rows = db.fetchall("SELECT query, MAX(timestamp) as ts FROM search_history GROUP BY query ORDER BY ts DESC LIMIT ?", (limit,))
        return [r['query'] for r in rows]
            
    def get_popular(self, limit: int = 10) -> List[Dict[str, Any]]:
        rows = db.fetchall("SELECT query, COUNT(*) as cnt FROM search_history GROUP BY query ORDER BY cnt DESC LIMIT ?", (limit,))
        return [{'query': r['query'], 'count': int(r['cnt'])} for r in rows]
            
    def suggest(self, prefix: str, limit: int = 5) -> List[str]:
        rows = db.fetchall("SELECT DISTINCT query FROM search_history WHERE query LIKE ? ORDER BY timestamp DESC LIMIT ?", (f"{prefix}%", limit))
        return [r['query'] for r in rows]
            
    def clear(self):
        db.execute("DELETE FROM search_history")

# ============================================================================
# RegulationQASystem
# ============================================================================
class RegulationQASystem:
    _CACHE_KEY_MODE = "relative_path_v1"

    def __init__(self):
        self.vector_store: Any = None
        self.embedding_model: Any = None
        self.model_id: str | None = None
        self.model_name = ""
        self.embed_backend = ""  # 현재 사용 중인 백엔드
        self.embed_normalize = True  # 현재 정규화 설정
        self.offline_mode = bool(getattr(AppConfig, 'OFFLINE_MODE', False))
        self.local_model_path = getattr(AppConfig, 'LOCAL_MODEL_PATH', '') or ''
        self.extractor = DocumentExtractor()
        self.cache_path = os.path.join(os.path.dirname(get_app_directory()), "reg_qa_server_v10")
        if not os.path.exists(self.cache_path):
             # Try temp dir if app dir not writable or structure different
             self.cache_path = os.path.join(os.environ.get('TEMP', '/tmp'), "reg_qa_server_v10")
             
        self.bm25: BM25Light | None = None
        self.documents: List[str] = []
        self.doc_meta: List[Dict[str, Any]] = []
        self.file_infos: Dict[str, FileInfo] = {}
        self.file_details: Dict[str, Dict[str, Any]] = {}
        self.current_folder = ""
        self._lock = threading.RLock()
        self._search_cache = SearchCache(AppConfig.SEARCH_CACHE_SIZE)
        self._search_history = SearchHistory()
        self._keyword_cache: List[str] = []
        self._executor = ThreadPoolExecutor(max_workers=AppConfig.MAX_WORKERS)
        # Reuse a small pool for per-request hybrid search to avoid per-query ThreadPool creation.
        self._search_executor = ThreadPoolExecutor(max_workers=2)
        self._is_ready = False
        self._is_loading = False
        self._load_progress = ""
        self._load_error = ""
        self._cancel_event = threading.Event()
        self._cancel_reason = ""
        
        # v2.0 Components
        self.tag_manager = TagManager()
        self.revision_tracker = RevisionTracker()
        self.article_parser = ArticleParser()
        self.doc_splitter = DocumentSplitter()

    def _normalize_doc_meta_locked(self) -> None:
        normalized_meta: List[Dict[str, Any]] = []
        for doc_id, content in enumerate(self.documents):
            meta = {}
            if doc_id < len(self.doc_meta) and isinstance(self.doc_meta[doc_id], dict):
                meta = dict(self.doc_meta[doc_id])

            path = str(meta.get('path', '') or '')
            source = str(meta.get('source', '') or (os.path.basename(path) if path else '?'))
            normalized_meta.append({
                **meta,
                'doc_id': doc_id,
                'source': source,
                'path': path,
                'file_id': meta.get('file_id') or (FileUtils.make_file_id(path) if path else ''),
            })
        self.doc_meta = normalized_meta

    def _rebuild_vector_store_locked(self) -> None:
        if not self.documents:
            self.vector_store = None
            return

        if not (self.embedding_model and FAISS and Document):
            self.vector_store = None
            return

        try:
            self._normalize_doc_meta_locked()
            docs = [
                Document(page_content=self.documents[idx], metadata=dict(self.doc_meta[idx]))
                for idx in range(len(self.documents))
            ]
            self.vector_store = FAISS.from_documents(docs, self.embedding_model) if docs else None
        except Exception as e:
            logger.warning(f"벡터 인덱스 재구축 실패: {e}")
            self.vector_store = None

    def _clear_index_locked(self, *, preserve_folder: bool = True) -> None:
        self.file_infos.clear()
        self.file_details.clear()
        self.documents = []
        self.doc_meta = []
        self.vector_store = None
        self.bm25 = None
        self._keyword_cache = []
        self._search_cache.clear()
        if not preserve_folder:
            self.current_folder = ""

    def clear_index(self, *, preserve_folder: bool = True) -> None:
        with self._lock:
            self._clear_index_locked(preserve_folder=preserve_folder)

    def remove_file_from_index(self, target_path: str, resolved_name: str, resolved_id: str) -> bool:
        with self._lock:
            deleted_from_index = False

            if target_path in self.file_infos:
                del self.file_infos[target_path]
                deleted_from_index = True

            if target_path in self.file_details:
                self.file_details.pop(target_path, None)

            if self.documents and self.doc_meta:
                indices_to_remove = [
                    i for i, meta in enumerate(self.doc_meta)
                    if meta.get('path') == target_path
                    or meta.get('file_id') == resolved_id
                    or (
                        not meta.get('path')
                        and not meta.get('file_id')
                        and meta.get('source') == resolved_name
                    )
                ]
                for idx in reversed(indices_to_remove):
                    if idx < len(self.documents):
                        del self.documents[idx]
                    if idx < len(self.doc_meta):
                        del self.doc_meta[idx]
                if indices_to_remove:
                    deleted_from_index = True

            self._keyword_cache = []
            self._search_cache.clear()
            self._normalize_doc_meta_locked()
            self._build_bm25()
            self._rebuild_vector_store_locked()
            return deleted_from_index

    def _get_cache_entry_key(self, folder: str, file_path: str) -> str:
        folder_abs = os.path.abspath(str(folder or ""))
        file_abs = os.path.abspath(str(file_path or ""))
        try:
            rel_path = os.path.relpath(file_abs, folder_abs)
            if rel_path.startswith('..'):
                raise ValueError("outside-folder")
            candidate = rel_path
        except Exception:
            candidate = file_abs
        normalized = os.path.normcase(os.path.normpath(candidate))
        return normalized.replace("\\", "/")
    
    def get_keywords(self, limit: int = 50) -> List[str]:
        if not self._keyword_cache and self.documents:
            self._keyword_cache = TextHighlighter.extract_keywords(self.documents, limit)
        return self._keyword_cache[:limit]
    
    @property
    def is_ready(self) -> bool:
        # AI 모델이 있거나, BM25가 있으면 ready (Lite 모드 지원)
        return self._is_ready or (self.bm25 is not None and len(self.documents) > 0)
    
    @property
    def is_loading(self) -> bool:
        return self._is_loading
    
    @property
    def load_progress(self) -> str:
        return self._load_progress
    
    @property
    def load_error(self) -> str:
        return self._load_error

    def request_sync_stop(self) -> TaskResult:
        """백그라운드 동기화 중단 요청"""
        if not self._is_loading:
            return TaskResult(True, "진행 중인 동기화 작업이 없습니다")

        self._cancel_reason = "사용자 요청으로 동기화가 중단되었습니다"
        self._cancel_event.set()
        self._load_progress = "중단 요청 처리 중..."
        logger.info("동기화 중단 요청 수신")
        return TaskResult(True, "동기화 중단 요청을 수신했습니다")

    def _cancelled_result(self, progress_cb=None) -> TaskResult:
        message = self._cancel_reason or "사용자 요청으로 동기화가 중단되었습니다"
        if progress_cb:
            progress_cb(100, "중단됨")
        self._load_error = message

        # 중단 시점까지 확보된 문서로 BM25 재구축하여 내부 상태 일관성 유지
        try:
            if self.documents:
                self._build_bm25()
        except Exception as e:
            logger.warning(f"중단 처리 중 BM25 재구축 실패: {e}")

        logger.info(message)
        return TaskResult(False, message)
    
    def load_model(
        self,
        model_name: str,
        offline_mode: bool | None = None,
        local_model_path: str | None = None
    ) -> TaskResult:
        if self._is_loading:
            return TaskResult(False, "이미 모델을 로딩 중입니다")
        
        is_offline = offline_mode if offline_mode is not None else bool(AppConfig.OFFLINE_MODE)
        model_path_override = local_model_path if local_model_path is not None else str(AppConfig.LOCAL_MODEL_PATH)
        model_id = AppConfig.AVAILABLE_MODELS.get(model_name, AppConfig.AVAILABLE_MODELS[AppConfig.DEFAULT_MODEL])
        embed_backend = str(getattr(AppConfig, 'EMBED_BACKEND', 'torch'))
        embed_normalize = bool(getattr(AppConfig, 'EMBED_NORMALIZE', True))
        self._load_error = ""
        self.offline_mode = bool(is_offline)
        self.local_model_path = model_path_override or ""
        
        try:
            self._is_loading = True
            self._load_progress = "라이브러리 로드 중..."
            logger.info("라이브러리 로드 중...")
            
            # Lazy import 실행 (무거운 라이브러리 로드)
            _lazy_import_langchain()
            
            self._load_progress = "AI 모델 초기화 중..."
            logger.info(f"모델 로드 시작: {model_name} ({model_id}), backend={embed_backend}")
            
            self._load_progress = "모델 다운로드 중... (최초 실행 시 시간이 걸릴 수 있습니다)"
            
            # ====================================================================
            # 임베딩 백엔드 팩토리 사용
            # ====================================================================
            from app.services.embeddings_backends import create_embeddings
            
            self.embedding_model = create_embeddings(
                model_name=model_name,
                model_id_or_path=model_id,
                backend=embed_backend,
                normalize=embed_normalize,
                offline_mode=is_offline,
                local_model_path=model_path_override if model_path_override else None
            )
            
            self.model_id = model_id
            self.model_name = model_name
            self.embed_backend = embed_backend
            self.embed_normalize = embed_normalize
            self._load_progress = "모델 로드 완료"
            self._is_ready = True
            
            logger.info(f"✅ 모델 로드 완료: {model_name} (backend={embed_backend}, normalize={embed_normalize})")
            return TaskResult(True, "모델 로드 완료")
            
        except (ModelLoadError, ModelOfflineError) as e:
            self._load_error = str(e)
            logger.error(f"모델 로드 오류: {e}")
            return TaskResult(False, str(e))
        except Exception as e:
            self._load_error = str(e)
            logger.error(f"모델 로드 예상치 못한 오류: {e}")
            return TaskResult(False, f"모델 로드 오류: {e}")
        finally:
            self._is_loading = False
            
    def _get_cache_dir(self, folder: str) -> str:
        """FAISS 캐시 디렉토리 경로 생성
        
        BM25-only 모드(모델 미로드)도 캐시 경로를 생성합니다.
        """
        model_key = self.model_id or "bm25-only"
        h1 = hashlib.md5(model_key.encode()).hexdigest()[:6]
        h2 = hashlib.md5(folder.encode()).hexdigest()[:6]
        return os.path.join(self.cache_path, f"{h2}_{h1}")

    def _remember_file_details(self, file_path: str, extracted) -> Dict[str, Any]:
        details = {
            'metadata': dict(getattr(extracted, 'metadata', {}) or {}),
            'tables': list(getattr(extracted, 'table_dicts', lambda: [])() or []),
            'diagnostics': dict(getattr(extracted, 'diagnostics', {}) or {}),
        }
        self.file_details[file_path] = details
        return details

    def _build_chunk_meta(
        self,
        *,
        doc_id: int,
        filename: str,
        file_path: str,
        file_id: str,
        chunk_id: int,
        total_chunks: int,
        details: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            'doc_id': doc_id,
            'source': filename,
            'path': file_path,
            'file_id': file_id,
            'chunk_id': chunk_id,
            'total_chunks': total_chunks,
        }
        if details:
            metadata = details.get('metadata')
            diagnostics = details.get('diagnostics')
            tables = details.get('tables')
            if metadata:
                meta['document_metadata'] = metadata
            if diagnostics:
                meta['diagnostics'] = diagnostics
            if tables:
                meta['table_count'] = len(tables)
        return meta
    
    def process_single_file(self, file_path: str) -> TaskResult:
        """단일 파일 업로드 후 즉시 인덱싱
        
        Args:
            file_path: 처리할 파일의 전체 경로
            
        Returns:
            TaskResult: 처리 결과
        """
        if not os.path.exists(file_path):
            return TaskResult(False, f"파일을 찾을 수 없습니다: {file_path}")
        
        try:
            logger.info(f"단일 파일 처리 시작: {file_path}")
            
            # 파일 추출
            extracted = self.extractor.extract_with_details(file_path)
            text = extracted.text
            error = extracted.error
            if error:
                return TaskResult(False, f"파일 추출 오류: {error}")
            if not text or not text.strip():
                return TaskResult(False, "파일에서 텍스트를 추출할 수 없습니다")
            
            # 문서 분할
            splitter = DocumentSplitter(
                chunk_size=AppConfig.CHUNK_SIZE,
                chunk_overlap=AppConfig.CHUNK_OVERLAP
            )
            chunks = splitter.split(text)
            
            if not chunks:
                return TaskResult(False, "문서 분할 결과가 없습니다")
            
            filename = os.path.basename(file_path)
            file_id = FileUtils.make_file_id(file_path)
            file_size = os.path.getsize(file_path)

            with self._lock:
                self._keyword_cache = []
                details = self._remember_file_details(file_path, extracted)
                
                # file_infos에 추가
                self.file_infos[file_path] = FileInfo(
                    path=file_path,
                    size=file_size,
                    status=FileStatus.SUCCESS,
                    chunks=len(chunks),
                    mod_time=os.path.getmtime(file_path)
                )
                
                # 문서 및 메타데이터 추가
                for i, chunk in enumerate(chunks):
                    doc_id = len(self.documents)
                    self.documents.append(chunk)
                    self.doc_meta.append(
                        self._build_chunk_meta(
                            doc_id=doc_id,
                            filename=filename,
                            file_path=file_path,
                            file_id=file_id,
                            chunk_id=i,
                            total_chunks=len(chunks),
                            details=details,
                        )
                    )
                
                # 벡터스토어 업데이트 (있으면 추가, 없으면 생성)
                if self.embedding_model and FAISS and Document:
                    try:
                        docs = []
                        base_doc_id = len(self.documents) - len(chunks)
                        for i, chunk in enumerate(chunks):
                            docs.append(
                                Document(
                                    page_content=chunk,
                                    metadata={
                                        **self._build_chunk_meta(
                                            doc_id=base_doc_id + i,
                                            filename=filename,
                                            file_path=file_path,
                                            file_id=file_id,
                                            chunk_id=i,
                                            total_chunks=len(chunks),
                                            details=details,
                                        ),
                                    },
                                )
                            )
                        
                        if self.vector_store:
                            self.vector_store.add_documents(docs)
                        else:
                            self.vector_store = FAISS.from_documents(docs, self.embedding_model)
                        
                        logger.info(f"벡터스토어 업데이트 완료: {len(chunks)} 청크")
                    except Exception as e:
                        logger.error(f"벡터스토어 업데이트 오류: {e}")
                        self._rebuild_vector_store_locked()
                
                # BM25 재구축
                self._build_bm25()
            
            logger.info(f"✅ 단일 파일 처리 완료: {filename} ({len(chunks)} 청크)")
            return TaskResult(True, f"파일 처리 완료: {filename} ({len(chunks)} 청크)", {
                'filename': filename,
                'chunks': len(chunks),
                'size': file_size,
                'metadata': details.get('metadata', {}),
                'diagnostics': details.get('diagnostics', {}),
                'table_count': len(details.get('tables', [])),
            })
            
        except Exception as e:
            logger.error(f"단일 파일 처리 오류: {e}")
            import traceback
            traceback.print_exc()
            return TaskResult(False, f"파일 처리 오류: {str(e)}")
    
    def process_documents(self, folder: str, files: List[str], progress_cb=None, force_reindex: bool = False) -> TaskResult:
        # AI 모델 없어도 BM25로 동작 가능 (Lite 모드 지원)
        if not self.embedding_model:
            logger.info("AI 모델 없음 - BM25 전용 모드로 문서 처리")
        
        if self._cancel_event.is_set():
            return self._cancelled_result(progress_cb)
        
        with self._lock:
            return self._process_internal(folder, files, progress_cb, force_reindex=force_reindex)
    
    def _process_internal(self, folder: str, files: List[str], progress_cb, force_reindex: bool = False) -> TaskResult:
        # 벡터 인덱싱 사용 시에만 LangChain 의존성 로드
        if self.embedding_model:
            _lazy_import_langchain()
        if self._cancel_event.is_set():
            return self._cancelled_result(progress_cb)
        
        self.current_folder = folder
        cache_dir = self._get_cache_dir(folder)
        self.file_infos.clear()
        self._search_cache.clear()
        self._keyword_cache = []
        
        # Init FileInfo
        for fp in files:
            meta = FileUtils.get_metadata(fp)
            self.file_infos[fp] = FileInfo(
                path=fp,
                name=os.path.basename(fp),
                extension=os.path.splitext(fp)[1].lower(),
                size=int(meta['size']) if meta and 'size' in meta else 0
            )
            
        if progress_cb:
            progress_cb(5, "캐시 확인...")
        if self._cancel_event.is_set():
            return self._cancelled_result(progress_cb)

        cache_info = {}
        to_process, cached = [], []

        if force_reindex:
            # 강제 재인덱싱: 기존 캐시를 우회하고 전체 파일 재처리.
            self.vector_store = None
            if os.path.exists(cache_dir):
                try:
                    shutil.rmtree(cache_dir)
                except Exception as e:
                    logger.debug(f"강제 재인덱싱 캐시 삭제 실패(무시): {e}")
            to_process = list(files)
        else:
            cache_info = self._load_cache_info(cache_dir)
            for fp in files:
                cache_key = self._get_cache_entry_key(folder, fp)
                meta = FileUtils.get_metadata(fp)
                if meta and cache_key in cache_info:
                    cm = cache_info[cache_key]
                    if cm.get('size') == meta['size'] and cm.get('mtime') == meta['mtime']:
                        cached.append(fp)
                        self.file_infos[fp].status = FileStatus.CACHED
                        self.file_infos[fp].chunks = cm.get('chunks', 0)
                        continue
                to_process.append(fp)
            
        self.documents, self.doc_meta = [], []
        self.file_details = {}
        if self._cancel_event.is_set():
            return self._cancelled_result(progress_cb)
        
        # Load Cache
        if (
            (not force_reindex)
            and self.embedding_model
            and FAISS
            and cached
            and os.path.exists(os.path.join(cache_dir, "index.faiss"))
        ):
            try:
                if progress_cb: progress_cb(10, "캐시 로드...")
                self.vector_store = FAISS.load_local(
                    cache_dir, self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                docs_path = os.path.join(cache_dir, "docs.json")
                if os.path.exists(docs_path):
                    with open(docs_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.documents = data.get('docs', [])
                        self.doc_meta = data.get('meta', [])
                        if isinstance(self.doc_meta, list):
                            self._normalize_doc_meta_locked()
            except Exception as e:
                logger.warning(f"캐시 로드 실패: {e}")
                to_process, cached = files, []
                self.vector_store = None

        # BM25-only 모드에서도 docs.json 기반 캐시 로드는 가능해야 함.
        if (not force_reindex) and cached and not self.documents:
            docs_path = os.path.join(cache_dir, "docs.json")
            if os.path.exists(docs_path):
                try:
                    with open(docs_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.documents = data.get('docs', [])
                        self.doc_meta = data.get('meta', [])
                        if isinstance(self.doc_meta, list):
                            self._normalize_doc_meta_locked()
                except Exception as e:
                    logger.warning(f"docs.json 캐시 로드 실패: {e}")
        
        if not to_process:
            self._build_bm25()
            if progress_cb: progress_cb(100, "완료!")
            return TaskResult(True, f"캐시에서 {len(cached)}개 파일 로드", 
                            {'chunks': len(self.documents), 'cached': len(cached), 'new': 0})
        
        splitter = DocumentSplitter(chunk_size=AppConfig.CHUNK_SIZE, chunk_overlap=AppConfig.CHUNK_OVERLAP)
        failed, new_docs, new_cache_info = [], [], {}
        
        # Parallel Extraction
        def extract_file(fp: str) -> Tuple[str, str, Any, Dict[str, Any] | None]:
            fname = os.path.basename(fp)
            try:
                extracted = self.extractor.extract_with_details(fp)
                meta = FileUtils.get_metadata(fp)
                return fp, fname, extracted, meta
            except Exception as e:
                return fp, fname, e, None
        
        extracted_results = []
        if progress_cb: progress_cb(15, f"문서 추출 중... (병렬 처리)")
        
        from concurrent.futures import as_completed

        with ThreadPoolExecutor(max_workers=min(AppConfig.MAX_WORKERS, len(to_process))) as executor:
            futures = {executor.submit(extract_file, fp): fp for fp in to_process}
            completed = 0
            for future in as_completed(futures):
                if self._cancel_event.is_set():
                    for pending in futures:
                        if not pending.done():
                            pending.cancel()
                    return self._cancelled_result(progress_cb)
                fp = futures[future]
                fname = os.path.basename(fp)
                try:
                    result = future.result(timeout=60)
                    extracted_results.append(result)
                except Exception as e:
                    extracted_results.append((fp, fname, RuntimeError(f"추출 실패/타임아웃: {e}"), None))
                completed += 1
                if progress_cb and (completed % 5 == 0 or completed == len(to_process)):
                    progress = 15 + int((completed / len(to_process)) * 30)
                    progress_cb(progress, f"추출 완료: {completed}/{len(to_process)}")
        
        if progress_cb: progress_cb(50, "텍스트 청킹 중...")
        
        for fp, fname, extracted, meta in extracted_results:
            if self._cancel_event.is_set():
                return self._cancelled_result(progress_cb)
            self.file_infos[fp].status = FileStatus.PROCESSING
            file_id = FileUtils.make_file_id(fp)
            if isinstance(extracted, Exception):
                failed.append(f"{fname} ({extracted})")
                self.file_infos[fp].status = FileStatus.FAILED
                self.file_infos[fp].error = str(extracted)
                continue
            content = extracted.text
            error = extracted.error
            if error:
                failed.append(f"{fname} ({error})")
                self.file_infos[fp].status = FileStatus.FAILED
                self.file_infos[fp].error = error
                continue
            if not content or not content.strip():
                failed.append(f"{fname} (빈 파일)")
                self.file_infos[fp].status = FileStatus.FAILED
                self.file_infos[fp].error = "빈 파일"
                continue
                
            try:
                details = self._remember_file_details(fp, extracted)
                chunks = splitter.split(content)
                chunk_count = 0
                for chunk in chunks:
                    if self._cancel_event.is_set():
                        return self._cancelled_result(progress_cb)
                    chunk_text = chunk.strip()
                    if not chunk_text:
                        continue
                    doc_id = len(self.documents)
                    if self.embedding_model and FAISS and Document:
                        new_docs.append(
                            Document(
                                page_content=chunk_text,
                                metadata=self._build_chunk_meta(
                                    doc_id=doc_id,
                                    filename=fname,
                                    file_path=fp,
                                    file_id=file_id,
                                    chunk_id=chunk_count,
                                    total_chunks=len(chunks),
                                    details=details,
                                ),
                            )
                        )
                    self.documents.append(chunk_text)
                    self.doc_meta.append(
                        self._build_chunk_meta(
                            doc_id=doc_id,
                            filename=fname,
                            file_path=fp,
                            file_id=file_id,
                            chunk_id=chunk_count,
                            total_chunks=len(chunks),
                            details=details,
                        )
                    )
                    chunk_count += 1
                self.file_infos[fp].status = FileStatus.SUCCESS
                self.file_infos[fp].chunks = chunk_count
                if meta:
                    cache_key = self._get_cache_entry_key(folder, fp)
                    new_cache_info[cache_key] = {'size': meta['size'], 'mtime': meta['mtime'], 'chunks': chunk_count}
            except Exception as e:
                failed.append(f"{fname} ({e})")
                self.file_infos[fp].status = FileStatus.FAILED
                self.file_infos[fp].error = str(e)

        if not new_docs and not self.vector_store and not self.documents:
            return TaskResult(False, "처리 가능한 문서 없음", failed_items=failed)
        if self._cancel_event.is_set():
            return self._cancelled_result(progress_cb)
            
        # 벡터 인덱스 생성 (AI 모델이 있을 때만)
        if self.embedding_model and FAISS:
            if progress_cb: progress_cb(75, "벡터 인덱스 생성...")
            try:
                if new_docs:
                    if self.vector_store:
                        batch_size = 100
                        for i in range(0, len(new_docs), batch_size):
                            self.vector_store.add_documents(new_docs[i:i + batch_size])
                    else:
                        self.vector_store = FAISS.from_documents(new_docs, self.embedding_model)
            except Exception as e:
                logger.warning(f"벡터 인덱스 생성 실패 (BM25만 사용): {e}")
        else:
            if progress_cb: progress_cb(75, "BM25 전용 모드...")
        if self._cancel_event.is_set():
            return self._cancelled_result(progress_cb)
             
        if progress_cb: progress_cb(85, "키워드 인덱스 생성...")
        self._build_bm25()
        if self._cancel_event.is_set():
            return self._cancelled_result(progress_cb)
        
        if progress_cb: progress_cb(90, "캐시 저장...")
        self._save_cache(cache_dir, cache_info, new_cache_info)
        
        gc.collect()
        if progress_cb: progress_cb(100, "완료!")
        
        return TaskResult(
            True, f"{len(files) - len(failed)}개 처리 완료",
            {'chunks': len(self.documents), 'new': len(to_process) - len(failed), 'cached': len(cached)},
            failed
        )

    def _build_bm25(self):
        if self.documents:
            self.bm25 = BM25Light()
            self.bm25.fit(self.documents)
        else:
            if self.bm25:
                self.bm25.clear()
            self.bm25 = None

    def _load_cache_info(self, cache_dir: str) -> Dict:
        """캐시 정보 로드 및 설정 검증 (불일치 시 빈 딕셔너리 반환)"""
        path = os.path.join(cache_dir, "cache_info.json")
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    cache_info = json.load(f)
                
                # 캐시 메타데이터 검증
                cache_meta = cache_info.get('_cache_meta', {})
                if cache_meta:
                    mismatches = []
                    if cache_meta.get('cache_key_mode') != self._CACHE_KEY_MODE:
                        mismatches.append(
                            f"cache_key_mode: {cache_meta.get('cache_key_mode')} -> {self._CACHE_KEY_MODE}"
                        )
                    
                    # 모델 ID 검증
                    expected_model_id = self.model_id or "bm25-only"
                    if cache_meta.get('model_id') != expected_model_id:
                        mismatches.append(f"model_id: {cache_meta.get('model_id')} -> {expected_model_id}")
                    
                    # 백엔드 검증
                    current_backend = getattr(AppConfig, 'EMBED_BACKEND', 'torch')
                    if cache_meta.get('embed_backend') != current_backend:
                        mismatches.append(f"embed_backend: {cache_meta.get('embed_backend')} -> {current_backend}")
                    
                    # 정규화 설정 검증
                    current_normalize = getattr(AppConfig, 'EMBED_NORMALIZE', True)
                    if cache_meta.get('embed_normalize') != current_normalize:
                        mismatches.append(f"embed_normalize: {cache_meta.get('embed_normalize')} -> {current_normalize}")
                    
                    # 청킹 설정 검증
                    if cache_meta.get('chunk_size') != AppConfig.CHUNK_SIZE:
                        mismatches.append(f"chunk_size: {cache_meta.get('chunk_size')} -> {AppConfig.CHUNK_SIZE}")
                    if cache_meta.get('chunk_overlap') != AppConfig.CHUNK_OVERLAP:
                        mismatches.append(f"chunk_overlap: {cache_meta.get('chunk_overlap')} -> {AppConfig.CHUNK_OVERLAP}")
                    
                    # 가중치 검증
                    if cache_meta.get('vector_weight') != AppConfig.VECTOR_WEIGHT:
                        mismatches.append(f"vector_weight: {cache_meta.get('vector_weight')} -> {AppConfig.VECTOR_WEIGHT}")
                    if cache_meta.get('bm25_weight') != AppConfig.BM25_WEIGHT:
                        mismatches.append(f"bm25_weight: {cache_meta.get('bm25_weight')} -> {AppConfig.BM25_WEIGHT}")
                    
                    if mismatches:
                        logger.warning(f"⚠️ 캐시 무효화 - 설정 변경 감지: {', '.join(mismatches)}")
                        # 캐시 디렉토리 삭제
                        try:
                            shutil.rmtree(cache_dir)
                            logger.info(f"캐시 디렉토리 삭제됨: {cache_dir}")
                        except Exception as e:
                            logger.warning(f"캐시 디렉토리 삭제 실패: {e}")
                        return {}
                
                # _cache_meta 제외하고 반환
                return {k: v for k, v in cache_info.items() if k != '_cache_meta'}
                
            except json.JSONDecodeError as e:
                logger.debug(f"캐시 정보 JSON 파싱 실패: {path} - {e}")
            except IOError as e:
                logger.debug(f"캐시 정보 파일 읽기 실패: {path} - {e}")
            except Exception as e:
                logger.debug(f"캐시 정보 로드 중 예상치 못한 오류: {path} - {e}")
        return {}
    
    def _save_cache(self, cache_dir: str, old_info: Dict, new_info: Dict):
        try:
            os.makedirs(cache_dir, exist_ok=True)
            if self.vector_store and hasattr(self.vector_store, "save_local"):
                self.vector_store.save_local(cache_dir)
            
            # 캐시 메타데이터 포함
            cache_meta = {
                'model_id': self.model_id or "bm25-only",
                'cache_key_mode': self._CACHE_KEY_MODE,
                'embed_backend': getattr(AppConfig, 'EMBED_BACKEND', 'torch'),
                'embed_normalize': getattr(AppConfig, 'EMBED_NORMALIZE', True),
                'chunk_size': AppConfig.CHUNK_SIZE,
                'chunk_overlap': AppConfig.CHUNK_OVERLAP,
                'vector_weight': AppConfig.VECTOR_WEIGHT,
                'bm25_weight': AppConfig.BM25_WEIGHT
            }
            
            cache_info = {**old_info, **new_info, '_cache_meta': cache_meta}
            
            with open(os.path.join(cache_dir, "cache_info.json"), 'w', encoding='utf-8') as f:
                json.dump(cache_info, f, ensure_ascii=False)
            with open(os.path.join(cache_dir, "docs.json"), 'w', encoding='utf-8') as f:
                json.dump({'docs': self.documents, 'meta': self.doc_meta}, f, ensure_ascii=False)
                
            logger.debug(f"캐시 저장 완료: {cache_dir} (meta: {cache_meta})")
        except Exception as e:
            logger.warning(f"캐시 저장 실패: {e}")

    def initialize(self, folder_path: str, force_reindex: bool = False) -> TaskResult:
        # AI 모델 로드 시도 (실패해도 BM25로 계속 진행)
        if not self._is_ready and not self.embedding_model:
            try:
                res = self.load_model(AppConfig.DEFAULT_MODEL)
                if not res.success:
                    logger.warning(f"AI 모델 로드 실패, BM25 모드로 진행: {res.message}")
            except Exception as e:
                logger.warning(f"AI 모델 로드 오류, BM25 모드로 진행: {e}")

        self.current_folder = folder_path
        self._cancel_event.clear()
        self._cancel_reason = ""
        
        stats_path = os.path.join(get_app_directory(), 'config', 'stats.json')
        if os.path.exists(stats_path):
             try:
                 with open(stats_path, 'r', encoding='utf-8') as f:
                     items = json.load(f)
                     for item in items:
                         # from_dict 사용 - 누락된 필드와 status 변환 자동 처리
                         fi = FileInfo.from_dict(item)
                         self.file_infos[fi.path] = fi
             except json.JSONDecodeError as e:
                 logger.debug(f"통계 파일 JSON 파싱 실패: {stats_path} - {e}")
             except IOError as e:
                 logger.debug(f"통계 파일 읽기 실패: {stats_path} - {e}")
             except Exception as e:
                 logger.warning(f"통계 파일 로드 중 오류: {stats_path} - {e}")

        files = []
        for root, _, filenames in os.walk(folder_path):
            for f in filenames:
                if FileUtils.allowed_file(f):
                    files.append(os.path.join(root, f))
        
        def bg_process():
            self._is_loading = True
            self._load_error = ""
            try:
                def cb(p, msg): self._load_progress = f"{p}% {msg}"
                res = self.process_documents(folder_path, files, cb, force_reindex=force_reindex)
                if not res.success: self._load_error = res.message
                
                try:
                    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
                    data = [info.to_dict() for info in self.file_infos.values()]
                    with open(stats_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False)
                except IOError as e:
                    logger.debug(f"통계 파일 저장 실패: {stats_path} - {e}")
                except Exception as e:
                    logger.warning(f"통계 저장 중 오류: {e}")
            except Exception as e:
                logger.error(f"초기화 오류: {e}")
                self._load_error = str(e)
            finally:
                self._is_loading = False
                if self._cancel_event.is_set():
                    self._load_progress = "중단됨"
                else:
                    self._load_progress = "완료" if not self._load_error else f"오류: {self._load_error}"
                self._cancel_event.clear()
                self._cancel_reason = ""
        
        self._executor.submit(bg_process)
        return TaskResult(True, "초기화 시작됨 (백그라운드 처리)")

    def search(
        self,
        query: str,
        k: int = 5,
        hybrid: bool = True,
        sort_by: str = 'relevance',
        filter_file: str | None = None,
        filter_file_id: str | None = None
    ) -> TaskResult:
        """검색 수행 (성능 모니터링 포함)"""
        start_time = time.perf_counter()  # time 모듈은 이미 모듈 상단에서 import됨
        
        # BM25 전용 모드 지원: vector_store 또는 bm25 중 하나라도 있으면 검색 가능
        if not self.vector_store and not self.bm25:
            return TaskResult(False, "문서가 로드되지 않음")
        
        query = query.strip()
        if len(query) < 2: return TaskResult(False, "검색어가 너무 짧습니다 (최소 2자)")
        
        cached_result = self._search_cache.get(query, k, hybrid, sort_by, filter_file, filter_file_id)
        if cached_result is not None: return TaskResult(True, "검색 완료 (캐시)", cached_result)
        
        try:
            k = max(1, min(k, AppConfig.MAX_SEARCH_RESULTS))
            
            results = {}
            
            # 병렬 검색 활성화 여부 (설정에서 제어 가능)
            parallel_search = getattr(AppConfig, 'PARALLEL_SEARCH', True)
            
            # 하이브리드 검색 시 Vector와 BM25를 병렬로 실행 (성능 최적화 v2.6.1)
            if hybrid and self.vector_store and self.bm25 and parallel_search:
                vector_store = self.vector_store
                bm25 = self.bm25
                
                fetch_k = k * 2
                
                def vector_search():
                    """Vector 검색 수행"""
                    try:
                        return vector_store.similarity_search_with_score(query, k=fetch_k)
                    except Exception as e:
                        logger.debug(f"Vector 검색 오류: {e}")
                        return []
                
                def bm25_search():
                    """BM25 검색 수행"""
                    try:
                        return bm25.search(query, top_k=fetch_k)
                    except Exception as e:
                        logger.debug(f"BM25 검색 오류: {e}")
                        return []
                
                # 병렬 실행 (reused executor)
                vec_future = self._search_executor.submit(vector_search)
                bm25_future = self._search_executor.submit(bm25_search)
                vec_results = vec_future.result(timeout=30)
                bm_res = bm25_future.result(timeout=30)
                
                # Vector 결과 처리
                if vec_results:
                    distances = [r[1] for r in vec_results]
                    min_d = min(distances)
                    max_d = max(distances)
                    rng = max_d - min_d if max_d != min_d else 1
                    
                    for doc, dist in vec_results:
                        meta = doc.metadata or {}
                        doc_id = meta.get('doc_id')
                        key = doc_id if isinstance(doc_id, int) else doc.page_content[:100]
                        score = max(0.1, 1 - ((dist - min_d) / (rng + 0.001)))
                        results[key] = {
                            'content': self.documents[doc_id] if isinstance(doc_id, int) and 0 <= doc_id < len(self.documents) else doc.page_content,
                            'source': meta.get('source', '?'),
                            'path': meta.get('path', ''),
                            'file_id': meta.get('file_id') or (FileUtils.make_file_id(meta.get('path', '')) if meta.get('path') else ''),
                            'vec_score': score,
                            'bm25_score': 0
                        }
                
                # BM25 결과 처리
                if bm_res:
                    bm_scores = [r[1] for r in bm_res]
                    max_bm = max(bm_scores) if bm_scores else 1
                    for idx, sc in bm_res:
                        if 0 <= idx < len(self.documents):
                            key = idx
                            norm = sc / (max_bm + 0.001)
                            if key in results: 
                                results[key]['bm25_score'] = norm
                            else:
                                meta = self.doc_meta[idx] if idx < len(self.doc_meta) else {}
                                results[key] = {
                                    'content': self.documents[idx],
                                    'source': meta.get('source', '?'),
                                    'path': meta.get('path', ''),
                                    'file_id': meta.get('file_id') or (FileUtils.make_file_id(meta.get('path', '')) if meta.get('path') else ''),
                                    'vec_score': 0,
                                    'bm25_score': norm
                                }
            else:
                # 순차 검색 (기존 로직 - 단일 검색 또는 병렬 비활성화 시)
                # Vector 검색 (vector_store가 있는 경우에만)
                if self.vector_store:
                    # 하이브리드 검색 시에만 k*2 사용, 단독 검색 시 k 사용 (성능 최적화)
                    fetch_k = k * 2 if hybrid else k
                    vec_results = self.vector_store.similarity_search_with_score(query, k=fetch_k)
                    
                    if vec_results:
                        distances = [r[1] for r in vec_results]
                        min_d = min(distances)
                        max_d = max(distances)
                        rng = max_d - min_d if max_d != min_d else 1
                        
                        for doc, dist in vec_results:
                            meta = doc.metadata or {}
                            doc_id = meta.get('doc_id')
                            key = doc_id if isinstance(doc_id, int) else doc.page_content[:100]
                            score = max(0.1, 1 - ((dist - min_d) / (rng + 0.001)))
                            results[key] = {
                                'content': self.documents[doc_id] if isinstance(doc_id, int) and 0 <= doc_id < len(self.documents) else doc.page_content,
                                'source': meta.get('source', '?'),
                                'path': meta.get('path', ''),
                                'file_id': meta.get('file_id') or (FileUtils.make_file_id(meta.get('path', '')) if meta.get('path') else ''),
                                'vec_score': score,
                                'bm25_score': 0
                            }
                else:
                    vec_results = []
                        
                if hybrid and self.bm25:
                    try:
                        bm_res = self.bm25.search(query, top_k=k*2)
                    except Exception as e:
                        logger.debug(f"BM25 검색 중 오류 (무시됨): {e}")
                        bm_res = []
                    if bm_res:
                        bm_scores = [r[1] for r in bm_res]
                        max_bm = max(bm_scores) if bm_scores else 1
                        for idx, sc in bm_res:
                            if 0 <= idx < len(self.documents):
                                key = idx
                                norm = sc / (max_bm + 0.001)
                                if key in results: results[key]['bm25_score'] = norm
                                else:
                                    meta = self.doc_meta[idx] if idx < len(self.doc_meta) else {}
                                    results[key] = {
                                        'content': self.documents[idx],
                                        'source': meta.get('source', '?'),
                                        'path': meta.get('path', ''),
                                        'file_id': meta.get('file_id') or (FileUtils.make_file_id(meta.get('path', '')) if meta.get('path') else ''),
                                        'vec_score': 0,
                                        'bm25_score': norm
                                    }
                                
            if filter_file_id:
                results = {k: v for k, v in results.items() if v.get('file_id') == filter_file_id}
            if filter_file:
                results = {k: v for k, v in results.items() if v['source'] == filter_file}
                
            for item in results.values():
                item['score'] = (AppConfig.VECTOR_WEIGHT * item['vec_score'] + AppConfig.BM25_WEIGHT * item['bm25_score'])
                
            if sort_by == 'filename':
                sorted_res = sorted(results.values(), key=lambda x: x['source'])[:k]
            elif sort_by == 'length':
                sorted_res = sorted(results.values(), key=lambda x: len(x['content']), reverse=True)[:k]
            else:
                sorted_res = sorted(results.values(), key=lambda x: x['score'], reverse=True)[:k]
                
            self._search_cache.set(query, k, hybrid, sorted_res, sort_by, filter_file, filter_file_id)
            
            # 대용량 문서 컬렉션 시 메모리 모니터링
            if len(self.documents) > 5000:
                from app.utils import MemoryMonitor
                warning = MemoryMonitor.check_memory_warning(threshold_mb=512)
                if warning:
                    logger.warning(f"검색 후 메모리 경고: {warning}")
            
            # 성능 모니터링: 느린 쿼리 경고
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if elapsed_ms > 1000.0:
                logger.warning(
                    "search_slow query_len=%d results=%d elapsed_ms=%.1f",
                    len(query), len(sorted_res), elapsed_ms
                )
            elif elapsed_ms > 500.0:
                logger.info(
                    "search_done query_len=%d results=%d elapsed_ms=%.1f",
                    len(query), len(sorted_res), elapsed_ms
                )
                
            return TaskResult(True, "검색 완료", sorted_res)
            
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"검색 오류 ({elapsed_ms:.1f}ms): {e}")
            return TaskResult(False, f"검색 오류: {e}")
            
    def get_file_infos(self) -> List[Dict]:
        return [info.to_dict() for info in self.file_infos.values()]

    def get_stats(self) -> Dict:
        total_size = sum(info.size for info in self.file_infos.values())
        total_chunks = sum(info.chunks for info in self.file_infos.values())
        return {
            'files': len(self.file_infos),
            'chunks': total_chunks,
            'size': total_size,
            'size_formatted': FileUtils.format_size(total_size),
            'folder': self.current_folder
        }
    
    def cleanup(self):
        """리소스 정리 및 메모리 해제
        
        ThreadPoolExecutor를 안전하게 종료하고 모든 내부 상태를 초기화합니다.
        """
        logger.info("QA System 정리 시작...")
        
        # 문서 데이터 초기화
        self.documents.clear()
        self.doc_meta.clear()
        self.file_infos.clear()
        self.file_details.clear()
        
        # BM25 인덱스 정리
        if self.bm25:
            self.bm25.clear()
            self.bm25 = None
        
        # 벡터 스토어 정리
        self.vector_store = None
        
        # 캐시 정리
        if hasattr(self, '_search_cache'):
            self._search_cache.clear()
        
        # ThreadPoolExecutor 안전 종료
        if hasattr(self, '_executor') and self._executor:
            try:
                # cancel_futures=True로 대기 중인 작업 취소 (Python 3.9+)
                self._executor.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                # Python 3.8 이하 호환
                self._executor.shutdown(wait=False)
            except Exception as e:
                logger.warning(f"ThreadPoolExecutor 종료 중 오류: {e}")

        if hasattr(self, '_search_executor') and self._search_executor:
            try:
                self._search_executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                self._search_executor.shutdown(wait=False)
            except Exception as e:
                logger.warning(f"Search executor 종료 중 오류: {e}")
        
        # 상태 초기화
        self._is_ready = False
        self._is_loading = False
        self._load_progress = ""
        self._load_error = ""
        self._cancel_event.clear()
        self._cancel_reason = ""
        
        # 가비지 컬렉션
        gc.collect()
        logger.info("QA System 정리 완료")

rate_limiter = RateLimiter(AppConfig.RATE_LIMIT_PER_MINUTE)
search_queue = SearchQueue(AppConfig.MAX_CONCURRENT_SEARCHES)
qa_system = RegulationQASystem()
