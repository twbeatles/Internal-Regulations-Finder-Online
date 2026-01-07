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
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# ============================================================================
# Lazy Import 패턴 - 무거운 라이브러리는 실제 사용 시점에 로드
# GUI 시작 시간 단축을 위해 PyTorch, LangChain 등은 지연 로딩
# ============================================================================

# 이 변수들은 실제 사용 시점에 _lazy_import_*() 함수로 로드됨
CharacterTextSplitter = None
Document = None
HuggingFaceEmbeddings = None
FAISS = None
_lazy_imports_loaded = False

def _lazy_import_langchain():
    """LangChain 관련 라이브러리 지연 로드"""
    global CharacterTextSplitter, Document, HuggingFaceEmbeddings, FAISS, _lazy_imports_loaded
    
    if _lazy_imports_loaded:
        return
    
    # LangChain 호환성 임포트 (최신 버전 우선, 구버전 폴백)
    try:
        from langchain_text_splitters import CharacterTextSplitter as _CharacterTextSplitter
        CharacterTextSplitter = _CharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import CharacterTextSplitter as _CharacterTextSplitter
            CharacterTextSplitter = _CharacterTextSplitter
        except ImportError:
            pass

    try:
        from langchain_core.documents import Document as _Document
        Document = _Document
    except ImportError:
        try:
            from langchain.docstore.document import Document as _Document
            Document = _Document
        except ImportError:
            pass

    # HuggingFaceEmbeddings - langchain-huggingface 패키지 우선 사용
    try:
        from langchain_huggingface import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
        HuggingFaceEmbeddings = _HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
            HuggingFaceEmbeddings = _HuggingFaceEmbeddings
        except ImportError:
            try:
                from langchain.embeddings import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
                HuggingFaceEmbeddings = _HuggingFaceEmbeddings
            except ImportError:
                pass

    # FAISS 벡터스토어
    try:
        from langchain_community.vectorstores import FAISS as _FAISS
        FAISS = _FAISS
    except ImportError:
        try:
            from langchain.vectorstores import FAISS as _FAISS
            FAISS = _FAISS
        except ImportError:
            pass
    
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
    __slots__ = ['k1', 'b', 'corpus', 'doc_lens', 'avgdl', 'idf', 'N', '_lock', 'doc_tfs']
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[List[str]] = []
        self.doc_lens: List[int] = []
        self.doc_tfs: List[Counter] = []  # 사전 계산된 term frequency
        self.avgdl = 0.0
        self.idf: Dict[str, float] = {}
        self.N = 0
        self._lock = threading.RLock()
    
    def _tokenize(self, text: str) -> List[str]:
        """사전 컴파일된 정규식을 사용한 토큰화"""
        if not text:
            return []
        # 사전 컴파일된 패턴 사용
        text = _TOKENIZE_PATTERN.sub(' ', text.lower())
        return [t for t in text.split() if len(t) >= 2]
    
    def fit(self, docs: List[str]):
        """
        문서 인덱싱 - term frequency를 미리 계산하여 검색 성능 향상
        """
        with self._lock:
            self.corpus = []
            self.doc_lens = []
            self.doc_tfs = []  # term frequency 사전 계산
            df = Counter()
            
            for doc in docs:
                tokens = self._tokenize(doc)
                self.corpus.append(tokens)
                self.doc_lens.append(len(tokens))
                # term frequency 미리 계산 (검색 시 재계산 방지)
                tf = Counter(tokens)
                self.doc_tfs.append(tf)
                df.update(set(tokens))
            
            self.N = len(docs)
            self.avgdl = sum(self.doc_lens) / self.N if self.N else 0
            self.idf = {t: math.log((self.N - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}
            del df
            gc.collect()
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """검색 수행 - 사전 계산된 term frequency 활용"""
        with self._lock:
            if not self.corpus or not query:
                return []
            q_tokens = self._tokenize(query)
            if not q_tokens:
                return []
            
            # 쿼리 토큰의 IDF 값 미리 조회 (반복 조회 최소화)
            query_idf = {term: self.idf.get(term, 0) for term in q_tokens if term in self.idf}
            if not query_idf:
                return []
            
            scores = []
            for idx, doc_tf in enumerate(self.doc_tfs):
                if not doc_tf:
                    continue
                score = self._score_optimized(query_idf, doc_tf, self.doc_lens[idx])
                if score > 0:
                    scores.append((idx, score))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]
    
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
            self.corpus.clear()
            self.doc_lens.clear()
            self.doc_tfs.clear()
            self.idf.clear()
            gc.collect()

# ============================================================================
# 검색 캐시, Rate Limiter, Search Queue
# ============================================================================
class SearchCache:
    """LRU 기반 검색 캐시 (OrderedDict 사용으로 O(1) 성능)
    
    Features:
        - TTL 기반 만료
        - LRU eviction
        - 캐시 히트율 통계
        - 메모리 사용량 추정
    """
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        from collections import OrderedDict
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
        # 통계
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def _make_key(self, query: str, k: int, hybrid: bool) -> str:
        return f"{query}|{k}|{hybrid}"
    
    def get(self, query: str, k: int, hybrid: bool) -> Optional[Any]:
        key = self._make_key(query, k, hybrid)
        with self._lock:
            if key in self.cache:
                timestamp, result = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    # LRU: 최근 사용으로 이동
                    self.cache.move_to_end(key)
                    self._hits += 1
                    return result
                # 만료된 항목 제거
                del self.cache[key]
                self._evictions += 1
            self._misses += 1
        return None
    
    def set(self, query: str, k: int, hybrid: bool, result: Any):
        key = self._make_key(query, k, hybrid)
        with self._lock:
            # 이미 존재하면 삭제 후 다시 추가 (순서 갱신)
            if key in self.cache:
                del self.cache[key]
            # 크기 초과 시 가장 오래된 항목 제거 (O(1))
            while len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
                self._evictions += 1
            self.cache[key] = (time.time(), result)
    
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
                'ttl_seconds': self.ttl
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
            for key, (_, result) in self.cache.items():
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
        query = query.strip()
        if len(query) < 2: return
        try:
            db.execute("INSERT INTO search_history (query) VALUES (?)", (query,))
        except Exception: pass
            
    def get_recent(self, limit: int = 10) -> List[str]:
        rows = db.fetchall("SELECT query, MAX(timestamp) as ts FROM search_history GROUP BY query ORDER BY ts DESC LIMIT ?", (limit,))
        return [r['query'] for r in rows]
            
    def get_popular(self, limit: int = 10) -> List[Tuple[str, int]]:
        rows = db.fetchall("SELECT query, COUNT(*) as cnt FROM search_history GROUP BY query ORDER BY cnt DESC LIMIT ?", (limit,))
        return [(r['query'], r['cnt']) for r in rows]
            
    def suggest(self, prefix: str, limit: int = 5) -> List[str]:
        rows = db.fetchall("SELECT DISTINCT query FROM search_history WHERE query LIKE ? ORDER BY timestamp DESC LIMIT ?", (f"{prefix}%", limit))
        return [r['query'] for r in rows]
            
    def clear(self):
        db.execute("DELETE FROM search_history")

# ============================================================================
# RegulationQASystem
# ============================================================================
class RegulationQASystem:
    def __init__(self):
        self.vector_store = None
        self.embedding_model = None
        self.model_id = None
        self.model_name = ""
        self.extractor = DocumentExtractor()
        self.cache_path = os.path.join(os.path.dirname(get_app_directory()), "reg_qa_server_v10")
        if not os.path.exists(self.cache_path):
             # Try temp dir if app dir not writable or structure different
             self.cache_path = os.path.join(os.environ.get('TEMP', '/tmp'), "reg_qa_server_v10")
             
        self.bm25 = None
        self.documents: List[str] = []
        self.doc_meta: List[Dict] = []
        self.file_infos: Dict[str, FileInfo] = {}
        self.current_folder = ""
        self._lock = threading.RLock()
        self._search_cache = SearchCache(AppConfig.SEARCH_CACHE_SIZE)
        self._search_history = SearchHistory()
        self._keyword_cache: List[str] = []
        self._executor = ThreadPoolExecutor(max_workers=AppConfig.MAX_WORKERS)
        self._is_ready = False
        self._is_loading = False
        self._load_progress = ""
        self._load_error = ""
        
        # v2.0 Components
        self.tag_manager = TagManager()
        self.revision_tracker = RevisionTracker()
        self.article_parser = ArticleParser()
        self.doc_splitter = DocumentSplitter()
    
    def get_keywords(self, limit: int = 50) -> List[str]:
        if not self._keyword_cache and self.documents:
            self._keyword_cache = TextHighlighter.extract_keywords(self.documents, limit)
        return self._keyword_cache[:limit]
    
    @property
    def is_ready(self) -> bool:
        return self._is_ready and self.embedding_model is not None
    
    @property
    def is_loading(self) -> bool:
        return self._is_loading
    
    @property
    def load_progress(self) -> str:
        return self._load_progress
    
    @property
    def load_error(self) -> str:
        return self._load_error
    
    def load_model(self, model_name: str, offline_mode: bool = None, local_model_path: str = None) -> TaskResult:
        if self._is_loading:
            return TaskResult(False, "이미 모델을 로딩 중입니다")
        
        is_offline = offline_mode if offline_mode is not None else AppConfig.OFFLINE_MODE
        model_path_override = local_model_path if local_model_path is not None else AppConfig.LOCAL_MODEL_PATH
        model_id = AppConfig.AVAILABLE_MODELS.get(model_name, AppConfig.AVAILABLE_MODELS[AppConfig.DEFAULT_MODEL])
        self._load_error = ""
        
        try:
            self._is_loading = True
            self._load_progress = "라이브러리 로드 중..."
            logger.info("라이브러리 로드 중...")
            
            # Lazy import 실행 (무거운 라이브러리 로드)
            _lazy_import_langchain()
            
            try:
                import torch
            except ImportError as e:
                self._load_error = str(e)
                raise ModelLoadError(model_name, f"PyTorch 로드 실패: {e}")
            
            # 전역 변수로 이미 로드됨 (lazy import)
            if HuggingFaceEmbeddings is None:
                self._load_error = "LangChain HuggingFaceEmbeddings 로드 실패"
                raise ModelLoadError(model_name, "LangChain HuggingFaceEmbeddings 없음")
            
            self._load_progress = "AI 모델 초기화 중..."
            logger.info(f"모델 로드 시작: {model_name} ({model_id})")
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"사용 디바이스: {device}")
            
            # ====================================================================
            # 모델 저장 경로 설정 (프로젝트 폴더 우선)
            # ====================================================================
            # 프로젝트 내 models 폴더를 기본 캐시 위치로 사용
            project_models_dir = os.path.join(get_app_directory(), 'models')
            os.makedirs(project_models_dir, exist_ok=True)
            
            # HuggingFace 캐시 디렉토리를 프로젝트 폴더로 설정
            os.environ['HF_HOME'] = project_models_dir
            os.environ['HUGGINGFACE_HUB_CACHE'] = project_models_dir
            os.environ['TRANSFORMERS_CACHE'] = project_models_dir
            
            logger.info(f"모델 저장 경로: {project_models_dir}")
            
            load_path = model_id
            
            # 오프라인 모드 설정 (폐쇄망 지원)
            if is_offline:
                # HuggingFace Hub 오프라인 모드 환경 변수 설정
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                
                # 로컬 모델 경로 우선 사용
                local_path = model_path_override or os.path.join(project_models_dir, model_id.replace('/', '--'))
                
                if os.path.exists(local_path):
                    load_path = local_path
                    logger.info(f"로컬 모델 경로 사용: {load_path}")
                else:
                    self._load_error = "오프라인 모드에서 로컬 모델을 찾을 수 없습니다"
                    logger.error(f"오프라인 모드 오류: 모델 경로 없음 - {local_path}")
                    raise ModelOfflineError(local_path)
            
            self._load_progress = "모델 다운로드 중... (최초 실행 시 시간이 걸릴 수 있습니다)"
            
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=load_path,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True},
                cache_folder=project_models_dir  # 프로젝트 폴더에 캐시
            )
            
            self.model_id = model_id
            self.model_name = model_name
            self._load_progress = "모델 로드 완료"
            self._is_ready = True
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
        
        Raises:
            ModelNotLoadedError: 모델이 로드되지 않은 경우
        """
        if not self.model_id:
            raise ModelNotLoadedError()
        h1 = hashlib.md5(self.model_id.encode()).hexdigest()[:6]
        h2 = hashlib.md5(folder.encode()).hexdigest()[:6]
        return os.path.join(self.cache_path, f"{h2}_{h1}")
    
    def process_single_file(self, file_path: str) -> TaskResult:
        """단일 파일 업로드 후 즉시 인덱싱
        
        Args:
            file_path: 처리할 파일의 전체 경로
            
        Returns:
            TaskResult: 처리 결과
        """
        if not self.embedding_model:
            return TaskResult(False, "모델이 로드되지 않았습니다")
        
        if not os.path.exists(file_path):
            return TaskResult(False, f"파일을 찾을 수 없습니다: {file_path}")
        
        try:
            logger.info(f"단일 파일 처리 시작: {file_path}")
            
            # 파일 추출
            text, error = self.extractor.extract(file_path)
            if error:
                return TaskResult(False, f"파일 추출 오류: {error}")
            if not text or not text.strip():
                return TaskResult(False, "파일에서 텍스트를 추출할 수 없습니다")
            
            # 문서 분할
            splitter = DocumentSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split(text)
            
            if not chunks:
                return TaskResult(False, "문서 분할 결과가 없습니다")
            
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            
            # file_infos에 추가
            self.file_infos[file_path] = FileInfo(
                path=file_path,
                size=file_size,
                status=FileStatus.READY,
                chunks=len(chunks),
                mod_time=os.path.getmtime(file_path)
            )
            
            # 문서 및 메타데이터 추가
            for i, chunk in enumerate(chunks):
                self.documents.append(chunk)
                self.doc_meta.append({
                    'source': file_path,
                    'filename': filename,
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                })
            
            # 벡터스토어 업데이트 (있으면 추가, 없으면 생성)
            if self.embedding_model and FAISS:
                try:
                    docs = [Document(page_content=chunk, metadata={'source': file_path, 'filename': filename}) 
                            for chunk in chunks]
                    
                    if self.vector_store:
                        # 기존 벡터스토어에 추가
                        self.vector_store.add_documents(docs)
                    else:
                        # 새로 생성
                        self.vector_store = FAISS.from_documents(docs, self.embedding_model)
                    
                    logger.info(f"벡터스토어 업데이트 완료: {len(chunks)} 청크")
                except Exception as e:
                    logger.error(f"벡터스토어 업데이트 오류: {e}")
            
            # BM25 재구축
            self._build_bm25()
            
            logger.info(f"✅ 단일 파일 처리 완료: {filename} ({len(chunks)} 청크)")
            return TaskResult(True, f"파일 처리 완료: {filename} ({len(chunks)} 청크)", {
                'filename': filename,
                'chunks': len(chunks),
                'size': file_size
            })
            
        except Exception as e:
            logger.error(f"단일 파일 처리 오류: {e}")
            import traceback
            traceback.print_exc()
            return TaskResult(False, f"파일 처리 오류: {str(e)}")
    
    def process_documents(self, folder: str, files: List[str], progress_cb=None) -> TaskResult:
        if not self.embedding_model:
            return TaskResult(False, "모델이 로드되지 않았습니다")
        
        with self._lock:
            return self._process_internal(folder, files, progress_cb)
    
    def _process_internal(self, folder: str, files: List[str], progress_cb) -> TaskResult:
        # Lazy import 실행 (이미 로드되었으면 스킵)
        _lazy_import_langchain()
        
        self.current_folder = folder
        cache_dir = self._get_cache_dir(folder)
        self.file_infos.clear()
        self._search_cache.clear()
        
        # Init FileInfo
        for fp in files:
            meta = FileUtils.get_metadata(fp)
            self.file_infos[fp] = FileInfo(
                fp, os.path.basename(fp),
                os.path.splitext(fp)[1].lower(),
                meta['size'] if meta else 0
            )
            
        if progress_cb: progress_cb(5, "캐시 확인...")
        
        cache_info = self._load_cache_info(cache_dir)
        to_process, cached = [], []
        
        for fp in files:
            fname = os.path.basename(fp)
            meta = FileUtils.get_metadata(fp)
            if meta and fname in cache_info:
                cm = cache_info[fname]
                if cm.get('size') == meta['size'] and cm.get('mtime') == meta['mtime']:
                    cached.append(fp)
                    self.file_infos[fp].status = FileStatus.CACHED
                    self.file_infos[fp].chunks = cm.get('chunks', 0)
                    continue
            to_process.append(fp)
            
        self.documents, self.doc_meta = [], []
        
        # Load Cache
        if cached and os.path.exists(os.path.join(cache_dir, "index.faiss")):
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
            except Exception as e:
                logger.warning(f"캐시 로드 실패: {e}")
                to_process, cached = files, []
                self.vector_store = None
        
        if not to_process:
            self._build_bm25()
            if progress_cb: progress_cb(100, "완료!")
            return TaskResult(True, f"캐시에서 {len(cached)}개 파일 로드", 
                            {'chunks': len(self.documents), 'cached': len(cached), 'new': 0})
        
        splitter = CharacterTextSplitter(
            separator="\n\n", chunk_size=AppConfig.CHUNK_SIZE, chunk_overlap=AppConfig.CHUNK_OVERLAP
        )
        failed, new_docs, new_cache_info = [], [], {}
        
        # Parallel Extraction
        def extract_file(fp: str) -> Tuple[str, str, Optional[str], Optional[Dict]]:
            fname = os.path.basename(fp)
            try:
                content, error = self.extractor.extract(fp)
                meta = FileUtils.get_metadata(fp)
                return fp, fname, content, error, meta
            except Exception as e:
                return fp, fname, None, str(e), None
        
        extracted_results = []
        if progress_cb: progress_cb(15, f"문서 추출 중... (병렬 처리)")
        
        with ThreadPoolExecutor(max_workers=min(AppConfig.MAX_WORKERS, len(to_process))) as executor:
            futures = {executor.submit(extract_file, fp): fp for fp in to_process}
            completed = 0
            for future in futures:
                try:
                    result = future.result(timeout=60)
                    extracted_results.append(result)
                    completed += 1
                    if progress_cb and completed % 5 == 0:
                        progress = 15 + int((completed / len(to_process)) * 30)
                        progress_cb(progress, f"추출 완료: {completed}/{len(to_process)}")
                except Exception as e:
                    fp = futures[future]
                    fname = os.path.basename(fp)
                    extracted_results.append((fp, fname, None, f"추출 타임아웃: {e}", None))
        
        if progress_cb: progress_cb(50, "텍스트 청킹 중...")
        
        for fp, fname, content, error, meta in extracted_results:
            self.file_infos[fp].status = FileStatus.PROCESSING
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
                chunks = splitter.split_text(content)
                chunk_count = 0
                for chunk in chunks:
                    if chunk.strip():
                        new_docs.append(Document(
                            page_content=chunk.strip(), metadata={"source": fname, "path": fp}
                        ))
                        self.documents.append(chunk.strip())
                        self.doc_meta.append({"source": fname, "path": fp})
                        chunk_count += 1
                self.file_infos[fp].status = FileStatus.SUCCESS
                self.file_infos[fp].chunks = chunk_count
                if meta:
                    new_cache_info[fname] = {'size': meta['size'], 'mtime': meta['mtime'], 'chunks': chunk_count}
            except Exception as e:
                failed.append(f"{fname} ({e})")
                self.file_infos[fp].status = FileStatus.FAILED
                self.file_infos[fp].error = str(e)

        if not new_docs and not self.vector_store:
            return TaskResult(False, "처리 가능한 문서 없음", failed_items=failed)
            
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
             return TaskResult(False, f"인덱스 생성 실패: {e}")
             
        if progress_cb: progress_cb(85, "키워드 인덱스 생성...")
        self._build_bm25()
        
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

    def _load_cache_info(self, cache_dir: str) -> Dict:
        """캐시 정보 로드 (에러 시 빈 딕셔너리 반환)"""
        path = os.path.join(cache_dir, "cache_info.json")
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
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
            self.vector_store.save_local(cache_dir)
            with open(os.path.join(cache_dir, "cache_info.json"), 'w', encoding='utf-8') as f:
                json.dump({**old_info, **new_info}, f, ensure_ascii=False)
            with open(os.path.join(cache_dir, "docs.json"), 'w', encoding='utf-8') as f:
                json.dump({'docs': self.documents, 'meta': self.doc_meta}, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"캐시 저장 실패: {e}")

    def initialize(self, folder_path: str, force_reindex: bool = False) -> TaskResult:
        if not self._is_ready:
             res = self.load_model(AppConfig.DEFAULT_MODEL)
             if not res.success: return res

        self.current_folder = folder_path
        
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
            try:
                def cb(p, msg): self._load_progress = f"{p}% {msg}"
                res = self.process_documents(folder_path, files, cb)
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
                self._load_progress = "완료" if not self._load_error else f"오류: {self._load_error}"
        
        self._executor.submit(bg_process)
        return TaskResult(True, "초기화 시작됨 (백그라운드 처리)")

    def search(self, query: str, k: int = 5, hybrid: bool = True, sort_by: str = 'relevance', filter_file: str = None) -> TaskResult:
        if not self.vector_store: return TaskResult(False, "문서가 로드되지 않음")
        
        query = query.strip()
        if len(query) < 2: return TaskResult(False, "검색어가 너무 짧습니다 (최소 2자)")
        
        cached_result = self._search_cache.get(query, k, hybrid)
        if cached_result is not None: return TaskResult(True, "검색 완료 (캐시)", cached_result)
        
        try:
            k = max(1, min(k, AppConfig.MAX_SEARCH_RESULTS))
            vec_results = self.vector_store.similarity_search_with_score(query, k=k*2)
            
            results = {}
            if vec_results:
                distances = [r[1] for r in vec_results]
                min_d = min(distances)
                max_d = max(distances)
                rng = max_d - min_d if max_d != min_d else 1
                
                for doc, dist in vec_results:
                    key = doc.page_content[:100]
                    score = max(0.1, 1 - ((dist - min_d) / (rng + 0.001)))
                    results[key] = {
                        'content': doc.page_content,
                        'source': doc.metadata.get('source', '?'),
                        'path': doc.metadata.get('path', ''),
                        'vec_score': score,
                        'bm25_score': 0
                    }
                    
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
                            key = self.documents[idx][:100]
                            norm = sc / (max_bm + 0.001)
                            if key in results: results[key]['bm25_score'] = norm
                            else:
                                meta = self.doc_meta[idx] if idx < len(self.doc_meta) else {}
                                results[key] = {
                                    'content': self.documents[idx],
                                    'source': meta.get('source', '?'),
                                    'path': meta.get('path', ''),
                                    'vec_score': 0,
                                    'bm25_score': norm
                                }
                                
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
                
            if not filter_file: self._search_cache.set(query, k, hybrid, sorted_res)
            return TaskResult(True, "검색 완료", sorted_res)
            
        except Exception as e:
            logger.error(f"검색 오류: {e}")
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
        
        # 상태 초기화
        self._is_ready = False
        self._is_loading = False
        self._load_progress = ""
        self._load_error = ""
        
        # 가비지 컬렉션
        gc.collect()
        logger.info("QA System 정리 완료")

rate_limiter = RateLimiter(AppConfig.RATE_LIMIT_PER_MINUTE)
search_queue = SearchQueue(AppConfig.MAX_CONCURRENT_SEARCHES)
qa_system = RegulationQASystem()
