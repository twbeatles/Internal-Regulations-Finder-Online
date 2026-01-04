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
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError:
    HuggingFaceEmbeddings = None
    FAISS = None

from app.config import AppConfig
from app.utils import logger, FileInfo, FileStatus, TaskResult, FileUtils, get_app_directory
from app.services.db import db
from app.services.document import DocumentExtractor, TextHighlighter, DocumentSplitter, ArticleParser
from app.services.metadata import TagManager
from app.services.file_manager import RevisionTracker

# ============================================================================
# BM25 경량 구현
# ============================================================================
class BM25Light:
    __slots__ = ['k1', 'b', 'corpus', 'doc_lens', 'avgdl', 'idf', 'N', '_lock']
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[List[str]] = []
        self.doc_lens: List[int] = []
        self.avgdl = 0.0
        self.idf: Dict[str, float] = {}
        self.N = 0
        self._lock = threading.RLock()
    
    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        text = re.sub(r'[^\w\s가-힣]', ' ', text.lower())
        return [t for t in text.split() if len(t) >= 2]
    
    def fit(self, docs: List[str]):
        with self._lock:
            self.corpus = []
            self.doc_lens = []
            df = Counter()
            for doc in docs:
                tokens = self._tokenize(doc)
                self.corpus.append(tokens)
                self.doc_lens.append(len(tokens))
                df.update(set(tokens))
            self.N = len(docs)
            self.avgdl = sum(self.doc_lens) / self.N if self.N else 0
            self.idf = {t: math.log((self.N - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}
            del df
            gc.collect()
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        with self._lock:
            if not self.corpus or not query:
                return []
            q_tokens = self._tokenize(query)
            if not q_tokens:
                return []
            scores = []
            for idx, doc_tokens in enumerate(self.corpus):
                if not doc_tokens:
                    continue
                score = self._score(q_tokens, doc_tokens, self.doc_lens[idx])
                if score > 0:
                    scores.append((idx, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]
    
    def _score(self, query: List[str], doc: List[str], doc_len: int) -> float:
        score = 0.0
        doc_tf = Counter(doc)
        for term in query:
            if term not in self.idf:
                continue
            tf = doc_tf.get(term, 0)
            idf = self.idf[term]
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * num / den if den > 0 else 0
        return score
    
    def clear(self):
        with self._lock:
            self.corpus.clear()
            self.doc_lens.clear()
            self.idf.clear()
            gc.collect()

# ============================================================================
# 검색 캐시, Rate Limiter, Search Queue
# ============================================================================
class SearchCache:
    """LRU 기반 검색 캐시 (OrderedDict 사용으로 O(1) 성능)"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        from collections import OrderedDict
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
    
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
                    return result
                # 만료된 항목 제거
                del self.cache[key]
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
            self.cache[key] = (time.time(), result)
    
    def clear(self):
        with self._lock:
            self.cache.clear()
    
    def size(self) -> int:
        """현재 캐시 크기"""
        with self._lock:
            return len(self.cache)

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests: Dict[str, List[float]] = {}
        self.limit = requests_per_minute
        self._lock = threading.Lock()
        self._cleanup_interval = 60
        self._last_cleanup = time.time()
    
    def is_allowed(self, ip: str) -> bool:
        current_time = time.time()
        with self._lock:
            if current_time - self._last_cleanup > self._cleanup_interval:
                self._cleanup(current_time)
                self._last_cleanup = current_time
            if ip not in self.requests:
                self.requests[ip] = []
            cutoff = current_time - 60
            self.requests[ip] = [t for t in self.requests[ip] if t > cutoff]
            if len(self.requests[ip]) >= self.limit:
                logger.warning(f"Rate limit exceeded for IP: {ip}")
                return False
            self.requests[ip].append(current_time)
            return True
    
    def _cleanup(self, current_time: float):
        cutoff = current_time - 60
        expired_ips = [ip for ip, ts in self.requests.items() if not [t for t in ts if t > cutoff]]
        for ip in expired_ips:
            del self.requests[ip]

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
            
            try:
                import torch
            except ImportError as e:
                self._load_error = str(e)
                return TaskResult(False, f"PyTorch 로드 실패: {e}")
            
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                from langchain_community.vectorstores import FAISS
            except ImportError as e:
                self._load_error = str(e)
                return TaskResult(False, f"LangChain 로드 실패: {e}")
            
            self._load_progress = "AI 모델 초기화 중..."
            logger.info(f"모델 로드 시작: {model_name} ({model_id})")
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"사용 디바이스: {device}")
            
            load_path = model_id
            if is_offline and model_path_override and os.path.exists(model_path_override):
                load_path = model_path_override
                logger.info(f"로컬 모델 경로 사용: {load_path}")
            
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=load_path,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            self.model_id = model_id
            self.model_name = model_name
            self._load_progress = "모델 로드 완료"
            self._is_ready = True
            return TaskResult(True, "모델 로드 완료")
            
        except Exception as e:
            self._load_error = str(e)
            logger.error(f"모델 로드 오류: {e}")
            return TaskResult(False, f"모델 로드 오류: {e}")
        finally:
            self._is_loading = False
            
    def _get_cache_dir(self, folder: str) -> str:
        if not self.model_id:
            raise ValueError("모델이 로드되지 않았습니다")
        h1 = hashlib.md5(self.model_id.encode()).hexdigest()[:6]
        h2 = hashlib.md5(folder.encode()).hexdigest()[:6]
        return os.path.join(self.cache_path, f"{h2}_{h1}")
    
    def process_documents(self, folder: str, files: List[str], progress_cb=None) -> TaskResult:
        if not self.embedding_model:
            return TaskResult(False, "모델이 로드되지 않았습니다")
        
        with self._lock:
            return self._process_internal(folder, files, progress_cb)
    
    def _process_internal(self, folder: str, files: List[str], progress_cb) -> TaskResult:
        # LangChain import check
        try: from langchain_text_splitters import CharacterTextSplitter
        except ImportError: from langchain.text_splitter import CharacterTextSplitter
        
        try: from langchain_core.documents import Document
        except ImportError: from langchain.docstore.document import Document
        
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
        path = os.path.join(cache_dir, "cache_info.json")
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f: return json.load(f)
            except: pass
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
                         fi = FileInfo(**item)
                         if isinstance(fi.status, str):
                             try: fi.status = FileStatus(fi.status)
                             except: fi.status = FileStatus.PENDING
                         self.file_infos[fi.name] = fi
             except Exception: pass

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
                except: pass
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
                try: bm_res = self.bm25.search(query, top_k=k*2)
                except: bm_res = []
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
        self.documents.clear()
        self.doc_meta.clear()
        if self.bm25: self.bm25.clear()
        self._search_cache.clear()
        try:
            self._executor.shutdown(wait=False)
        except: pass
        gc.collect()

rate_limiter = RateLimiter(AppConfig.RATE_LIMIT_PER_MINUTE)
search_queue = SearchQueue(AppConfig.MAX_CONCURRENT_SEARCHES)
qa_system = RegulationQASystem()
