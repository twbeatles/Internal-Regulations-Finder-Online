# -*- coding: utf-8 -*-
"""
사내 규정 검색기 - 웹 서버 버전 v1.0
Flask 기반 웹 서버로 다중 사용자 동시 접속 지원
"""

from __future__ import annotations
import sys
import os
import json
import threading
import tempfile
import hashlib
import shutil
import logging
import subprocess
import platform
import re
import gc
import math
import time
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import Counter
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import signal
import atexit

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ============================================================================
# 상수 및 설정
# ============================================================================
class AppConfig:
    APP_NAME = "사내 규정 검색기"
    APP_VERSION = "1.0 (웹 서버)"
    
    # 서버 설정
    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = 8080
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    
    # AI 모델 설정
    AVAILABLE_MODELS: Dict[str, str] = {
        "SNU SBERT (고성능)": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        "BM-K Simal (균형)": "BM-K/ko-simal-roberta-base",
        "JHGan SBERT (빠름)": "jhgan/ko-sbert-nli"
    }
    DEFAULT_MODEL = "JHGan SBERT (빠름)"
    
    # 파일 설정
    UPLOAD_FOLDER = "uploads"
    SUPPORTED_EXTENSIONS = {'.txt', '.docx', '.pdf'}
    
    # 검색 설정
    MAX_SEARCH_RESULTS = 10
    DEFAULT_SEARCH_RESULTS = 3
    
    # 청킹 설정
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 80
    VECTOR_WEIGHT = 0.7
    BM25_WEIGHT = 0.3
    
    # 동시성 설정
    MAX_WORKERS = 4
    REQUEST_TIMEOUT = 60
    SEARCH_CACHE_SIZE = 100


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


# ============================================================================
# 로깅 설정
# ============================================================================
def get_app_directory() -> str:
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def setup_logger() -> logging.Logger:
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


# ============================================================================
# 유틸리티
# ============================================================================
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


# ============================================================================
# BM25 경량 구현 (스레드 안전)
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
# 문서 추출기
# ============================================================================
class DocumentExtractor:
    def __init__(self):
        self._docx_module = None
        self._pdf_module = None
    
    @property
    def docx(self):
        if self._docx_module is None:
            try:
                from docx import Document
                self._docx_module = Document
            except ImportError:
                self._docx_module = False
        return self._docx_module
    
    @property
    def pdf(self):
        if self._pdf_module is None:
            try:
                from pypdf import PdfReader
                self._pdf_module = PdfReader
            except ImportError:
                self._pdf_module = False
        return self._pdf_module
    
    def extract(self, path: str) -> Tuple[str, Optional[str]]:
        if not path or not os.path.exists(path):
            return "", f"파일 없음: {path}"
        if not os.path.isfile(path):
            return "", f"파일이 아님: {path}"
        ext = os.path.splitext(path)[1].lower()
        if ext == '.txt':
            return self._extract_txt(path)
        elif ext == '.docx':
            return self._extract_docx(path)
        elif ext == '.pdf':
            return self._extract_pdf(path)
        return "", f"지원하지 않는 형식: {ext}"
    
    def _extract_txt(self, path: str) -> Tuple[str, Optional[str]]:
        return FileUtils.safe_read(path)
    
    def _extract_docx(self, path: str) -> Tuple[str, Optional[str]]:
        if not self.docx:
            return "", "DOCX 라이브러리 없음 (pip install python-docx)"
        try:
            doc = self.docx(path)
            parts = []
            for para in doc.paragraphs:
                if para.text.strip():
                    parts.append(para.text.strip())
            for table in doc.tables:
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        parts.append(' | '.join(cells))
            return '\n\n'.join(parts), None
        except Exception as e:
            return "", f"DOCX 오류: {e}"
    
    def _extract_pdf(self, path: str) -> Tuple[str, Optional[str]]:
        if not self.pdf:
            return "", "PDF 라이브러리 없음 (pip install pypdf)"
        try:
            reader = self.pdf(path)
            if reader.is_encrypted:
                try:
                    reader.decrypt('')
                except Exception:
                    return "", "암호화된 PDF"
            texts = []
            for page in reader.pages:
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        texts.append(text.strip())
                except Exception:
                    continue
            if not texts:
                return "", "텍스트 없음 (이미지 PDF)"
            return '\n\n'.join(texts), None
        except Exception as e:
            return "", f"PDF 오류: {e}"


# ============================================================================
# 검색 캐시 (LRU)
# ============================================================================
class SearchCache:
    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, Tuple[float, Any]] = {}
        self.max_size = max_size
        self._lock = threading.Lock()
    
    def _make_key(self, query: str, k: int, hybrid: bool) -> str:
        return f"{query}|{k}|{hybrid}"
    
    def get(self, query: str, k: int, hybrid: bool) -> Optional[Any]:
        key = self._make_key(query, k, hybrid)
        with self._lock:
            if key in self.cache:
                timestamp, result = self.cache[key]
                # 5분 이내 캐시만 유효
                if time.time() - timestamp < 300:
                    return result
                del self.cache[key]
        return None
    
    def set(self, query: str, k: int, hybrid: bool, result: Any):
        key = self._make_key(query, k, hybrid)
        with self._lock:
            if len(self.cache) >= self.max_size:
                # 가장 오래된 항목 제거
                oldest_key = min(self.cache.keys(), key=lambda x: self.cache[x][0])
                del self.cache[oldest_key]
            self.cache[key] = (time.time(), result)
    
    def clear(self):
        with self._lock:
            self.cache.clear()


# ============================================================================
# 검색 히스토리 (최근 검색어 + 인기 검색어)
# ============================================================================
class SearchHistory:
    def __init__(self, max_recent: int = 20, max_popular: int = 10):
        self.recent: List[Dict] = []  # [{query, timestamp}, ...]
        self.popular: Counter = Counter()  # query -> count
        self.max_recent = max_recent
        self.max_popular = max_popular
        self._lock = threading.Lock()
    
    def add(self, query: str):
        """검색어 추가"""
        query = query.strip()
        if len(query) < 2:
            return
        
        with self._lock:
            # 최근 검색어에서 중복 제거
            self.recent = [r for r in self.recent if r['query'] != query]
            # 새 검색어 추가
            self.recent.insert(0, {
                'query': query,
                'timestamp': time.time()
            })
            # 최대 크기 유지
            self.recent = self.recent[:self.max_recent]
            # 인기 검색어 업데이트
            self.popular[query] += 1
    
    def get_recent(self, limit: int = 10) -> List[str]:
        """최근 검색어 반환"""
        with self._lock:
            return [r['query'] for r in self.recent[:limit]]
    
    def get_popular(self, limit: int = 10) -> List[Tuple[str, int]]:
        """인기 검색어 반환 (검색어, 횟수)"""
        with self._lock:
            return self.popular.most_common(min(limit, self.max_popular))
    
    def suggest(self, prefix: str, limit: int = 5) -> List[str]:
        """접두사 기반 검색어 추천"""
        prefix = prefix.strip().lower()
        if len(prefix) < 1:
            return []
        
        with self._lock:
            suggestions = []
            # 최근 검색어에서 매칭
            for r in self.recent:
                if r['query'].lower().startswith(prefix):
                    suggestions.append(r['query'])
            # 인기 검색어에서 매칭
            for q, _ in self.popular.most_common():
                if q.lower().startswith(prefix) and q not in suggestions:
                    suggestions.append(q)
            return suggestions[:limit]
    
    def clear(self):
        """히스토리 초기화"""
        with self._lock:
            self.recent.clear()
            self.popular.clear()


# ============================================================================
# 텍스트 하이라이터
# ============================================================================
class TextHighlighter:
    @staticmethod
    def highlight(text: str, query: str, tag: str = 'mark') -> str:
        """검색어를 태그로 감싸서 하이라이트"""
        if not text or not query:
            return text
        
        # 검색어를 공백으로 분리
        keywords = [kw.strip() for kw in query.split() if len(kw.strip()) >= 2]
        if not keywords:
            return text
        
        # 각 키워드에 대해 하이라이트 적용
        result = text
        for keyword in keywords:
            # 대소문자 무시 검색
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            result = pattern.sub(f'<{tag}>\\g<0></{tag}>', result)
        
        return result
    
    @staticmethod
    def extract_keywords(documents: List[str], top_k: int = 50) -> List[str]:
        """문서에서 핵심 키워드 추출"""
        if not documents:
            return []
        
        # 간단한 키워드 추출 (빈도 기반)
        word_freq = Counter()
        for doc in documents:
            # 한글, 영문 단어 추출
            words = re.findall(r'[가-힣]{2,}|[a-zA-Z]{3,}', doc)
            word_freq.update(words)
        
        # 불용어 필터링 (간단한 한글 불용어)
        stopwords = {'있는', '하는', '및', '등', '이', '가', '을', '를', '의', '에', '로', '으로'}
        keywords = [w for w, _ in word_freq.most_common(top_k * 2) if w not in stopwords]
        
        return keywords[:top_k]


# ============================================================================
# 핵심 QA 시스템 (스레드 안전)
# ============================================================================
class RegulationQASystem:
    def __init__(self):
        self.vector_store = None
        self.embedding_model = None
        self.model_id = None
        self.model_name = ""
        self.extractor = DocumentExtractor()
        self.cache_path = os.path.join(tempfile.gettempdir(), "reg_qa_server_v10")
        self.bm25 = None
        self.documents: List[str] = []
        self.doc_meta: List[Dict] = []
        self.file_infos: Dict[str, FileInfo] = {}
        self.current_folder = ""
        self._lock = threading.RLock()
        self._search_cache = SearchCache(AppConfig.SEARCH_CACHE_SIZE)
        self._search_history = SearchHistory()  # 검색 히스토리
        self._keyword_cache: List[str] = []  # 문서 키워드 캐시
        self._executor = ThreadPoolExecutor(max_workers=AppConfig.MAX_WORKERS)
        self._is_ready = False
        self._is_loading = False
        self._load_progress = ""
    
    def get_keywords(self, limit: int = 50) -> List[str]:
        """문서에서 추출한 키워드 반환 (자동완성용)"""
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
    
    def load_model(self, model_name: str) -> TaskResult:
        """AI 임베딩 모델 로드"""
        if self._is_loading:
            return TaskResult(False, "이미 모델을 로딩 중입니다")
        
        model_id = AppConfig.AVAILABLE_MODELS.get(model_name, AppConfig.AVAILABLE_MODELS[AppConfig.DEFAULT_MODEL])
        
        try:
            self._is_loading = True
            self._load_progress = "라이브러리 로드 중..."
            logger.info("라이브러리 로드 중...")
            
            import torch
            from langchain_huggingface import HuggingFaceEmbeddings
            
            self._load_progress = "모델 다운로드/로딩 중..."
            logger.info(f"모델 로딩 중: {model_name}")
            
            cache_dir = os.path.join(get_app_directory(), 'models')
            os.makedirs(cache_dir, exist_ok=True)
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"사용 디바이스: {device}")
            
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_id,
                cache_folder=cache_dir,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            self.model_id = model_id
            self.model_name = model_name
            self._is_ready = True
            
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            self._load_progress = "완료"
            logger.info(f"모델 로드 완료: {model_name} ({device})")
            return TaskResult(True, f"모델 로드 완료 ({device})")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            self._load_progress = f"실패: {e}"
            return TaskResult(False, f"모델 로드 실패: {e}")
        finally:
            self._is_loading = False
    
    def _get_cache_dir(self, folder: str) -> str:
        if not self.model_id:
            raise ValueError("모델이 로드되지 않았습니다")
        h1 = hashlib.md5(self.model_id.encode()).hexdigest()[:6]
        h2 = hashlib.md5(folder.encode()).hexdigest()[:6]
        return os.path.join(self.cache_path, f"{h2}_{h1}")
    
    def process_documents(self, folder: str, files: List[str], progress_cb=None) -> TaskResult:
        """문서 처리 및 인덱싱"""
        if not self.embedding_model:
            return TaskResult(False, "모델이 로드되지 않았습니다")
        
        with self._lock:
            return self._process_internal(folder, files, progress_cb)
    
    def _process_internal(self, folder: str, files: List[str], progress_cb) -> TaskResult:
        from langchain.text_splitter import CharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain.docstore.document import Document
        
        self.current_folder = folder
        cache_dir = self._get_cache_dir(folder)
        self.file_infos.clear()
        self._search_cache.clear()
        
        # 파일 정보 초기화
        for fp in files:
            meta = FileUtils.get_metadata(fp)
            self.file_infos[fp] = FileInfo(
                fp, os.path.basename(fp),
                os.path.splitext(fp)[1].lower(),
                meta['size'] if meta else 0
            )
        
        if progress_cb:
            progress_cb(5, "캐시 확인...")
        
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
        
        # 캐시된 데이터 로드
        if cached and os.path.exists(os.path.join(cache_dir, "index.faiss")):
            try:
                if progress_cb:
                    progress_cb(10, "캐시 로드...")
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
            if progress_cb:
                progress_cb(100, "완료!")
            return TaskResult(
                True,
                f"캐시에서 {len(cached)}개 파일 로드",
                {'chunks': len(self.documents), 'cached': len(cached), 'new': 0}
            )
        
        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=AppConfig.CHUNK_SIZE,
            chunk_overlap=AppConfig.CHUNK_OVERLAP
        )
        failed, new_docs, new_cache_info = [], [], {}
        
        for i, fp in enumerate(to_process):
            fname = os.path.basename(fp)
            if progress_cb:
                progress_cb(15 + int((i / len(to_process)) * 55), f"처리: {fname}")
            self.file_infos[fp].status = FileStatus.PROCESSING
            
            try:
                content, error = self.extractor.extract(fp)
                if error:
                    failed.append(f"{fname} ({error})")
                    self.file_infos[fp].status = FileStatus.FAILED
                    self.file_infos[fp].error = error
                    continue
                if not content.strip():
                    failed.append(f"{fname} (빈 파일)")
                    self.file_infos[fp].status = FileStatus.FAILED
                    self.file_infos[fp].error = "빈 파일"
                    continue
                
                chunks = splitter.split_text(content)
                chunk_count = 0
                for chunk in chunks:
                    if chunk.strip():
                        new_docs.append(Document(
                            page_content=chunk.strip(),
                            metadata={"source": fname, "path": fp}
                        ))
                        self.documents.append(chunk.strip())
                        self.doc_meta.append({"source": fname, "path": fp})
                        chunk_count += 1
                
                self.file_infos[fp].status = FileStatus.SUCCESS
                self.file_infos[fp].chunks = chunk_count
                
                meta = FileUtils.get_metadata(fp)
                if meta:
                    new_cache_info[fname] = {
                        'size': meta['size'],
                        'mtime': meta['mtime'],
                        'chunks': chunk_count
                    }
            except Exception as e:
                failed.append(f"{fname} ({e})")
                self.file_infos[fp].status = FileStatus.FAILED
                self.file_infos[fp].error = str(e)
        
        if not new_docs and not self.vector_store:
            return TaskResult(False, "처리 가능한 문서 없음", failed_items=failed)
        
        if progress_cb:
            progress_cb(75, "벡터 인덱스 생성...")
        
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
        
        if progress_cb:
            progress_cb(85, "키워드 인덱스 생성...")
        self._build_bm25()
        
        if progress_cb:
            progress_cb(90, "캐시 저장...")
        self._save_cache(cache_dir, cache_info, new_cache_info)
        
        gc.collect()
        if progress_cb:
            progress_cb(100, "완료!")
        
        return TaskResult(
            True,
            f"{len(files) - len(failed)}개 처리 완료",
            {
                'chunks': len(self.documents),
                'new': len(to_process) - len(failed),
                'cached': len(cached)
            },
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
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.debug(f"캐시 정보 로드 실패: {e}")
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
    
    def search(self, query: str, k: int = 3, hybrid: bool = True) -> TaskResult:
        """하이브리드 검색 수행"""
        if not self.vector_store:
            return TaskResult(False, "문서가 로드되지 않음")
        
        query = query.strip()
        if len(query) < 2:
            return TaskResult(False, "검색어가 너무 짧습니다 (최소 2자)")
        
        # 캐시 확인
        cached_result = self._search_cache.get(query, k, hybrid)
        if cached_result is not None:
            return TaskResult(True, "검색 완료 (캐시)", cached_result)
        
        try:
            k = max(1, min(k, AppConfig.MAX_SEARCH_RESULTS))
            
            # 벡터 검색
            vec_results = self.vector_store.similarity_search_with_score(query, k=k*2)
            
            results = {}
            if vec_results and len(vec_results) > 0:
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
            
            # BM25 검색 (하이브리드)
            if hybrid and self.bm25:
                try:
                    bm_res = self.bm25.search(query, top_k=k*2)
                except Exception as bm_err:
                    logger.warning(f"BM25 검색 오류: {bm_err}")
                    bm_res = []
                if bm_res and len(bm_res) > 0:
                    bm_scores = [r[1] for r in bm_res]
                    max_bm = max(bm_scores) if bm_scores else 1
                    for idx, sc in bm_res:
                        if 0 <= idx < len(self.documents):
                            key = self.documents[idx][:100]
                            norm = sc / (max_bm + 0.001)
                            if key in results:
                                results[key]['bm25_score'] = norm
                            else:
                                meta = self.doc_meta[idx] if idx < len(self.doc_meta) else {}
                                results[key] = {
                                    'content': self.documents[idx],
                                    'source': meta.get('source', '?'),
                                    'path': meta.get('path', ''),
                                    'vec_score': 0,
                                    'bm25_score': norm
                                }
            
            # 최종 점수 계산
            for item in results.values():
                item['score'] = (
                    AppConfig.VECTOR_WEIGHT * item['vec_score'] +
                    AppConfig.BM25_WEIGHT * item['bm25_score']
                )
            
            sorted_res = sorted(results.values(), key=lambda x: x['score'], reverse=True)[:k]
            
            # 캐시 저장
            self._search_cache.set(query, k, hybrid, sorted_res)
            
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
    
    def clear_cache(self) -> TaskResult:
        if os.path.exists(self.cache_path):
            shutil.rmtree(self.cache_path, ignore_errors=True)
        self._search_cache.clear()
        return TaskResult(True, "캐시 삭제 완료")
    
    def cleanup(self):
        self.documents.clear()
        self.doc_meta.clear()
        if self.bm25:
            self.bm25.clear()
        self._search_cache.clear()
        try:
            self._executor.shutdown(wait=False)
        except Exception as e:
            logger.debug(f"Executor shutdown error: {e}")
        gc.collect()


# ============================================================================
# Flask 애플리케이션
# ============================================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = AppConfig.MAX_CONTENT_LENGTH
app.config['JSON_AS_ASCII'] = False
CORS(app)

# 전역 QA 시스템
qa_system = RegulationQASystem()

# 업로드 폴더 설정
UPLOAD_DIR = os.path.join(get_app_directory(), AppConfig.UPLOAD_FOLDER)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============================================================================
# API 라우트
# ============================================================================
@app.route('/')
def index():
    """메인 검색 페이지"""
    return render_template('index.html')


@app.route('/admin')
def admin():
    """관리자 페이지"""
    return render_template('admin.html')


@app.route('/api/status')
def api_status():
    """서버 상태 조회"""
    return jsonify({
        'success': True,
        'ready': qa_system.is_ready,
        'loading': qa_system.is_loading,
        'progress': qa_system.load_progress,
        'model': qa_system.model_name,
        'stats': qa_system.get_stats() if qa_system.is_ready else None
    })


@app.route('/api/health')
def api_health():
    """헬스체크 엔드포인트"""
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
    except ImportError:
        cpu_percent = None
        memory_percent = None
    
    return jsonify({
        'status': 'healthy' if qa_system.is_ready else 'initializing',
        'ready': qa_system.is_ready,
        'model_loaded': qa_system.embedding_model is not None,
        'documents_loaded': qa_system.vector_store is not None,
        'document_count': len(qa_system.documents) if qa_system.documents else 0,
        'file_count': len(qa_system.file_infos),
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
        'version': AppConfig.APP_VERSION
    })


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """파일 업로드 및 처리"""
    if not qa_system.is_ready:
        return jsonify({'success': False, 'message': '서버가 준비되지 않았습니다'}), 503
    
    if 'files' not in request.files:
        return jsonify({'success': False, 'message': '파일이 없습니다'}), 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'success': False, 'message': '파일이 선택되지 않았습니다'}), 400
    
    saved_files = []
    for file in files:
        if file and FileUtils.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # 한글 파일명 보존
            if filename != file.filename:
                filename = file.filename.replace('/', '_').replace('\\', '_')
            filepath = os.path.join(UPLOAD_DIR, filename)
            file.save(filepath)
            saved_files.append(filepath)
            logger.info(f"파일 업로드: {filename}")
    
    if not saved_files:
        return jsonify({'success': False, 'message': '지원되는 파일이 없습니다'}), 400
    
    # 문서 처리
    def progress_cb(percent, msg):
        logger.info(f"처리 진행: {percent}% - {msg}")
    
    result = qa_system.process_documents(UPLOAD_DIR, saved_files, progress_cb)
    
    return jsonify({
        'success': result.success,
        'message': result.message,
        'data': result.data,
        'failed': result.failed_items
    })


@app.route('/api/process', methods=['POST'])
def api_process():
    """업로드된 모든 파일 재처리"""
    if not qa_system.is_ready:
        return jsonify({'success': False, 'message': '서버가 준비되지 않았습니다'}), 503
    
    files = [
        os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR)
        if FileUtils.allowed_file(f)
    ]
    
    if not files:
        return jsonify({'success': False, 'message': '처리할 파일이 없습니다'}), 400
    
    result = qa_system.process_documents(UPLOAD_DIR, files, None)
    
    return jsonify({
        'success': result.success,
        'message': result.message,
        'data': result.data,
        'failed': result.failed_items
    })


@app.route('/api/search', methods=['POST'])
def api_search():
    """검색 수행"""
    start_time = time.time()
    
    if not qa_system.is_ready:
        return jsonify({'success': False, 'message': '서버가 준비되지 않았습니다'}), 503
    
    if not qa_system.vector_store:
        return jsonify({'success': False, 'message': '문서가 로드되지 않았습니다'}), 400
    
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'success': False, 'message': '검색어가 필요합니다'}), 400
    
    query = data.get('query', '').strip()
    k = min(data.get('k', AppConfig.DEFAULT_SEARCH_RESULTS), AppConfig.MAX_SEARCH_RESULTS)
    hybrid = data.get('hybrid', True)
    highlight = data.get('highlight', True)  # 하이라이팅 옵션
    
    result = qa_system.search(query, k, hybrid)
    
    # 검색 성공 시 히스토리에 추가
    if result.success and query:
        qa_system._search_history.add(query)
    
    # 하이라이팅 적용
    results_data = result.data if result.success else []
    if highlight and results_data:
        for item in results_data:
            item['content_highlighted'] = TextHighlighter.highlight(item['content'], query)
    
    # 응답 시간 계산
    response_time_ms = round((time.time() - start_time) * 1000, 2)
    
    return jsonify({
        'success': result.success,
        'message': result.message,
        'results': results_data,
        'query': query,
        'response_time_ms': response_time_ms,
        'result_count': len(results_data)
    })


@app.route('/api/search/history')
def api_search_history():
    """검색 히스토리 조회"""
    limit = request.args.get('limit', 10, type=int)
    recent = qa_system._search_history.get_recent(limit)
    popular = qa_system._search_history.get_popular(limit)
    
    return jsonify({
        'success': True,
        'recent': recent,
        'popular': [{'query': q, 'count': c} for q, c in popular]
    })


@app.route('/api/search/suggest')
def api_search_suggest():
    """검색어 자동완성 제안"""
    prefix = request.args.get('q', '').strip()
    limit = request.args.get('limit', 8, type=int)
    
    if len(prefix) < 1:
        return jsonify({'success': True, 'suggestions': []})
    
    suggestions = []
    
    # 1. 검색 히스토리에서 매칭
    history_suggestions = qa_system._search_history.suggest(prefix, limit)
    suggestions.extend(history_suggestions)
    
    # 2. 문서 키워드에서 매칭
    if len(suggestions) < limit:
        keywords = qa_system.get_keywords()
        prefix_lower = prefix.lower()
        for kw in keywords:
            if kw.lower().startswith(prefix_lower) and kw not in suggestions:
                suggestions.append(kw)
                if len(suggestions) >= limit:
                    break
    
    return jsonify({
        'success': True,
        'suggestions': suggestions[:limit]
    })


@app.route('/api/files')
def api_files():
    """로드된 파일 목록"""
    return jsonify({
        'success': True,
        'files': qa_system.get_file_infos(),
        'stats': qa_system.get_stats()
    })


@app.route('/api/cache', methods=['DELETE'])
def api_clear_cache():
    """캐시 삭제"""
    result = qa_system.clear_cache()
    return jsonify({'success': result.success, 'message': result.message})


@app.route('/api/models')
def api_models():
    """사용 가능한 모델 목록"""
    return jsonify({
        'success': True,
        'models': list(AppConfig.AVAILABLE_MODELS.keys()),
        'current': qa_system.model_name
    })


@app.route('/api/files/<filename>', methods=['DELETE'])
def api_delete_file(filename):
    """개별 파일 삭제"""
    if not qa_system.is_ready:
        return jsonify({'success': False, 'message': '서버가 준비되지 않았습니다'}), 503
    
    # 경로 검증 (경로 탐색 공격 방지)
    safe_filename = secure_filename(filename)
    if safe_filename != filename and not filename.replace(' ', '_') == safe_filename:
        # 한글 파일명 처리
        safe_filename = filename.replace('/', '_').replace('\\', '_').replace('..', '')
    
    filepath = os.path.join(UPLOAD_DIR, safe_filename)
    
    # 파일 존재 확인
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'message': '파일을 찾을 수 없습니다'}), 404
    
    try:
        # 파일 삭제
        os.remove(filepath)
        logger.info(f"파일 삭제: {safe_filename}")
        
        # 인덱스에서 해당 파일 관련 데이터 제거
        if filepath in qa_system.file_infos:
            del qa_system.file_infos[filepath]
        
        # 검색 캐시 무효화
        qa_system._search_cache.clear()
        qa_system._keyword_cache.clear()
        
        # 남은 파일로 인덱스 재구성 필요 알림
        remaining_files = [
            os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR)
            if FileUtils.allowed_file(f)
        ]
        
        return jsonify({
            'success': True,
            'message': f'{safe_filename} 삭제 완료',
            'remaining_files': len(remaining_files),
            'reindex_required': True  # 프론트엔드에서 재처리 안내
        })
        
    except OSError as e:
        logger.error(f"파일 삭제 실패: {e}")
        return jsonify({'success': False, 'message': f'삭제 실패: {e}'}), 500


@app.route('/api/files/<filename>/preview')
def api_file_preview(filename):
    """파일 내용 미리보기"""
    if not qa_system.is_ready:
        return jsonify({'success': False, 'message': '서버가 준비되지 않았습니다'}), 503
    
    # 경로 검증
    safe_filename = filename.replace('/', '_').replace('\\', '_').replace('..', '')
    filepath = os.path.join(UPLOAD_DIR, safe_filename)
    
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'message': '파일을 찾을 수 없습니다'}), 404
    
    try:
        # 파일 내용 추출
        content, error = qa_system.extractor.extract(filepath)
        
        if error:
            return jsonify({
                'success': False,
                'message': f'파일 읽기 실패: {error}'
            }), 400
        
        # 미리보기 길이 제한 (기본 2000자)
        max_length = request.args.get('length', 2000, type=int)
        max_length = min(max_length, 5000)  # 최대 5000자
        
        preview_content = content[:max_length]
        is_truncated = len(content) > max_length
        
        # 파일 정보
        file_info = qa_system.file_infos.get(filepath)
        
        return jsonify({
            'success': True,
            'filename': safe_filename,
            'content': preview_content,
            'total_length': len(content),
            'is_truncated': is_truncated,
            'chunks': file_info.chunks if file_info else 0,
            'status': file_info.status.value if file_info else 'unknown'
        })
        
    except Exception as e:
        logger.error(f"미리보기 실패: {e}")
        return jsonify({'success': False, 'message': f'미리보기 실패: {e}'}), 500


# ============================================================================
# 메인 실행
# ============================================================================
def initialize_server():
    """서버 초기화 - 모델 로드"""
    logger.info("=" * 60)
    logger.info(f"🚀 {AppConfig.APP_NAME} {AppConfig.APP_VERSION}")
    logger.info("=" * 60)
    
    # 모델 로드
    result = qa_system.load_model(AppConfig.DEFAULT_MODEL)
    if result.success:
        logger.info(f"✅ {result.message}")
        
        # 기존 업로드된 파일 자동 처리
        if os.path.exists(UPLOAD_DIR):
            files = [
                os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR)
                if FileUtils.allowed_file(f)
            ]
            if files:
                logger.info(f"📂 기존 파일 {len(files)}개 처리 중...")
                result = qa_system.process_documents(UPLOAD_DIR, files, None)
                if result.success:
                    logger.info(f"✅ {result.message}")
                else:
                    logger.warning(f"⚠️ {result.message}")
    else:
        logger.error(f"❌ 모델 로드 실패: {result.message}")


def graceful_shutdown(signum=None, frame=None):
    """서버 정상 종료 처리"""
    logger.info("🛑 서버 종료 중...")
    qa_system.cleanup()
    logger.info("✅ 정상 종료 완료")
    sys.exit(0)


# 종료 시그널 핸들러 등록
atexit.register(graceful_shutdown)


if __name__ == '__main__':
    # SIGINT, SIGTERM 핸들러 등록
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    
    # 서버 초기화 (별도 스레드)
    init_thread = threading.Thread(target=initialize_server, daemon=True)
    init_thread.start()
    
    logger.info(f"🌐 서버 시작: http://localhost:{AppConfig.SERVER_PORT}")
    logger.info(f"📚 관리자 페이지: http://localhost:{AppConfig.SERVER_PORT}/admin")
    logger.info("=" * 60)
    
    # 프로덕션 서버 (waitress 권장, 없으면 Flask 기본)
    try:
        from waitress import serve
        logger.info("🚀 Waitress 프로덕션 서버로 실행")
        serve(app, host=AppConfig.SERVER_HOST, port=AppConfig.SERVER_PORT, threads=8)
    except ImportError:
        logger.warning("⚠️ waitress 없음 - Flask 개발 서버 사용 (프로덕션에서는 waitress 권장)")
        app.run(
            host=AppConfig.SERVER_HOST,
            port=AppConfig.SERVER_PORT,
            debug=False,
            threaded=True
        )
