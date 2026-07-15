# -*- coding: utf-8 -*-
"""RegulationQASystem — 인덱싱·검색 파사드."""
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
import importlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from app.config import AppConfig
from app.utils import logger, FileInfo, FileStatus, TaskResult, FileUtils, get_app_directory
from app.constants import ErrorMessages, SuccessMessages, InfoMessages, Limits, Weights, Patterns
from app.exceptions import (
    ModelNotLoadedError, ModelLoadError, ModelOfflineError,
    SearchError, SearchTimeoutError, SearchRateLimitError, SearchQueueFullError,
    DocumentExtractionError, IndexingError, CacheError
)
from app.services.db import db
from app.services import search as _search_exports
from app.services.document import DocumentExtractor, TextHighlighter, ArticleParser
from app.services.metadata import TagManager
from app.services.file_manager import RevisionTracker

from app.services.search.langchain_lazy import _lazy_import_langchain, CharacterTextSplitter, HuggingFaceEmbeddings
from app.services.search.bm25 import BM25Light
from app.services.search.cache import SearchCache
from app.services.search.rate_limiter import RateLimiter
from app.services.search.queue import SearchQueue
from app.services.search.history import SearchHistory
from app.services.search.hybrid_search import HybridSearchService
from app.services.search.index import (
    bm25_index_service,
    build_chunk_meta,
    get_cache_dir,
    get_cache_entry_key,
    index_lifecycle,
    load_cache_info,
    normalize_doc_meta,
    remember_file_details,
    save_cache,
    vector_index_service,
)

_hybrid_search = HybridSearchService()


class RegulationQASystem:
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
        self.doc_splitter = _search_exports.DocumentSplitter()

    def _normalize_doc_meta_locked(self) -> None:
        normalize_doc_meta(self)

    def _rebuild_vector_store_locked(self) -> None:
        vector_index_service.rebuild(self)

    def _clear_index_locked(self, *, preserve_folder: bool = True) -> None:
        index_lifecycle.clear(self, preserve_folder=preserve_folder)

    def clear_index(self, *, preserve_folder: bool = True) -> None:
        with self._lock:
            self._clear_index_locked(preserve_folder=preserve_folder)

    def remove_file_from_index(self, target_path: str, resolved_name: str, resolved_id: str) -> bool:
        with self._lock:
            return index_lifecycle.remove_file(self, target_path, resolved_name, resolved_id)

    def _get_cache_entry_key(self, folder: str, file_path: str) -> str:
        return get_cache_entry_key(folder, file_path)
    
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
        return get_cache_dir(self, folder)

    def _remember_file_details(self, file_path: str, extracted) -> Dict[str, Any]:
        return remember_file_details(self, file_path, extracted)

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
        article_no: str = "",
        article_title: str = "",
        chunk_type: str = "generic",
    ) -> Dict[str, Any]:
        return build_chunk_meta(
            doc_id=doc_id,
            filename=filename,
            file_path=file_path,
            file_id=file_id,
            chunk_id=chunk_id,
            total_chunks=total_chunks,
            details=details,
            article_no=article_no,
            article_title=article_title,
            chunk_type=chunk_type,
        )
    
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
            
            # 문서 분할 (조문 인식 청킹 우선)
            from rag.pipeline.chunking import build_chunks_from_text

            splitter = _search_exports.DocumentSplitter(
                chunk_size=AppConfig.CHUNK_SIZE,
                chunk_overlap=AppConfig.CHUNK_OVERLAP,
            )
            chunk_items = build_chunks_from_text(
                text,
                chunk_size=AppConfig.CHUNK_SIZE,
                chunk_overlap=AppConfig.CHUNK_OVERLAP,
                article_parser=self.article_parser,
                doc_splitter=splitter,
            )
            chunks = [item["text"] for item in chunk_items]

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
                for i, item in enumerate(chunk_items):
                    chunk = item["text"]
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
                            article_no=str(item.get("article_no", "")),
                            article_title=str(item.get("article_title", "")),
                            chunk_type=str(item.get("chunk_type", "generic")),
                        )
                    )
                
                # 벡터스토어 업데이트 (있으면 추가, 없으면 생성)
                if self.embedding_model and _search_exports.FAISS and _search_exports.Document:
                    try:
                        docs = []
                        base_doc_id = len(self.documents) - len(chunks)
                        for i, item in enumerate(chunk_items):
                            chunk = item["text"]
                            docs.append(
                                _search_exports.Document(
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
                                            article_no=str(item.get("article_no", "")),
                                            article_title=str(item.get("article_title", "")),
                                            chunk_type=str(item.get("chunk_type", "generic")),
                                        ),
                                    },
                                )
                            )
                        
                        if self.vector_store:
                            self.vector_store.add_documents(docs)
                        else:
                            self.vector_store = _search_exports.FAISS.from_documents(docs, self.embedding_model)
                        
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
            and _search_exports.FAISS
            and cached
            and os.path.exists(os.path.join(cache_dir, "index.faiss"))
        ):
            try:
                if progress_cb: progress_cb(10, "캐시 로드...")
                from app.services.search.index.cache_store import (
                    get_stored_integrity,
                    verify_cache_integrity,
                )
                if not verify_cache_integrity(cache_dir, get_stored_integrity(cache_dir)):
                    raise ValueError("캐시 무결성 검증 실패")
                # FAISS 로컬 로드는 pickle 기반일 수 있음 — 무결성 검증 후 로드
                self.vector_store = _search_exports.FAISS.load_local(
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
        
        splitter = _search_exports.DocumentSplitter(chunk_size=AppConfig.CHUNK_SIZE, chunk_overlap=AppConfig.CHUNK_OVERLAP)
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
                from rag.pipeline.chunking import build_chunks_from_text

                details = self._remember_file_details(fp, extracted)
                chunk_items = build_chunks_from_text(
                    content,
                    chunk_size=AppConfig.CHUNK_SIZE,
                    chunk_overlap=AppConfig.CHUNK_OVERLAP,
                    article_parser=self.article_parser,
                    doc_splitter=splitter,
                )
                chunk_count = 0
                for item in chunk_items:
                    if self._cancel_event.is_set():
                        return self._cancelled_result(progress_cb)
                    chunk_text = str(item.get("text", "")).strip()
                    if not chunk_text:
                        continue
                    doc_id = len(self.documents)
                    meta = self._build_chunk_meta(
                        doc_id=doc_id,
                        filename=fname,
                        file_path=fp,
                        file_id=file_id,
                        chunk_id=chunk_count,
                        total_chunks=len(chunk_items),
                        details=details,
                        article_no=str(item.get("article_no", "")),
                        article_title=str(item.get("article_title", "")),
                        chunk_type=str(item.get("chunk_type", "generic")),
                    )
                    if self.embedding_model and _search_exports.FAISS and _search_exports.Document:
                        new_docs.append(
                            _search_exports.Document(
                                page_content=chunk_text,
                                metadata=meta,
                            )
                        )
                    self.documents.append(chunk_text)
                    self.doc_meta.append(meta)
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
        if self.embedding_model and _search_exports.FAISS:
            if progress_cb: progress_cb(75, "벡터 인덱스 생성...")
            try:
                if new_docs:
                    if self.vector_store:
                        batch_size = 100
                        for i in range(0, len(new_docs), batch_size):
                            self.vector_store.add_documents(new_docs[i:i + batch_size])
                    else:
                        self.vector_store = _search_exports.FAISS.from_documents(new_docs, self.embedding_model)
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
        bm25_index_service.build(self)

    def _load_cache_info(self, cache_dir: str) -> Dict:
        return load_cache_info(self, cache_dir)

    def _save_cache(self, cache_dir: str, old_info: Dict, new_info: Dict):
        save_cache(self, cache_dir, old_info, new_info)

    def initialize(self, folder_path: str, force_reindex: bool = False) -> TaskResult:
        # 단일 비행: 진행 중이면 중복 백그라운드 인덱싱 거부
        with self._lock:
            if self._is_loading:
                from app.constants import ErrorMessages
                return TaskResult(False, ErrorMessages.SYNC_ALREADY_RUNNING)

        # AI 모델 로드 시도 (실패해도 BM25로 계속 진행)
        if not self._is_ready and not self.embedding_model:
            try:
                res = self.load_model(AppConfig.DEFAULT_MODEL)
                if not res.success:
                    logger.warning(f"AI 모델 로드 실패, BM25 모드로 진행: {res.message}")
            except Exception as e:
                logger.warning(f"AI 모델 로드 오류, BM25 모드로 진행: {e}")

        with self._lock:
            if self._is_loading:
                from app.constants import ErrorMessages
                return TaskResult(False, ErrorMessages.SYNC_ALREADY_RUNNING)
            self._is_loading = True
            self._load_progress = "인덱싱 준비 중..."
            self._load_error = ""
            self._cancel_event.clear()
            self._cancel_reason = ""
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
                with self._lock:
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
        """검색 수행 — HybridSearchService에 위임."""
        return _hybrid_search.search(
            self, query, k, hybrid, sort_by, filter_file, filter_file_id
        )
            
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

