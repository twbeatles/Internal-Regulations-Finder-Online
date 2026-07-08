# -*- coding: utf-8 -*-
"""FAISS/BM25 인덱스 디스크 캐시 저장·로드."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from typing import Any, Dict

from app.config import AppConfig
from app.utils import logger

CACHE_KEY_MODE = "relative_path_v1"


def get_cache_dir(qa, folder: str) -> str:
    """FAISS 캐시 디렉토리 경로 생성.

    BM25-only 모드(모델 미로드)도 캐시 경로를 생성합니다.
    """
    model_key = qa.model_id or "bm25-only"
    h1 = hashlib.md5(model_key.encode()).hexdigest()[:6]
    h2 = hashlib.md5(folder.encode()).hexdigest()[:6]
    return os.path.join(qa.cache_path, f"{h2}_{h1}")


def get_cache_entry_key(folder: str, file_path: str) -> str:
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


def load_cache_info(qa, cache_dir: str) -> Dict[str, Any]:
    """캐시 정보 로드 및 설정 검증 (불일치 시 빈 딕셔너리 반환)."""
    path = os.path.join(cache_dir, "cache_info.json")
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                cache_info = json.load(f)

            cache_meta = cache_info.get('_cache_meta', {})
            if cache_meta:
                mismatches = []
                if cache_meta.get('cache_key_mode') != CACHE_KEY_MODE:
                    mismatches.append(
                        f"cache_key_mode: {cache_meta.get('cache_key_mode')} -> {CACHE_KEY_MODE}"
                    )

                expected_model_id = qa.model_id or "bm25-only"
                if cache_meta.get('model_id') != expected_model_id:
                    mismatches.append(f"model_id: {cache_meta.get('model_id')} -> {expected_model_id}")

                current_backend = getattr(AppConfig, 'EMBED_BACKEND', 'torch')
                if cache_meta.get('embed_backend') != current_backend:
                    mismatches.append(f"embed_backend: {cache_meta.get('embed_backend')} -> {current_backend}")

                current_normalize = getattr(AppConfig, 'EMBED_NORMALIZE', True)
                if cache_meta.get('embed_normalize') != current_normalize:
                    mismatches.append(f"embed_normalize: {cache_meta.get('embed_normalize')} -> {current_normalize}")

                if cache_meta.get('chunk_size') != AppConfig.CHUNK_SIZE:
                    mismatches.append(f"chunk_size: {cache_meta.get('chunk_size')} -> {AppConfig.CHUNK_SIZE}")
                if cache_meta.get('chunk_overlap') != AppConfig.CHUNK_OVERLAP:
                    mismatches.append(f"chunk_overlap: {cache_meta.get('chunk_overlap')} -> {AppConfig.CHUNK_OVERLAP}")

                if cache_meta.get('vector_weight') != AppConfig.VECTOR_WEIGHT:
                    mismatches.append(f"vector_weight: {cache_meta.get('vector_weight')} -> {AppConfig.VECTOR_WEIGHT}")
                if cache_meta.get('bm25_weight') != AppConfig.BM25_WEIGHT:
                    mismatches.append(f"bm25_weight: {cache_meta.get('bm25_weight')} -> {AppConfig.BM25_WEIGHT}")

                if mismatches:
                    logger.warning(f"⚠️ 캐시 무효화 - 설정 변경 감지: {', '.join(mismatches)}")
                    try:
                        shutil.rmtree(cache_dir)
                        logger.info(f"캐시 디렉토리 삭제됨: {cache_dir}")
                    except Exception as e:
                        logger.warning(f"캐시 디렉토리 삭제 실패: {e}")
                    return {}

            return {k: v for k, v in cache_info.items() if k != '_cache_meta'}

        except json.JSONDecodeError as e:
            logger.debug(f"캐시 정보 JSON 파싱 실패: {path} - {e}")
        except IOError as e:
            logger.debug(f"캐시 정보 파일 읽기 실패: {path} - {e}")
        except Exception as e:
            logger.debug(f"캐시 정보 로드 중 예상치 못한 오류: {path} - {e}")
    return {}


def save_cache(qa, cache_dir: str, old_info: Dict[str, Any], new_info: Dict[str, Any]) -> None:
    try:
        os.makedirs(cache_dir, exist_ok=True)
        if qa.vector_store and hasattr(qa.vector_store, "save_local"):
            qa.vector_store.save_local(cache_dir)

        cache_meta = {
            'model_id': qa.model_id or "bm25-only",
            'cache_key_mode': CACHE_KEY_MODE,
            'embed_backend': getattr(AppConfig, 'EMBED_BACKEND', 'torch'),
            'embed_normalize': getattr(AppConfig, 'EMBED_NORMALIZE', True),
            'chunk_size': AppConfig.CHUNK_SIZE,
            'chunk_overlap': AppConfig.CHUNK_OVERLAP,
            'vector_weight': AppConfig.VECTOR_WEIGHT,
            'bm25_weight': AppConfig.BM25_WEIGHT,
        }

        cache_info = {**old_info, **new_info, '_cache_meta': cache_meta}

        with open(os.path.join(cache_dir, "cache_info.json"), 'w', encoding='utf-8') as f:
            json.dump(cache_info, f, ensure_ascii=False)
        with open(os.path.join(cache_dir, "docs.json"), 'w', encoding='utf-8') as f:
            json.dump({'docs': qa.documents, 'meta': qa.doc_meta}, f, ensure_ascii=False)

        logger.debug(f"캐시 저장 완료: {cache_dir} (meta: {cache_meta})")
    except Exception as e:
        logger.warning(f"캐시 저장 실패: {e}")