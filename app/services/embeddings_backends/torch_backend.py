# -*- coding: utf-8 -*-
"""
Torch 기반 임베딩 백엔드

기존 search.py의 HuggingFaceEmbeddings 생성 로직을 모듈화
"""

import os
import importlib
from typing import Any, Optional

from app.utils import logger
from app.config import AppConfig
from app.exceptions import ModelLoadError, ModelOfflineError

# Lazy import 변수
HuggingFaceEmbeddings: Any | None = None
_torch_imports_loaded = False


def _lazy_import_torch_deps():
    """PyTorch/LangChain 의존성 지연 로드"""
    global HuggingFaceEmbeddings, _torch_imports_loaded
    
    if _torch_imports_loaded:
        return
    
    for module_name in (
        'langchain_huggingface',
        'langchain_community.embeddings',
        'langchain.embeddings',
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        HuggingFaceEmbeddings = getattr(module, 'HuggingFaceEmbeddings', None)
        if HuggingFaceEmbeddings is not None:
            break
    
    _torch_imports_loaded = True


def create_torch_embeddings(
    model_path_or_id: str,
    normalize: bool = True,
    device: Optional[str] = None,
    cache_folder: Optional[str] = None,
    offline_mode: bool = False
):
    """
    PyTorch 기반 HuggingFaceEmbeddings 생성
    
    Args:
        model_path_or_id: 모델 경로 또는 HuggingFace 모델 ID
        normalize: L2 정규화 여부
        device: 'cuda' 또는 'cpu' (None이면 자동 감지)
        cache_folder: 모델 캐시 폴더
        offline_mode: 오프라인 모드 여부
        
    Returns:
        HuggingFaceEmbeddings 인스턴스
        
    Raises:
        ModelLoadError: 모델 로드 실패 시
        ModelOfflineError: 오프라인 모드에서 로컬 모델을 찾을 수 없을 때
    """
    try:
        # Lazy import 실행
        _lazy_import_torch_deps()
        
        if HuggingFaceEmbeddings is None:
            raise ModelLoadError(model_path_or_id, "LangChain HuggingFaceEmbeddings를 찾을 수 없습니다")
        
        # PyTorch import
        try:
            torch = importlib.import_module('torch')
        except ImportError as e:
            raise ModelLoadError(model_path_or_id, f"PyTorch 로드 실패: {e}")
        
        # 디바이스 자동 감지
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Torch 백엔드 초기화: {model_path_or_id} (device: {device})")
        
        # 오프라인 모드 환경변수 설정
        if offline_mode:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            
            # 로컬 경로 확인
            if not os.path.exists(model_path_or_id):
                raise ModelOfflineError(model_path_or_id)
            
            logger.info(f"오프라인 모드: 로컬 모델 사용 - {model_path_or_id}")
        
        # 로컬 모델 자동 감지 (models 폴더에 이미 다운로드된 경우)
        if cache_folder and os.path.isdir(model_path_or_id):
            has_model_files = (
                os.path.exists(os.path.join(model_path_or_id, 'config.json')) or
                os.path.exists(os.path.join(model_path_or_id, 'pytorch_model.bin')) or
                os.path.exists(os.path.join(model_path_or_id, 'model.safetensors'))
            )
            if has_model_files:
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                logger.info(f"로컬 모델 자동 감지: {model_path_or_id}")
        
        # HuggingFaceEmbeddings 생성
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': normalize}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path_or_id,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder=cache_folder
        )
        
        logger.info(f"Torch 임베딩 모델 로드 완료: {model_path_or_id}")
        return embeddings
        
    except (ModelLoadError, ModelOfflineError):
        raise
    except Exception as e:
        logger.error(f"Torch 백엔드 초기화 실패: {e}")
        raise ModelLoadError(model_path_or_id, str(e))
