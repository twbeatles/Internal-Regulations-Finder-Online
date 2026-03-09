# -*- coding: utf-8 -*-
"""
임베딩 백엔드 팩토리

설정에 따라 torch, onnx_fp32, onnx_int8 백엔드를 선택하고
실패 시 자동 fallback 처리
"""

import os
from typing import Any, Optional

from app.utils import logger, get_app_directory
from app.config import AppConfig
from app.exceptions import ModelLoadError, ModelOfflineError


def create_embeddings(
    model_name: str,
    model_id_or_path: str,
    backend: Optional[str] = None,
    normalize: Optional[bool] = None,
    offline_mode: Optional[bool] = None,
    local_model_path: Optional[str] = None
) -> Any:
    """
    임베딩 모델 생성 팩토리
    
    Args:
        model_name: 모델 표시 이름 (로깅용)
        model_id_or_path: HuggingFace 모델 ID 또는 로컬 경로
        backend: 백엔드 선택 ("torch", "onnx_fp32", "onnx_int8"), None이면 AppConfig 사용
        normalize: L2 정규화 여부, None이면 AppConfig 사용
        offline_mode: 오프라인 모드, None이면 AppConfig 사용
        local_model_path: 로컬 모델 경로 override
        
    Returns:
        임베딩 모델 인스턴스 (HuggingFaceEmbeddings 또는 OnnxEmbeddings)
        
    Raises:
        ModelLoadError: 모든 백엔드에서 로드 실패
        ModelOfflineError: 오프라인 모드에서 로컬 모델 없음
    """
    # 설정값 결정
    backend_name = str(backend if backend is not None else getattr(AppConfig, 'EMBED_BACKEND', 'torch'))
    normalize_embeddings = bool(
        normalize if normalize is not None else getattr(AppConfig, 'EMBED_NORMALIZE', True)
    )
    is_offline = bool(offline_mode if offline_mode is not None else AppConfig.OFFLINE_MODE)
    model_path_override = str(local_model_path if local_model_path is not None else (AppConfig.LOCAL_MODEL_PATH or ""))
    
    # 프로젝트 models 폴더 경로
    project_models_dir = os.path.join(get_app_directory(), 'models')
    os.makedirs(project_models_dir, exist_ok=True)
    
    # HuggingFace 캐시 디렉토리 설정
    os.environ['HF_HOME'] = project_models_dir
    os.environ['HUGGINGFACE_HUB_CACHE'] = project_models_dir
    os.environ['TRANSFORMERS_CACHE'] = project_models_dir
    
    # 로컬 모델 경로 결정
    local_model_dir = model_path_override or os.path.join(
        project_models_dir, 
        model_id_or_path.replace('/', '--')
    )
    
    # 로컬 모델 존재 확인
    has_local_model = os.path.exists(local_model_dir) and os.path.isdir(local_model_dir)
    if has_local_model:
        has_model_files = (
            os.path.exists(os.path.join(local_model_dir, 'config.json')) or
            os.path.exists(os.path.join(local_model_dir, 'pytorch_model.bin')) or
            os.path.exists(os.path.join(local_model_dir, 'model.safetensors'))
        )
        has_local_model = has_model_files
    
    # 최종 모델 경로 결정
    if has_local_model:
        load_path = local_model_dir
        logger.info(f"로컬 모델 사용: {load_path}")
    elif is_offline:
        raise ModelOfflineError(local_model_dir)
    else:
        load_path = model_id_or_path
    
    logger.info(
        f"임베딩 생성 시작: backend={backend_name}, model={model_name}, normalize={normalize_embeddings}"
    )
    
    # 백엔드 선택
    if backend_name.startswith('onnx'):
        return _create_onnx_with_fallback(
            model_name=model_name,
            model_path=load_path,
            backend=backend_name,
            normalize=normalize_embeddings,
            is_offline=is_offline,
            cache_folder=project_models_dir
        )
    else:
        # torch 백엔드
        return _create_torch_embeddings(
            model_name=model_name,
            load_path=load_path,
            normalize=normalize_embeddings,
            is_offline=is_offline,
            cache_folder=project_models_dir
        )


def _create_torch_embeddings(
    model_name: str,
    load_path: str,
    normalize: bool,
    is_offline: bool,
    cache_folder: str
) -> Any:
    """Torch 백엔드로 임베딩 생성"""
    from app.services.embeddings_backends.torch_backend import create_torch_embeddings
    
    return create_torch_embeddings(
        model_path_or_id=load_path,
        normalize=normalize,
        offline_mode=is_offline,
        cache_folder=cache_folder
    )


def _create_onnx_with_fallback(
    model_name: str,
    model_path: str,
    backend: str,
    normalize: bool,
    is_offline: bool,
    cache_folder: str
) -> Any:
    """
    ONNX 백엔드로 임베딩 생성 (실패 시 torch로 fallback)
    
    Args:
        model_name: 모델 표시 이름
        model_path: 모델 경로
        backend: "onnx_fp32" 또는 "onnx_int8"
        normalize: L2 정규화
        is_offline: 오프라인 모드
        cache_folder: 캐시 폴더
        
    Returns:
        OnnxEmbeddings 또는 HuggingFaceEmbeddings (fallback)
    """
    # ONNX 파일명 결정
    if backend == 'onnx_int8':
        onnx_file = 'model.int8.onnx'
        fallback_onnx: Optional[str] = 'model.onnx'
    else:
        onnx_file = 'model.onnx'
        fallback_onnx = None
    
    onnx_path = os.path.join(model_path, onnx_file)
    
    # int8 파일이 없으면 fp32로 fallback
    if backend == 'onnx_int8' and not os.path.exists(onnx_path):
        if fallback_onnx is not None:
            fallback_path = os.path.join(model_path, fallback_onnx)
            if os.path.exists(fallback_path):
                logger.warning(f"ONNX int8 파일 없음, fp32로 fallback: {fallback_path}")
                onnx_file = fallback_onnx
                onnx_path = fallback_path
    
    # ONNX 파일 존재 확인
    if not os.path.exists(onnx_path):
        logger.warning(f"ONNX 파일 없음 ({onnx_path}), torch 백엔드로 fallback")
        return _create_torch_embeddings(
            model_name=model_name,
            load_path=model_path,
            normalize=normalize,
            is_offline=is_offline,
            cache_folder=cache_folder
        )
    
    # ONNX 백엔드 시도
    try:
        from app.services.embeddings_backends.onnx_backend import create_onnx_embeddings
        
        embeddings = create_onnx_embeddings(
            model_path=model_path,
            onnx_file=onnx_file,
            normalize=normalize,
            offline_mode=is_offline
        )
        
        logger.info(f"ONNX 백엔드 로드 성공: {onnx_file}")
        return embeddings
        
    except ImportError as e:
        logger.warning(f"onnxruntime 패키지 없음, torch 백엔드로 fallback: {e}")
    except ModelLoadError as e:
        logger.warning(f"ONNX 백엔드 로드 실패, torch 백엔드로 fallback: {e}")
    except Exception as e:
        logger.warning(f"ONNX 백엔드 예상치 못한 오류, torch 백엔드로 fallback: {e}")
    
    # Torch fallback
    return _create_torch_embeddings(
        model_name=model_name,
        load_path=model_path,
        normalize=normalize,
        is_offline=is_offline,
        cache_folder=cache_folder
    )


def validate_embeddings_quality(
    torch_embeddings: Any,
    onnx_embeddings: Any,
    test_sentences: Optional[list] = None
) -> dict:
    """
    Torch와 ONNX 임베딩 품질 비교 검증 (개발 모드용)
    
    Args:
        torch_embeddings: Torch 임베딩 모델
        onnx_embeddings: ONNX 임베딩 모델
        test_sentences: 테스트 문장 리스트 (None이면 기본값 사용)
        
    Returns:
        {
            'avg_similarity': float,
            'min_similarity': float,
            'max_similarity': float,
            'passed': bool,
            'threshold': float
        }
    """
    import numpy as np
    
    if test_sentences is None:
        test_sentences = [
            "이 문서는 사내 규정에 대한 내용입니다.",
            "직원들은 출퇴근 시간을 준수해야 합니다.",
            "연차 휴가는 1년에 15일이 부여됩니다.",
            "회의실 예약은 사전에 시스템을 통해 진행합니다.",
            "보안 정책에 따라 비밀번호를 주기적으로 변경해야 합니다.",
            "경비 지출은 팀장 승인 후 처리됩니다.",
            "신입 사원 교육은 입사 첫 주에 진행됩니다.",
            "야근 시 식대가 지원됩니다.",
            "재택근무는 주 2회까지 가능합니다.",
            "성과 평가는 분기별로 진행됩니다."
        ]
    
    try:
        # 임베딩 생성
        torch_vecs = np.array(torch_embeddings.embed_documents(test_sentences))
        onnx_vecs = np.array(onnx_embeddings.embed_documents(test_sentences))
        
        # 코사인 유사도 계산
        similarities = []
        for i in range(len(test_sentences)):
            torch_vec = torch_vecs[i]
            onnx_vec = onnx_vecs[i]
            
            # L2 정규화된 벡터의 코사인 유사도 = dot product
            sim = np.dot(torch_vec, onnx_vec)
            similarities.append(sim)
        
        avg_sim = float(np.mean(similarities))
        min_sim = float(np.min(similarities))
        max_sim = float(np.max(similarities))
        
        # 기준: fp32는 0.99 이상, int8은 0.985 이상
        threshold = 0.99
        passed = avg_sim >= threshold
        
        result = {
            'avg_similarity': round(avg_sim, 4),
            'min_similarity': round(min_sim, 4),
            'max_similarity': round(max_sim, 4),
            'passed': passed,
            'threshold': threshold
        }
        
        if passed:
            logger.info(f"✅ 임베딩 품질 검증 통과: avg={avg_sim:.4f}, min={min_sim:.4f}")
        else:
            logger.warning(f"⚠️ 임베딩 품질 미달: avg={avg_sim:.4f} < {threshold}")
        
        return result
        
    except Exception as e:
        logger.error(f"임베딩 품질 검증 실패: {e}")
        return {
            'error': str(e),
            'passed': False
        }
