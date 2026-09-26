# -*- coding: utf-8 -*-
"""
Torch 기반 임베딩 백엔드

기존 search.py의 HuggingFaceEmbeddings 생성 로직을 모듈화
"""

import os
import importlib
from typing import Any, List, Optional

from app.utils import logger
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


class PrefixEmbeddingWrapper:
    """E5 계열 query/passage prefix를 적용하는 LangChain 호환 래퍼

    E5 계열 모델은 학습 시 "query: ..."/"passage: ..." 접두사로
    쿼리/문서를 구분했으므로, 추론 시 같은 접두사를 붙여야
    검색 품질이 정상적으로 나온다. FAISS는 embed_documents로
    인덱싱하고 embed_query로 검색하므로, 양쪽에 각 prefix를 적용한다.
    """

    def __init__(
        self,
        inner: Any,
        query_prefix: str = "",
        passage_prefix: str = "",
    ):
        self._inner = inner
        self._query_prefix = query_prefix or ""
        self._passage_prefix = passage_prefix or ""

    def _with_prefix(self, text: str, prefix: str) -> str:
        if prefix and not text.startswith(prefix):
            return prefix + text
        return text

    def embed_query(self, text: str) -> List[float]:
        return self._inner.embed_query(self._with_prefix(text, self._query_prefix))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [self._with_prefix(t, self._passage_prefix) for t in texts]
        return self._inner.embed_documents(prefixed)

    def __getattr__(self, name: str) -> Any:
        # inner 모델의 나머지 속성(model 등)을 그대로 노출
        return getattr(self.__dict__["_inner"], name)


def create_torch_embeddings(
    model_path_or_id: str,
    normalize: bool = True,
    device: Optional[str] = None,
    cache_folder: Optional[str] = None,
    offline_mode: bool = False,
    query_prefix: str = "",
    passage_prefix: str = ""
):
    """
    PyTorch 기반 HuggingFaceEmbeddings 생성

    Args:
        model_path_or_id: 모델 경로 또는 HuggingFace 모델 ID
        normalize: L2 정규화 여부
        device: 'cuda' 또는 'cpu' (None이면 자동 감지)
        cache_folder: 모델 캐시 폴더
        offline_mode: 오프라인 모드 여부
        query_prefix: 쿼리 임베딩 접두사 (E5 계열: "query: ")
        passage_prefix: 문서 임베딩 접두사 (E5 계열: "passage: ")

    Returns:
        HuggingFaceEmbeddings 인스턴스 (prefix 지정 시 래퍼로 감쌈)

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

        if query_prefix or passage_prefix:
            logger.info(
                f"E5 prefix 적용: query_prefix={query_prefix!r}, passage_prefix={passage_prefix!r}"
            )
            return PrefixEmbeddingWrapper(
                embeddings,
                query_prefix=query_prefix,
                passage_prefix=passage_prefix,
            )

        logger.info(f"Torch 임베딩 모델 로드 완료: {model_path_or_id}")
        return embeddings

    except (ModelLoadError, ModelOfflineError):
        raise
    except Exception as e:
        logger.error(f"Torch 백엔드 초기화 실패: {e}")
        raise ModelLoadError(model_path_or_id, str(e))
