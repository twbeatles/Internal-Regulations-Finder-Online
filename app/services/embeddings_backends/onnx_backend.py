# -*- coding: utf-8 -*-
"""
ONNX Runtime 기반 임베딩 백엔드

LangChain Embeddings 인터페이스와 호환되는 ONNX 임베딩 클래스
HuggingFaceEmbeddings(normalize_embeddings=True)와 동일한 출력을 생성
"""

import os
import importlib
import threading
from typing import List, Optional, Any

import numpy as np

from app.utils import logger
from app.config import AppConfig
from app.exceptions import ModelLoadError, ModelOfflineError


class OnnxEmbeddings:
    """
    ONNX Runtime 기반 임베딩 생성 클래스
    
    LangChain Embeddings 인터페이스와 완전 호환:
    - embed_query(text) -> List[float]
    - embed_documents(texts) -> List[List[float]]
    
    특징:
    - Mean pooling + attention mask (HuggingFaceEmbeddings와 동일)
    - L2 정규화 지원
    - 스레드 안전 (double-checked locking)
    - CPUExecutionProvider + Graph optimization
    """
    
    def __init__(
        self,
        model_path: str,
        onnx_file: str = "model.onnx",
        normalize: bool = True,
        offline_mode: bool = False
    ):
        """
        Args:
            model_path: 모델 디렉토리 경로 (tokenizer + onnx 파일 위치)
            onnx_file: ONNX 모델 파일명 (model.onnx 또는 model.int8.onnx)
            normalize: L2 정규화 적용 여부
            offline_mode: 오프라인 모드 (토크나이저 로컬 로드)
            
        Raises:
            ModelLoadError: 모델 또는 토크나이저 로드 실패
            ModelOfflineError: 오프라인 모드에서 로컬 파일을 찾을 수 없음
        """
        self.model_path = model_path
        self.onnx_file = onnx_file
        self.normalize = normalize
        self.offline_mode = offline_mode
        
        self._session: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._session_lock = threading.Lock()
        self._tokenizer_lock = threading.Lock()
        
        # 입출력 텐서 이름 (세션 초기화 시 자동 탐지)
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._has_last_hidden_state: bool = True
        
        # ONNX 파일 경로 확인
        self._onnx_path = os.path.join(model_path, onnx_file)
        if not os.path.exists(self._onnx_path):
            raise ModelLoadError(model_path, f"ONNX 파일을 찾을 수 없습니다: {self._onnx_path}")
        
        logger.info(f"ONNX 백엔드 초기화: {self._onnx_path} (normalize: {normalize})")
    
    def _ensure_tokenizer(self):
        """토크나이저 지연 로드 (double-checked locking)"""
        if self._tokenizer is not None:
            return
        
        with self._tokenizer_lock:
            if self._tokenizer is not None:
                return
            
            try:
                from transformers import AutoTokenizer
                
                # 오프라인 모드 환경변수 설정
                if self.offline_mode:
                    os.environ['HF_HUB_OFFLINE'] = '1'
                    os.environ['TRANSFORMERS_OFFLINE'] = '1'
                
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    local_files_only=self.offline_mode
                )
                logger.debug(f"토크나이저 로드 완료: {self.model_path}")
                
            except Exception as e:
                logger.error(f"토크나이저 로드 실패: {e}")
                raise ModelLoadError(self.model_path, f"토크나이저 로드 실패: {e}")
    
    def _ensure_session(self):
        """ONNX 세션 지연 로드 (double-checked locking)"""
        if self._session is not None:
            return
        
        with self._session_lock:
            if self._session is not None:
                return
            
            try:
                ort = importlib.import_module("onnxruntime")
                if ort is None:
                    raise ImportError("onnxruntime")
                
                # 세션 옵션 설정
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                # 스레드 설정 (환경변수에서)
                intra_threads = int(os.environ.get('ONNX_INTRA_OP_THREADS', '0'))
                inter_threads = int(os.environ.get('ONNX_INTER_OP_THREADS', '0'))
                if intra_threads > 0:
                    sess_options.intra_op_num_threads = intra_threads
                if inter_threads > 0:
                    sess_options.inter_op_num_threads = inter_threads
                
                # 세션 생성 (CPU만 사용)
                session = ort.InferenceSession(
                    self._onnx_path,
                    sess_options,
                    providers=['CPUExecutionProvider']
                )
                self._session = session
                
                # 입출력 텐서 이름 자동 탐지
                self._input_names = [inp.name for inp in session.get_inputs()]
                self._output_names = [out.name for out in session.get_outputs()]
                
                # last_hidden_state 확인
                self._has_last_hidden_state = 'last_hidden_state' in self._output_names
                if not self._has_last_hidden_state:
                    if 'pooler_output' in self._output_names:
                        logger.warning("ONNX 모델에 last_hidden_state 없음, pooler_output 사용 (품질 저하 가능)")
                    else:
                        # 첫 번째 출력 사용
                        logger.warning(f"ONNX 출력 텐서 이름: {self._output_names}, 첫 번째 출력 사용")
                
                logger.info(f"ONNX 세션 초기화 완료: inputs={self._input_names}, outputs={self._output_names}")
                
            except ImportError:
                raise ModelLoadError(self.model_path, "onnxruntime 패키지가 설치되지 않았습니다. pip install onnxruntime")
            except Exception as e:
                logger.error(f"ONNX 세션 초기화 실패: {e}")
                raise ModelLoadError(self.model_path, f"ONNX 세션 초기화 실패: {e}")
    
    def _mean_pooling(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """
        Mean pooling with attention mask
        
        HuggingFaceEmbeddings와 동일한 pooling 방식:
        pooled = sum(token_embeddings * mask) / sum(mask)
        
        Args:
            token_embeddings: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len]
            
        Returns:
            [batch, hidden_size] pooled embeddings
        """
        # attention_mask 확장: [batch, seq_len] -> [batch, seq_len, 1]
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        input_mask_expanded = np.broadcast_to(
            input_mask_expanded, 
            token_embeddings.shape
        ).astype(np.float32)
        
        # weighted sum
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.sum(input_mask_expanded, axis=1)
        sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)  # 0 나누기 방지
        
        return sum_embeddings / sum_mask
    
    def _l2_normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 정규화"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return embeddings / norms
    
    def _encode(self, texts: List[str]) -> np.ndarray:
        """
        텍스트 리스트를 임베딩으로 변환
        
        Args:
            texts: 입력 텍스트 리스트
            
        Returns:
            [len(texts), hidden_size] numpy array
        """
        self._ensure_tokenizer()
        self._ensure_session()
        tokenizer = self._tokenizer
        session = self._session
        if tokenizer is None or session is None:
            raise ModelLoadError(self.model_path, "토크나이저 또는 ONNX 세션이 초기화되지 않았습니다")
        
        # 토큰화
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="np"
        )
        
        # ONNX 입력 준비
        onnx_inputs = {}
        for name in self._input_names:
            if name == 'input_ids':
                onnx_inputs[name] = inputs['input_ids'].astype(np.int64)
            elif name == 'attention_mask':
                onnx_inputs[name] = inputs['attention_mask'].astype(np.int64)
            elif name == 'token_type_ids' and 'token_type_ids' in inputs:
                onnx_inputs[name] = inputs['token_type_ids'].astype(np.int64)
        
        # ONNX 추론
        outputs = session.run(None, onnx_inputs)
        
        # 출력 텐서 선택
        if self._has_last_hidden_state:
            output_idx = self._output_names.index('last_hidden_state')
            token_embeddings = outputs[output_idx]
        elif 'pooler_output' in self._output_names:
            output_idx = self._output_names.index('pooler_output')
            # pooler_output은 이미 pooled 상태
            embeddings = outputs[output_idx]
            if self.normalize:
                embeddings = self._l2_normalize(embeddings)
            return embeddings
        else:
            # 첫 번째 출력 사용
            token_embeddings = outputs[0]
        
        # Mean pooling
        attention_mask = inputs['attention_mask'].astype(np.float32)
        embeddings = self._mean_pooling(token_embeddings, attention_mask)
        
        # L2 정규화
        if self.normalize:
            embeddings = self._l2_normalize(embeddings)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        단일 쿼리 텍스트 임베딩
        
        Args:
            text: 입력 텍스트
            
        Returns:
            임베딩 벡터 (List[float])
        """
        embeddings = self._encode([text])
        return embeddings[0].tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        여러 문서 텍스트 임베딩
        
        Args:
            texts: 입력 텍스트 리스트
            
        Returns:
            임베딩 벡터 리스트 (List[List[float]])
        """
        if not texts:
            return []
        
        embeddings = self._encode(texts)
        return embeddings.tolist()


def create_onnx_embeddings(
    model_path: str,
    onnx_file: str = "model.onnx",
    normalize: bool = True,
    offline_mode: bool = False
) -> OnnxEmbeddings:
    """
    ONNX 임베딩 객체 생성 헬퍼 함수
    
    Args:
        model_path: 모델 디렉토리 경로
        onnx_file: ONNX 파일명
        normalize: L2 정규화 여부
        offline_mode: 오프라인 모드
        
    Returns:
        OnnxEmbeddings 인스턴스
        
    Raises:
        ModelLoadError: 모델 로드 실패
    """
    return OnnxEmbeddings(
        model_path=model_path,
        onnx_file=onnx_file,
        normalize=normalize,
        offline_mode=offline_mode
    )
