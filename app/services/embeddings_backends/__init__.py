# -*- coding: utf-8 -*-
"""
임베딩 백엔드 패키지

torch, onnx_fp32, onnx_int8 백엔드를 지원하는 임베딩 생성 시스템
"""

from app.services.embeddings_backends.factory import create_embeddings

__all__ = ['create_embeddings']
