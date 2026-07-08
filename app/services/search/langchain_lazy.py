# -*- coding: utf-8 -*-
"""LangChain 지연 로딩."""
import importlib
from typing import Any

# 이 변수들은 실제 사용 시점에 _lazy_import_*() 함수로 로드됨
CharacterTextSplitter: Any | None = None
Document: Any | None = None
HuggingFaceEmbeddings: Any | None = None
FAISS: Any | None = None
_lazy_imports_loaded = False


def _import_attr(module_name: str, attr_name: str) -> Any | None:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    return getattr(module, attr_name, None)

def _lazy_import_langchain():
    """LangChain 관련 라이브러리 지연 로드"""
    global CharacterTextSplitter, Document, HuggingFaceEmbeddings, FAISS, _lazy_imports_loaded
    
    if _lazy_imports_loaded:
        return
    
    CharacterTextSplitter = _import_attr('langchain_text_splitters', 'CharacterTextSplitter')
    if CharacterTextSplitter is None:
        CharacterTextSplitter = _import_attr('langchain.text_splitter', 'CharacterTextSplitter')

    Document = _import_attr('langchain_core.documents', 'Document')
    if Document is None:
        Document = _import_attr('langchain.docstore.document', 'Document')

    HuggingFaceEmbeddings = _import_attr('langchain_huggingface', 'HuggingFaceEmbeddings')
    if HuggingFaceEmbeddings is None:
        HuggingFaceEmbeddings = _import_attr('langchain_community.embeddings', 'HuggingFaceEmbeddings')
    if HuggingFaceEmbeddings is None:
        HuggingFaceEmbeddings = _import_attr('langchain.embeddings', 'HuggingFaceEmbeddings')

    FAISS = _import_attr('langchain_community.vectorstores', 'FAISS')
    if FAISS is None:
        FAISS = _import_attr('langchain.vectorstores', 'FAISS')
    
    _lazy_imports_loaded = True
