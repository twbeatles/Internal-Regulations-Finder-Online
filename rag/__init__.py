# -*- coding: utf-8 -*-
"""LLM RAG v3 패키지."""


def create_rag_blueprint():
    from rag.routes import create_rag_blueprint as _create

    return _create()


__all__ = ["create_rag_blueprint"]