# -*- coding: utf-8 -*-
from rag.routes.api_chat import rag_bp

__all__ = ["rag_bp", "create_rag_blueprint"]


def create_rag_blueprint():
    return rag_bp