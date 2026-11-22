# -*- coding: utf-8 -*-
"""RAG HTTP 라우트.

api_chat은 app.* 모듈을 임포트하므로 패키지 임포트 시점이 아니라
create_rag_blueprint() 호출 시점에 지연 로딩한다. 그렇지 않으면 rag를
먼저 임포트하는 진입점(PyInstaller 분석 등)에서 app/__init__의
create_app()와 상호 재진입하여 partially initialized ImportError가 난다.
"""

__all__ = ["rag_bp", "create_rag_blueprint"]


def create_rag_blueprint():
    from rag.routes.api_chat import rag_bp

    return rag_bp


def __getattr__(name):
    # 하위호환: from rag.routes import rag_bp 형태도 지연 로딩으로 지원
    if name == "rag_bp":
        from rag.routes.api_chat import rag_bp

        return rag_bp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")