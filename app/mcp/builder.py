# -*- coding: utf-8 -*-
"""FastMCP 서버 빌드."""

from __future__ import annotations

from typing import Any

from app.mcp import tools as mcp_tools


def build_mcp_server(host: str = "127.0.0.1", port: int = 8081):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError("mcp 패키지가 필요합니다 (pip install mcp)") from exc

    mcp = FastMCP("internal-regulations", host=host, port=port)

    @mcp.tool(name="regulations.search")
    def regulations_search(q: str, k: int = 5, hybrid: bool = True) -> dict[str, Any]:
        """사내 규정 하이브리드 검색."""
        return mcp_tools.regulations_search(q, k=k, hybrid=hybrid)

    @mcp.tool(name="regulations.rag_chat")
    def regulations_rag_chat(message: str, filter_file_id: str | None = None) -> dict[str, Any]:
        """RAG 질의응답 (근거 인용 포함)."""
        return mcp_tools.regulations_rag_chat(message, filter_file_id=filter_file_id)

    @mcp.tool(name="regulations.list_files")
    def regulations_list_files() -> dict[str, Any]:
        """인덱스된 규정 파일 목록."""
        return mcp_tools.regulations_list_files()

    @mcp.tool(name="regulations.status")
    def regulations_status() -> dict[str, Any]:
        """검색/RAG 준비 상태."""
        return mcp_tools.regulations_status()

    @mcp.tool(name="regulations.get_article")
    def regulations_get_article(file_id: str, article_query: str) -> dict[str, Any]:
        """특정 파일에서 조문 검색."""
        return mcp_tools.regulations_get_article(file_id, article_query)

    @mcp.tool(name="regulations.reindex")
    def regulations_reindex(admin_token: str = "") -> dict[str, Any]:
        """폴더 재인덱스 (settings mcp.admin_token 필수)."""
        return mcp_tools.regulations_reindex(admin_token=admin_token)

    return mcp