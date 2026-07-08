# -*- coding: utf-8 -*-
"""MCP 서버 백그라운드 실행."""

from __future__ import annotations

import threading

from app.services.settings_store import get_settings_store
from app.utils import logger


_mcp_thread: threading.Thread | None = None


def start_mcp_server_thread() -> bool:
    """settings.json mcp.enabled=true 일 때 별도 스레드에서 MCP 시작."""
    global _mcp_thread
    if _mcp_thread and _mcp_thread.is_alive():
        return True

    settings = get_settings_store().load()
    mcp_cfg = settings.get("mcp") if isinstance(settings.get("mcp"), dict) else {}
    if not mcp_cfg.get("enabled", False):
        return False

    host = str(mcp_cfg.get("host") or "127.0.0.1")
    port = int(mcp_cfg.get("port") or 8081)
    transport = str(mcp_cfg.get("transport") or "sse").strip().lower()

    def _run() -> None:
        try:
            from app.mcp.builder import build_mcp_server

            mcp = build_mcp_server(host=host, port=port)
            logger.info("MCP 서버 시작: %s:%s transport=%s", host, port, transport)
            mcp.run(transport=transport)
        except Exception as exc:
            logger.warning("MCP 서버 실행 실패: %s", exc)

    _mcp_thread = threading.Thread(target=_run, name="mcp-server", daemon=True)
    _mcp_thread.start()
    return True