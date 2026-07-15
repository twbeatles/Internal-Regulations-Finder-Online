# -*- coding: utf-8 -*-
"""PROJECT_AUDIT 권고사항 회귀 테스트."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _client():
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _login(client, password: str = "pw"):
    resp = client.post("/api/admin/auth", json={"password": password})
    assert resp.status_code == 200
    assert resp.get_json()["success"] is True


def test_initialize_rejects_when_loading(monkeypatch):
    from app.constants import ErrorMessages
    from app.services.search import qa_system
    from app.utils import TaskResult

    monkeypatch.setattr(qa_system, "_is_loading", True, raising=False)
    result = qa_system.initialize("/tmp/does-not-matter-loading")
    assert result.success is False
    assert ErrorMessages.SYNC_ALREADY_RUNNING in result.message
    monkeypatch.setattr(qa_system, "_is_loading", False, raising=False)


def test_search_blocked_during_indexing(monkeypatch):
    from app.config import AppConfig
    from app.constants import ErrorMessages
    from app.services.search import qa_system
    from app.services.search.bm25 import BM25Light

    monkeypatch.setattr(AppConfig, "BLOCK_SEARCH_DURING_INDEXING", True, raising=False)
    monkeypatch.setattr(qa_system, "_is_loading", True, raising=False)
    monkeypatch.setattr(qa_system, "bm25", BM25Light(), raising=False)
    monkeypatch.setattr(qa_system, "documents", ["휴가 규정"], raising=False)

    res = qa_system.search("휴가 규정", k=3)
    assert res.success is False
    assert ErrorMessages.SEARCH_INDEXING_IN_PROGRESS in res.message

    monkeypatch.setattr(qa_system, "_is_loading", False, raising=False)


def test_search_query_max_length(monkeypatch):
    from app.config import AppConfig
    from app.constants import ErrorMessages
    from app.services.search import qa_system
    from app.services.search.bm25 import BM25Light

    monkeypatch.setattr(AppConfig, "MAX_SEARCH_QUERY_LENGTH", 10, raising=False)
    monkeypatch.setattr(AppConfig, "BLOCK_SEARCH_DURING_INDEXING", True, raising=False)
    monkeypatch.setattr(qa_system, "_is_loading", False, raising=False)
    monkeypatch.setattr(qa_system, "bm25", BM25Light(), raising=False)
    monkeypatch.setattr(qa_system, "vector_store", None, raising=False)

    res = qa_system.search("가" * 20, k=3)
    assert res.success is False
    assert ErrorMessages.SEARCH_QUERY_TOO_LONG in res.message


def test_rag_message_max_length(monkeypatch):
    from app.config import AppConfig

    monkeypatch.setattr(AppConfig, "MAX_RAG_MESSAGE_LENGTH", 20, raising=False)
    client = _client()
    res = client.post("/api/rag/chat/sync", json={"message": "가" * 50})
    assert res.status_code == 400
    body = res.get_json()
    assert body.get("success") is False


def test_admin_auth_rate_limited(monkeypatch):
    from app.routes import api_system as sys_mod

    monkeypatch.setenv("ADMIN_PASSWORD", "pw")
    # 테스트용 초저 한도
    sys_mod._admin_auth_limiter.limit = 3
    sys_mod._admin_auth_limiter.reset()

    client = _client()
    # 실패 로그인으로 한도 소진
    codes = []
    for _ in range(5):
        r = client.post("/api/admin/auth", json={"password": "wrong"})
        codes.append(r.status_code)
    assert 429 in codes
    # 다른 테스트 오염 방지
    sys_mod._admin_auth_limiter.limit = 20
    sys_mod._admin_auth_limiter.reset()


def test_mcp_reindex_requires_token(monkeypatch):
    from app.constants import ErrorMessages
    from app.mcp import tools as mcp_tools

    class _Store:
        def load(self):
            return {"folder": "/tmp", "mcp": {}}

    with patch("app.services.settings_store.get_settings_store", lambda: _Store()):
        out = mcp_tools.regulations_reindex(admin_token="")
    assert out["success"] is False
    assert ErrorMessages.MCP_ADMIN_TOKEN_REQUIRED in out["message"]


def test_mcp_reindex_token_mismatch(monkeypatch):
    from app.mcp import tools as mcp_tools

    class _Store:
        def load(self):
            return {"folder": "/tmp", "mcp": {"admin_token": "secret"}}

    with patch("app.services.settings_store.get_settings_store", lambda: _Store()):
        out = mcp_tools.regulations_reindex(admin_token="wrong")
    assert out["success"] is False
    assert "불일치" in out["message"]


def test_conversation_access_scoped_to_session(monkeypatch):
    from rag.store.conversations import ConversationStore

    store = ConversationStore()
    cid = store.create_conversation(title="t")
    store.add_message(cid, "user", "hello")

    client = _client()
    # 다른 세션 — 소유권 없음
    res = client.get(f"/api/rag/conversations/{cid}")
    assert res.status_code == 403

    # 목록에도 안 보임
    res_list = client.get("/api/rag/conversations")
    assert res_list.status_code == 200
    items = (res_list.get_json().get("data") or {}).get("conversations") or []
    assert all(c.get("id") != cid for c in items)


def test_validate_folder_path_rejects_traversal(tmp_path):
    from app.services.files.path_validation import validate_folder_path

    path, err = validate_folder_path(str(tmp_path / ".." / "nope" / ".." / "x"))
    # 패턴에 '..' 포함 시 거부
    assert path is None
    assert err is not None


def test_validate_folder_path_accepts_real_dir(tmp_path):
    from app.services.files.path_validation import validate_folder_path

    path, err = validate_folder_path(str(tmp_path))
    assert err is None
    assert path is not None
    assert Path(path).is_dir()


def test_zip_copy_enforces_actual_bytes(tmp_path, monkeypatch):
    """메타 file_size를 속인 ZIP도 실제 바이트로 차단."""
    from app import create_app
    from app.config import AppConfig
    from app.services.search import qa_system
    from app.utils import TaskResult

    # 작은 ZIP을 만들되 내부 텍스트는 크게 — ZipInfo.file_size는 실제 크기로 기록되므로
    # 테스트는 max_single_file_bytes를 아주 작게 설정해 실제 해제 시 차단을 검증한다.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("doc.txt", "x" * 5000)
    buf.seek(0)

    monkeypatch.setenv("ADMIN_PASSWORD", "pw")
    monkeypatch.setattr(qa_system, "current_folder", str(tmp_path), raising=False)
    monkeypatch.setattr(
        qa_system,
        "process_single_file",
        lambda p: TaskResult(True, "ok", data={}),
        raising=False,
    )

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    _login(client)

    resp = client.post(
        "/api/upload/folder",
        data={
            "file": (buf, "pack.zip"),
            "max_single_file_bytes": "100",
            "max_uncompressed_bytes": "100000",
            "max_entries": "10",
        },
        content_type="multipart/form-data",
    )
    # 사전 메타 검사 또는 실제 바이트 검사로 실패
    assert resp.status_code in (200, 413)
    body = resp.get_json()
    if resp.status_code == 200:
        # 성공 응답이면 실패 항목에 잡혔어야 함
        assert body.get("failed_count", 0) > 0 or body.get("success") is False


def test_stream_replace_event_on_refusal():
    from rag.pipeline.orchestrator import RAGPipeline
    from rag.schemas import RAGResult, RetrievedChunk

    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe.config = MagicMock()
    pipe.query_processor = MagicMock()
    pipe.query_processor.process.return_value = MagicMock(primary=lambda: "q", queries=["q"])
    pipe.retriever = MagicMock()
    chunk = RetrievedChunk(
        doc_id=0,
        content="연차는 15일입니다",
        source="규정.txt",
        file_id="f1",
        path="/x",
        score=1.0,
        article_no="1",
        article_title="연차",
    )
    pipe.retriever.retrieve.return_value = [chunk]
    pipe.reranker = MagicMock()
    pipe.reranker.rerank.return_value = [chunk]
    pipe.context_builder = MagicMock()
    pipe.context_builder.build.return_value = ("ctx", [chunk])
    pipe.generator = MagicMock()
    pipe.generator.stream.return_value = iter(["환각 ", "답변"])
    pipe.guardrails = MagicMock()
    pipe.guardrails._build_citations.return_value = []
    pipe.guardrails.apply.return_value = RAGResult(
        answer="해당 규정에서 확인할 수 없습니다.",
        citations=[],
        confidence=0.1,
        verification_score=0.0,
        refused=True,
        retrieval_only=False,
    )

    with patch("rag.pipeline.orchestrator.create_llm_provider", return_value=MagicMock()):
        with patch("rag.pipeline.orchestrator.is_provider_healthy", return_value=True):
            events = list(pipe.stream("질문입니다"))

    kinds = [e["event"] for e in events]
    assert "token" in kinds
    assert "replace" in kinds
    assert "done" in kinds
    replace = next(e for e in events if e["event"] == "replace")
    assert "확인할 수 없습니다" in replace["data"]["answer"]
