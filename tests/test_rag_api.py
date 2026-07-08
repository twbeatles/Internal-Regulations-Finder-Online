# -*- coding: utf-8 -*-
"""RAG API 스모크 테스트."""


def _client():
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


class TestRagApi:
    def test_rag_status_endpoint(self):
        res = _client().get("/api/rag/status")
        assert res.status_code == 200
        data = res.get_json()
        assert data.get("success") is True
        assert "search_mode" in (data.get("data") or {})

    def test_search_mode_get(self):
        res = _client().get("/api/settings/search-mode")
        assert res.status_code == 200
        payload = res.get_json()
        mode = (payload.get("data") or {}).get("search_mode") or payload.get("search_mode")
        assert mode in ("rag", "legacy")

    def test_rag_chat_sync_validation(self):
        res = _client().post("/api/rag/chat/sync", json={"message": "a"})
        assert res.status_code == 400

    def test_rag_conversations_list(self):
        res = _client().get("/api/rag/conversations")
        assert res.status_code == 200
        data = res.get_json()
        assert data.get("success") is True