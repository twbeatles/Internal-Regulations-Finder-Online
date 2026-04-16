# -*- coding: utf-8 -*-
import io


def test_status_progress_alias():
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    resp = client.get("/api/status")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert "load_progress" in body
    assert "progress" in body
    assert body["progress"] == body["load_progress"]


def test_health_success_envelope():
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert "status" in body


def test_sync_status_alias_shape():
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    resp = client.get("/api/sync/status")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert "is_syncing" in body
    assert "status" in body
    assert isinstance(body["status"], dict)
    assert "running" in body["status"]
    assert "progress" in body["status"]


def test_revisions_alias_history_and_revisions():
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    resp = client.get("/api/revisions?filename=does-not-exist.txt")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert "history" in body
    assert "revisions" in body
    assert body["history"] == body["revisions"]


def test_search_history_schema_with_legacy_alias(monkeypatch):
    from app import create_app
    from app.services.search import qa_system

    class _DummyHistory:
        def get_recent(self, limit=10):
            return ["휴가", "복무"]

        def get_popular(self, limit=10):
            return [
                {"query": "휴가", "count": 3},
                {"query": "복무", "count": 2},
            ]

    monkeypatch.setattr(qa_system, "_search_history", _DummyHistory(), raising=False)

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    resp = client.get("/api/search/history?limit=5")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["recent"] == ["휴가", "복무"]
    assert body["popular"] == [{"query": "휴가", "count": 3}, {"query": "복무", "count": 2}]
    assert body["popular_legacy"] == [["휴가", 3], ["복무", 2]]


def test_search_suggest_schema():
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    resp = client.get("/api/search/suggest?q=휴")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert "suggestions" in body


def test_search_history_not_recorded_on_failed_search(monkeypatch):
    from app import create_app
    from app.services.search import qa_system, rate_limiter, search_queue
    from app.utils import TaskResult

    class _DummyHistory:
        def __init__(self):
            self.calls = []

        def add(self, query):
            self.calls.append(query)

    history = _DummyHistory()
    monkeypatch.setattr(qa_system, "_search_history", history, raising=False)
    monkeypatch.setattr(qa_system, "search", lambda *args, **kwargs: TaskResult(False, "failed"), raising=False)
    monkeypatch.setattr(rate_limiter, "is_allowed", lambda ip: True, raising=False)
    monkeypatch.setattr(search_queue, "acquire", lambda timeout=5: True, raising=False)
    monkeypatch.setattr(search_queue, "release", lambda: None, raising=False)

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    resp = client.post("/api/search", json={"query": "policy"})
    assert resp.status_code == 200
    assert resp.get_json()["success"] is False
    assert history.calls == []


def test_search_history_recorded_on_success(monkeypatch):
    from app import create_app
    from app.services.search import qa_system, rate_limiter, search_queue
    from app.utils import TaskResult

    class _DummyHistory:
        def __init__(self):
            self.calls = []

        def add(self, query):
            self.calls.append(query)

    history = _DummyHistory()
    monkeypatch.setattr(qa_system, "_search_history", history, raising=False)
    monkeypatch.setattr(
        qa_system,
        "search",
        lambda *args, **kwargs: TaskResult(True, "ok", [{"content": "policy guide", "source": "doc.txt", "file_id": "id-1", "score": 1.0}]),
        raising=False,
    )
    monkeypatch.setattr(rate_limiter, "is_allowed", lambda ip: True, raising=False)
    monkeypatch.setattr(search_queue, "acquire", lambda timeout=5: True, raising=False)
    monkeypatch.setattr(search_queue, "release", lambda: None, raising=False)

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    resp = client.post("/api/search", json={"query": "policy"})
    assert resp.status_code == 200
    assert resp.get_json()["success"] is True
    assert history.calls == ["policy"]
