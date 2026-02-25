# -*- coding: utf-8 -*-


def _login_admin(client):
    resp = client.post("/api/admin/auth", json={"password": "pw"})
    assert resp.status_code == 200
    assert resp.get_json()["success"] is True


def test_set_model_triggers_reindex_by_default(tmp_path, monkeypatch):
    from app import create_app
    from app.config import AppConfig
    from app.services.search import qa_system
    from app.utils import TaskResult

    monkeypatch.setenv("ADMIN_PASSWORD", "pw")
    monkeypatch.setattr(qa_system, "current_folder", str(tmp_path), raising=False)
    monkeypatch.setattr(qa_system, "offline_mode", False, raising=False)
    monkeypatch.setattr(qa_system, "local_model_path", "", raising=False)
    monkeypatch.setattr(
        qa_system,
        "load_model",
        lambda model_name, offline_mode=False, local_model_path="": TaskResult(True, "ok"),
        raising=False,
    )

    called = {}

    def _fake_initialize(folder_path, force_reindex=False):
        called["folder_path"] = folder_path
        called["force_reindex"] = force_reindex
        return TaskResult(True, "reindex started")

    monkeypatch.setattr(qa_system, "initialize", _fake_initialize, raising=False)

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    _login_admin(client)

    resp = client.post("/api/models", json={"model": AppConfig.DEFAULT_MODEL})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["reindex_requested"] is True
    assert body["reindex_triggered"] is True
    assert body["reindex_started"] is True
    assert called["folder_path"] == str(tmp_path)
    assert called["force_reindex"] is True


def test_set_model_without_folder_does_not_trigger_reindex(monkeypatch):
    from app import create_app
    from app.config import AppConfig
    from app.services.search import qa_system
    from app.utils import TaskResult

    monkeypatch.setenv("ADMIN_PASSWORD", "pw")
    monkeypatch.setattr(qa_system, "current_folder", "", raising=False)
    monkeypatch.setattr(qa_system, "offline_mode", False, raising=False)
    monkeypatch.setattr(qa_system, "local_model_path", "", raising=False)
    monkeypatch.setattr(
        qa_system,
        "load_model",
        lambda model_name, offline_mode=False, local_model_path="": TaskResult(True, "ok"),
        raising=False,
    )

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    _login_admin(client)

    resp = client.post("/api/models", json={"model": AppConfig.DEFAULT_MODEL})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["reindex_requested"] is True
    assert body["reindex_triggered"] is False


def test_sync_stop_requests_cancel_event(monkeypatch):
    from app import create_app
    from app.services.search import qa_system

    monkeypatch.setenv("ADMIN_PASSWORD", "pw")
    qa_system._is_loading = True
    qa_system._cancel_event.clear()

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    _login_admin(client)

    resp = client.post("/api/sync/stop")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert qa_system._cancel_event.is_set() is True

    qa_system._is_loading = False
    qa_system._cancel_event.clear()
