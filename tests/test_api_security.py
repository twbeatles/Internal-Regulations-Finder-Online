# -*- coding: utf-8 -*-
import io


def test_upload_sanitizes_filename(tmp_path, monkeypatch):
    from app import create_app
    from app.services.search import qa_system
    from app.utils import TaskResult

    app = create_app()
    app.config["TESTING"] = True

    monkeypatch.setattr(qa_system, "current_folder", str(tmp_path), raising=False)
    monkeypatch.setattr(qa_system, "process_single_file", lambda p: TaskResult(True, "ok", data={"path": p}))
    monkeypatch.setenv("ADMIN_PASSWORD", "pw")

    client = app.test_client()
    login = client.post("/api/admin/auth", json={"password": "pw"})
    assert login.status_code == 200
    assert login.get_json()["success"] is True

    resp = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(b"hello"), "../evil.txt")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    saved = body["filename"]
    assert ".." not in saved
    assert (tmp_path / saved).exists()


def test_admin_required_for_sync_start(tmp_path):
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True

    client = app.test_client()
    resp = client.post("/api/sync/start", json={"folder": str(tmp_path)})
    assert resp.status_code == 401


def test_admin_login_allows_sync_start(tmp_path, monkeypatch):
    from app import create_app
    from app.services.search import qa_system
    from app.utils import TaskResult

    app = create_app()
    app.config["TESTING"] = True

    monkeypatch.setenv("ADMIN_PASSWORD", "pw")
    monkeypatch.setattr(qa_system, "initialize", lambda folder_path, force_reindex=False: TaskResult(True, "ok"))

    client = app.test_client()

    resp = client.post("/api/admin/auth", json={"password": "pw"})
    assert resp.status_code == 200
    assert resp.get_json()["success"] is True

    resp = client.post("/api/sync/start", json={"folder": str(tmp_path), "force": False})
    assert resp.status_code == 200
    assert resp.get_json()["success"] is True


def test_admin_login_fails_when_password_not_configured(monkeypatch):
    from app import create_app

    class _DummyStore:
        def load(self):
            return {}

    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD_HASH", raising=False)
    monkeypatch.setattr("app.routes.api_system.get_settings_store", lambda: _DummyStore())

    app = create_app()
    app.config["TESTING"] = True

    client = app.test_client()
    resp = client.post("/api/admin/auth", json={"password": "admin"})
    assert resp.status_code == 503
    assert resp.get_json()["success"] is False


def test_state_changing_endpoints_require_admin(tmp_path):
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    assert client.post("/api/cache/clear").status_code == 401
    assert client.post("/api/revisions", json={"filename": "a.txt", "content": "x"}).status_code == 401
    assert client.post("/api/tags/set", json={"filename": "a.txt", "tags": []}).status_code == 401
    assert client.post("/api/tags/auto", json={"filename": "a.txt"}).status_code == 401
    assert client.post(
        "/api/upload",
        data={"file": (io.BytesIO(b"hello"), "a.txt")},
        content_type="multipart/form-data",
    ).status_code == 401
