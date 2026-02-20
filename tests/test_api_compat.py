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
    assert "load_progress" in body
    assert "progress" in body
    assert body["progress"] == body["load_progress"]


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
