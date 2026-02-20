# -*- coding: utf-8 -*-
import io
import zipfile


def _make_zip_payload() -> io.BytesIO:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ok1.txt", "hello")
        zf.writestr("nested/ok2.docx", "world")
        zf.writestr("../evil.txt", "blocked")
        zf.writestr("skip.exe", "skip")
    payload.seek(0)
    return payload


def test_upload_folder_zip_success_and_validation(tmp_path, monkeypatch):
    from app import create_app
    from app.services.search import qa_system
    from app.utils import TaskResult

    app = create_app()
    app.config["TESTING"] = True

    monkeypatch.setenv("ADMIN_PASSWORD", "pw")
    monkeypatch.setattr(qa_system, "current_folder", str(tmp_path), raising=False)
    monkeypatch.setattr(
        qa_system,
        "process_single_file",
        lambda p: TaskResult(True, "ok", data={"path": p}),
        raising=False
    )

    client = app.test_client()
    login = client.post("/api/admin/auth", json={"password": "pw"})
    assert login.status_code == 200
    assert login.get_json()["success"] is True

    resp = client.post(
        "/api/upload/folder",
        data={"file": (_make_zip_payload(), "docs.zip")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["processed_count"] >= 2
    assert body["success_count"] >= 2
    assert body["failed_count"] >= 1  # ../evil.txt
    assert body["skipped_count"] >= 1  # skip.exe
