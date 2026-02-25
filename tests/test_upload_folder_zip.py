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


def _make_large_zip_payload(entry_count=1, file_size=32) -> io.BytesIO:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as zf:
        content = "x" * file_size
        for i in range(entry_count):
            zf.writestr(f"f{i}.txt", content)
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


def test_upload_folder_zip_blocks_max_entries(tmp_path, monkeypatch):
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
        raising=False,
    )

    client = app.test_client()
    login = client.post("/api/admin/auth", json={"password": "pw"})
    assert login.status_code == 200

    resp = client.post(
        "/api/upload/folder",
        data={
            "file": (_make_large_zip_payload(entry_count=3, file_size=8), "too_many.zip"),
            "max_entries": "2",
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 413
    body = resp.get_json()
    assert body["success"] is False
    assert body["reason"] == "max_entries_exceeded"


def test_upload_folder_zip_blocks_size_limits(tmp_path, monkeypatch):
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
        raising=False,
    )

    client = app.test_client()
    login = client.post("/api/admin/auth", json={"password": "pw"})
    assert login.status_code == 200

    resp_single = client.post(
        "/api/upload/folder",
        data={
            "file": (_make_large_zip_payload(entry_count=1, file_size=64), "single_size.zip"),
            "max_single_file_bytes": "32",
        },
        content_type="multipart/form-data",
    )
    assert resp_single.status_code == 413
    assert resp_single.get_json()["reason"] == "max_single_file_bytes_exceeded"

    resp_total = client.post(
        "/api/upload/folder",
        data={
            "file": (_make_large_zip_payload(entry_count=3, file_size=16), "total_size.zip"),
            "max_uncompressed_bytes": "20",
        },
        content_type="multipart/form-data",
    )
    assert resp_total.status_code == 413
    assert resp_total.get_json()["reason"] == "max_uncompressed_bytes_exceeded"
