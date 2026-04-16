# -*- coding: utf-8 -*-
from pathlib import Path
from uuid import uuid4


def test_revision_save_blocks_path_in_filename(tmp_path, monkeypatch):
    from app import create_app
    from app.services.search import qa_system
    from app.utils import FileInfo, FileUtils, get_app_directory

    target = tmp_path / "doc.txt"
    target.write_text("sample", encoding="utf-8")
    fid = FileUtils.make_file_id(str(target))

    monkeypatch.setattr(
        qa_system,
        "file_infos",
        {str(target): FileInfo(path=str(target), size=target.stat().st_size, chunks=1)},
        raising=False
    )
    monkeypatch.setenv("ADMIN_PASSWORD", "pw")

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    login = client.post("/api/admin/auth", json={"password": "pw"})
    assert login.status_code == 200
    assert login.get_json()["success"] is True

    resp = client.post(
        "/api/revisions",
        json={"file_id": fid, "filename": "../x", "content": "revision body", "comment": "test"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    rev_file = body["revision"]["file"]
    assert ".." not in rev_file
    assert "/" not in rev_file
    assert "\\" not in rev_file

    revisions_root = Path(get_app_directory()) / "revisions"
    saved = (revisions_root / rev_file).resolve()
    assert saved.exists()
    assert revisions_root.resolve() in saved.parents


def test_revision_version_continues_from_legacy_filename_key(tmp_path):
    from app.services.db import db
    from app.services.file_manager import RevisionTracker
    from app.utils import get_app_directory

    tracker = RevisionTracker()
    legacy_key = f"legacy-{uuid4().hex}.txt"
    file_id = f"fid-{uuid4().hex}"
    created_files = []

    try:
        first = tracker.save_revision(
            file_key=legacy_key,
            content="legacy body",
            note="legacy",
            display_name="doc.txt",
        )
        created_files.append(first["file"])

        second = tracker.save_revision(
            file_key=file_id,
            content="migrated body",
            note="migrated",
            display_name="doc.txt",
            legacy_key=legacy_key,
        )
        created_files.append(second["file"])

        history = tracker.get_history(file_id, legacy_key=legacy_key)

        assert first["version"] == "v1"
        assert second["version"] == "v2"
        assert {item["version"] for item in history} == {"v1", "v2"}
        assert tracker.get_revision(file_id, "v1", legacy_key=legacy_key) == "legacy body"
        assert tracker.get_revision(file_id, "v2", legacy_key=legacy_key) == "migrated body"
    finally:
        db.execute("DELETE FROM revisions WHERE filename=?", (legacy_key,))
        db.execute("DELETE FROM revisions WHERE filename=?", (file_id,))
        revisions_root = Path(get_app_directory()) / "revisions"
        for filename in created_files:
            path = (revisions_root / filename).resolve()
            if path.exists():
                path.unlink()
