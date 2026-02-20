# -*- coding: utf-8 -*-
from pathlib import Path


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
