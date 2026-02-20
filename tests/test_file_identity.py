# -*- coding: utf-8 -*-
from pathlib import Path


def test_make_file_id_stable():
    from app.utils import FileUtils

    path = r"C:\temp\example\doc.txt"
    v1 = FileUtils.make_file_id(path)
    v2 = FileUtils.make_file_id(path)
    assert v1 == v2
    assert len(v1) == 24


def test_files_names_include_file_id(tmp_path, monkeypatch):
    from app import create_app
    from app.services.search import qa_system
    from app.utils import FileInfo

    file1 = tmp_path / "A" / "same.txt"
    file2 = tmp_path / "B" / "same.txt"
    file1.parent.mkdir(parents=True, exist_ok=True)
    file2.parent.mkdir(parents=True, exist_ok=True)
    file1.write_text("hello", encoding="utf-8")
    file2.write_text("world", encoding="utf-8")

    infos = {
        str(file1): FileInfo(path=str(file1), size=file1.stat().st_size, chunks=1),
        str(file2): FileInfo(path=str(file2), size=file2.stat().st_size, chunks=1),
    }
    monkeypatch.setattr(qa_system, "file_infos", infos, raising=False)

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    resp = client.get("/api/files/names")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert "files" in body
    assert len(body["files"]) == 2
    ids = [f["file_id"] for f in body["files"]]
    assert len(set(ids)) == 2


def test_sanitize_upload_filename_special_chars():
    from app.utils import FileUtils

    assert FileUtils.sanitize_upload_filename("a?b.txt") == "a_b.txt"
    assert FileUtils.sanitize_upload_filename("a:b.txt") == "a_b.txt"
    assert FileUtils.sanitize_upload_filename("name    with   spaces.txt") == "name with spaces.txt"
    assert FileUtils.sanitize_upload_filename("CON.txt").startswith("_CON")
