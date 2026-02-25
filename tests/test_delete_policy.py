# -*- coding: utf-8 -*-


def _login_admin(client):
    resp = client.post("/api/admin/auth", json={"password": "pw"})
    assert resp.status_code == 200
    assert resp.get_json()["success"] is True


def test_delete_file_default_index_only_keeps_source(tmp_path, monkeypatch):
    from app import create_app
    from app.services.search import qa_system
    from app.utils import FileInfo, FileUtils

    target = tmp_path / "doc.txt"
    target.write_text("sample", encoding="utf-8")
    fid = FileUtils.make_file_id(str(target))

    monkeypatch.setenv("ADMIN_PASSWORD", "pw")
    monkeypatch.setattr(qa_system, "current_folder", str(tmp_path), raising=False)
    monkeypatch.setattr(
        qa_system,
        "file_infos",
        {str(target): FileInfo(path=str(target), size=target.stat().st_size, chunks=1)},
        raising=False,
    )
    monkeypatch.setattr(qa_system, "documents", ["sample"], raising=False)
    monkeypatch.setattr(
        qa_system,
        "doc_meta",
        [{"doc_id": 0, "source": "doc.txt", "path": str(target), "file_id": fid}],
        raising=False,
    )

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    _login_admin(client)

    resp = client.delete(f"/api/files/by-id/{fid}")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["deletion_policy"] == "index_only"
    assert body["deleted_source"] is False
    assert body["deleted_from_index"] is True
    assert target.exists() is True


def test_delete_file_with_delete_source_removes_file(tmp_path, monkeypatch):
    from app import create_app
    from app.services.search import qa_system
    from app.utils import FileInfo, FileUtils

    target = tmp_path / "doc.txt"
    target.write_text("sample", encoding="utf-8")
    fid = FileUtils.make_file_id(str(target))

    monkeypatch.setenv("ADMIN_PASSWORD", "pw")
    monkeypatch.setattr(qa_system, "current_folder", str(tmp_path), raising=False)
    monkeypatch.setattr(
        qa_system,
        "file_infos",
        {str(target): FileInfo(path=str(target), size=target.stat().st_size, chunks=1)},
        raising=False,
    )
    monkeypatch.setattr(qa_system, "documents", ["sample"], raising=False)
    monkeypatch.setattr(
        qa_system,
        "doc_meta",
        [{"doc_id": 0, "source": "doc.txt", "path": str(target), "file_id": fid}],
        raising=False,
    )

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    _login_admin(client)

    resp = client.delete(f"/api/files/by-id/{fid}?delete_source=true")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["deletion_policy"] == "delete_source"
    assert body["deleted_source"] is True
    assert body["deleted_from_index"] is True
    assert target.exists() is False


def test_delete_all_default_index_only_keeps_sources(tmp_path, monkeypatch):
    from app import create_app
    from app.services.search import qa_system
    from app.utils import FileInfo

    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("a", encoding="utf-8")
    f2.write_text("b", encoding="utf-8")

    monkeypatch.setenv("ADMIN_PASSWORD", "pw")
    monkeypatch.setattr(qa_system, "current_folder", str(tmp_path), raising=False)
    monkeypatch.setattr(
        qa_system,
        "file_infos",
        {
            str(f1): FileInfo(path=str(f1), size=f1.stat().st_size, chunks=1),
            str(f2): FileInfo(path=str(f2), size=f2.stat().st_size, chunks=1),
        },
        raising=False,
    )
    monkeypatch.setattr(qa_system, "documents", ["a", "b"], raising=False)
    monkeypatch.setattr(
        qa_system,
        "doc_meta",
        [
            {"doc_id": 0, "source": "a.txt", "path": str(f1)},
            {"doc_id": 1, "source": "b.txt", "path": str(f2)},
        ],
        raising=False,
    )

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    _login_admin(client)

    resp = client.delete("/api/files/all")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["deletion_policy"] == "index_only"
    assert body["deleted_source"] == 0
    assert body["deleted_from_index"] == 2
    assert f1.exists() is True
    assert f2.exists() is True
