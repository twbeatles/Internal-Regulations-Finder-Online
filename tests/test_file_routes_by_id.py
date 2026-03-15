from __future__ import annotations


def test_preview_route_resolves_file_by_id(tmp_path, monkeypatch) -> None:
    from app import create_app
    from app.services.search import qa_system
    from app.utils import FileInfo, FileUtils

    target = tmp_path / "preview.txt"
    target.write_text("preview body", encoding="utf-8")
    file_id = FileUtils.make_file_id(str(target))

    monkeypatch.setattr(
        qa_system,
        "file_infos",
        {str(target): FileInfo(path=str(target), size=target.stat().st_size, chunks=1)},
        raising=False,
    )
    monkeypatch.setattr(qa_system.extractor, "extract", lambda path: ("preview body", None), raising=False)

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get(f"/api/files/by-id/{file_id}/preview")

    assert response.status_code == 200
    body = response.get_json()
    assert body["success"] is True
    assert body["file_id"] == file_id
    assert body["filename"] == target.name
    assert "preview body" in body["content"]
