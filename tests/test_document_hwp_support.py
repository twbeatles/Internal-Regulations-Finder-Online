from __future__ import annotations

import zipfile
from pathlib import Path


def _write_hwpx(path: Path, *, section_name: str = "Contents/section0.xml") -> None:
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<hp:section xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph">
  <hp:p>
    <hp:run><hp:t>첫 번째 문단</hp:t></hp:run>
  </hp:p>
  <hp:p>
    <hp:run><hp:t>두 번째 문단</hp:t></hp:run>
  </hp:p>
  <hp:tbl>
    <hp:tr>
      <hp:tc><hp:p><hp:run><hp:t>A1</hp:t></hp:run></hp:p></hp:tc>
      <hp:tc><hp:p><hp:run><hp:t>B1</hp:t></hp:run></hp:p></hp:tc>
    </hp:tr>
    <hp:tr>
      <hp:tc><hp:p><hp:run><hp:t>A2</hp:t></hp:run></hp:p></hp:tc>
      <hp:tc><hp:p><hp:run><hp:t>B2</hp:t></hp:run></hp:p></hp:tc>
    </hp:tr>
  </hp:tbl>
</hp:section>
"""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(section_name, xml)


def test_extract_with_details_routes_hwp_and_hwpx(tmp_path, monkeypatch) -> None:
    from app.services.document import DocumentExtractor
    from app.services.parsers.hwp_models import ExtractedDocument

    hwp_path = tmp_path / "sample.hwp"
    hwp_path.write_bytes(b"dummy")
    hwpx_path = tmp_path / "sample.hwpx"
    _write_hwpx(hwpx_path)

    calls: list[str] = []

    def fake_hwp(self, path: str) -> ExtractedDocument:
        calls.append("hwp")
        return ExtractedDocument(text="hwp body")

    def fake_hwpx(self, path: str) -> ExtractedDocument:
        calls.append("hwpx")
        return ExtractedDocument(text="hwpx body")

    monkeypatch.setattr(DocumentExtractor, "_extract_hwp_document", fake_hwp)
    monkeypatch.setattr(DocumentExtractor, "_extract_hwpx_document", fake_hwpx)

    extractor = DocumentExtractor()

    assert extractor.extract_with_details(str(hwp_path)).text == "hwp body"
    assert extractor.extract_with_details(str(hwpx_path)).text == "hwpx body"
    assert calls == ["hwp", "hwpx"]


def test_hwp_missing_olefile_gracefully_returns_error(tmp_path, monkeypatch) -> None:
    from app.services import document as document_module

    target = tmp_path / "missing-ole.hwp"
    target.write_bytes(b"not-an-ole-file")

    real_import_module = document_module.importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "olefile":
            raise ImportError("missing optional dependency: olefile")
        return real_import_module(name, package)

    monkeypatch.setattr(document_module.importlib, "import_module", fake_import_module)

    extractor = document_module.DocumentExtractor()
    result = extractor.extract_with_details(str(target))

    assert result.text == ""
    assert result.error is not None
    assert "olefile" in result.error
    assert result.diagnostics["engine_used"] == "hwp-unavailable"
    assert result.diagnostics["fallback_used"] is True


def test_hwpx_parse_failure_is_captured(tmp_path) -> None:
    from app.services.document import DocumentExtractor

    target = tmp_path / "broken.hwpx"
    target.write_text("not-a-zip-container", encoding="utf-8")

    result = DocumentExtractor().extract_with_details(str(target))

    assert result.text == ""
    assert result.error is not None
    assert "HWPX ZIP 오류" in result.error


def test_hwpx_extracts_text_tables_and_metadata(tmp_path) -> None:
    from app.services.document import DocumentExtractor

    target = tmp_path / "normal.hwpx"
    _write_hwpx(target)

    result = DocumentExtractor().extract_with_details(str(target))

    assert result.error is None
    assert "첫 번째 문단" in result.text
    assert "두 번째 문단" in result.text
    assert "A1 | B1" in result.text
    assert result.metadata["source_format"] == "hwpx"
    assert result.metadata["table_count"] == 1
    assert len(result.tables) == 1
    assert result.table_dicts()[0]["text_rows"][0] == "A1 | B1"


def test_process_single_file_preserves_parser_details_in_chunk_meta(tmp_path, monkeypatch) -> None:
    from app.services.search import RegulationQASystem
    from app.services.parsers.hwp_models import ExtractedDocument, ExtractedTable, ExtractedTableCell

    class _DummySplitter:
        def __init__(self, chunk_size, chunk_overlap):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split(self, text):
            return ["chunk-1", "chunk-2"]

    target = tmp_path / "sample.hwpx"
    target.write_bytes(b"placeholder")

    extracted = ExtractedDocument(
        text="chunk-1\n\nchunk-2",
        metadata={"source_format": "hwpx", "title": "sample"},
        tables=[
            ExtractedTable(
                index=0,
                rows=1,
                cols=1,
                cells=[ExtractedTableCell(row=0, col=0, text="표 내용")],
            )
        ],
        diagnostics={
            "engine_used": "hwpx-xml",
            "fallback_used": False,
            "quality_score": 0.9,
            "warnings": [],
        },
        error=None,
    )

    qa = RegulationQASystem()
    try:
        monkeypatch.setattr("app.services.search.DocumentSplitter", _DummySplitter)
        monkeypatch.setattr(qa.extractor, "extract_with_details", lambda _: extracted, raising=False)

        result = qa.process_single_file(str(target))

        assert result.success is True
        assert len(qa.documents) == 2
        assert qa.file_details[str(target)]["metadata"]["source_format"] == "hwpx"
        assert qa.file_details[str(target)]["diagnostics"]["engine_used"] == "hwpx-xml"
        assert qa.doc_meta[0]["document_metadata"]["source_format"] == "hwpx"
        assert qa.doc_meta[0]["table_count"] == 1
        assert result.data["table_count"] == 1
    finally:
        qa.cleanup()


def test_preview_route_returns_parser_metadata_and_diagnostics(tmp_path, monkeypatch) -> None:
    from app import create_app
    from app.routes import api_files as api_files_module
    from app.services.parsers.hwp_models import ExtractedDocument
    from app.services.search import qa_system
    from app.utils import FileInfo, FileUtils

    target = tmp_path / "preview.hwpx"
    target.write_bytes(b"placeholder")
    file_id = FileUtils.make_file_id(str(target))

    monkeypatch.setattr(
        qa_system,
        "file_infos",
        {str(target): FileInfo(path=str(target), size=target.stat().st_size, chunks=1)},
        raising=False,
    )
    monkeypatch.setattr(
        api_files_module._preview_extractor,
        "extract_with_details",
        lambda path: ExtractedDocument(
            text="미리보기 본문",
            metadata={"source_format": "hwpx", "title": "preview"},
            tables=[],
            diagnostics={
                "engine_used": "hwpx-xml",
                "fallback_used": False,
                "quality_score": 0.8,
                "warnings": [],
            },
            error=None,
        ),
        raising=False,
    )

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get(f"/api/files/by-id/{file_id}/preview")

    assert response.status_code == 200
    body = response.get_json()
    assert body["success"] is True
    assert body["metadata"]["source_format"] == "hwpx"
    assert body["diagnostics"]["engine_used"] == "hwpx-xml"
    assert body["table_count"] == 0
