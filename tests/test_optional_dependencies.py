from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_utf8_guard_script_passes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "check_utf8.py")],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_document_extractor_marks_ocr_unavailable_when_optional_deps_missing(monkeypatch) -> None:
    from app.services import document as document_module

    extractor = document_module.DocumentExtractor()
    real_import_module = document_module.importlib.import_module
    attempts: list[str] = []

    def fake_import_module(name: str, package: str | None = None):
        attempts.append(name)
        if name in {"pytesseract", "PIL.Image"}:
            raise ImportError(f"missing optional dependency: {name}")
        return real_import_module(name, package)

    monkeypatch.setattr(document_module.importlib, "import_module", fake_import_module)

    assert extractor.ocr_available is False
    assert extractor.ocr_available is False
    assert attempts.count("pytesseract") == 1
