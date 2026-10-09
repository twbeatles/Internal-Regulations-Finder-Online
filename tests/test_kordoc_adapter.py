# -*- coding: utf-8 -*-
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services.parsers.kordoc_adapter import KordocAdapter


def test_kordoc_unavailable_returns_error() -> None:
    with patch.object(KordocAdapter, "is_available", return_value=False):
        doc = KordocAdapter().extract("sample.hwp")
    assert doc.error
    assert not doc.text


def test_kordoc_success_parses_json() -> None:
    payload = '{"markdown":"# 제1조\\n내용","metadata":{"title":"테스트"},"warnings":[]}'
    with patch.object(KordocAdapter, "is_available", return_value=True):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=payload, stderr="")
            doc = KordocAdapter().extract(str(Path("test.hwp")))
    assert "제1조" in doc.text
    assert doc.metadata.get("parser") == "kordoc"