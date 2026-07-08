# -*- coding: utf-8 -*-
"""Kordoc(Node.js) 기반 문서 파서 브리지."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from app.utils import logger

from .hwp_models import (
    ExtractedDocument,
    build_basic_metadata,
    build_diagnostics,
    is_usable_text,
    normalize_text,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BRIDGE_SCRIPT = _REPO_ROOT / "scripts" / "kordoc_bridge.mjs"
_KORDOC_SUFFIXES = {".hwp", ".hwpx", ".docx", ".pdf"}


class KordocAdapter:
    """Node kordoc_bridge.mjs subprocess 어댑터."""

    @staticmethod
    def is_available() -> bool:
        if not shutil.which("node"):
            return False
        if not _BRIDGE_SCRIPT.is_file():
            return False
        node_modules = _REPO_ROOT / "node_modules" / "kordoc"
        return node_modules.is_dir()

    @staticmethod
    def supports(path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in _KORDOC_SUFFIXES

    def extract(self, path: str) -> ExtractedDocument:
        source_format = os.path.splitext(path)[1].lower().lstrip(".") or "unknown"
        metadata = build_basic_metadata(path, source_format)
        warnings: list[str] = []

        if not self.is_available():
            error = "Kordoc 미설치 (npm install 필요)"
            warnings.append(error)
            return ExtractedDocument(
                text="",
                metadata=metadata,
                tables=[],
                diagnostics=build_diagnostics("kordoc-unavailable", text="", fallback_used=False, warnings=warnings),
                error=error,
            )

        try:
            proc = subprocess.run(
                ["node", str(_BRIDGE_SCRIPT), path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
                cwd=str(_REPO_ROOT),
            )
        except subprocess.TimeoutExpired:
            return self._error_result(path, metadata, "Kordoc 파싱 시간 초과", warnings)
        except Exception as exc:
            return self._error_result(path, metadata, f"Kordoc 실행 오류: {exc}", warnings)

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()[:500]
            return self._error_result(path, metadata, f"Kordoc 파싱 실패: {detail}", warnings)

        try:
            payload = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError as exc:
            return self._error_result(path, metadata, f"Kordoc JSON 파싱 실패: {exc}", warnings)

        text = normalize_text(str(payload.get("markdown") or ""))
        if not is_usable_text(text):
            return self._error_result(path, metadata, "Kordoc 추출 텍스트 없음", warnings)

        raw_meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        if raw_meta.get("title"):
            metadata["title"] = raw_meta["title"]
        metadata["parser"] = "kordoc"
        metadata["page_count"] = len(payload.get("pageMap") or [])

        kordoc_warnings = [str(w) for w in (payload.get("warnings") or []) if str(w).strip()]
        warnings.extend(kordoc_warnings)

        tables = self._tables_from_blocks(payload.get("blocks"))
        if tables:
            metadata["table_count"] = len(tables)

        return ExtractedDocument(
            text=text,
            metadata=metadata,
            tables=tables,
            diagnostics=build_diagnostics("kordoc", text=text, fallback_used=False, warnings=warnings),
            error=None,
        )

    def _tables_from_blocks(self, blocks: Any) -> list[dict[str, Any]]:
        if not isinstance(blocks, list):
            return []
        tables: list[dict[str, Any]] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if str(block.get("type", "")).lower() != "table":
                continue
            summary = normalize_text(str(block.get("text") or block.get("content") or ""))
            if summary:
                tables.append({"summary": summary, "source": "kordoc"})
        return tables

    def _error_result(
        self,
        path: str,
        metadata: dict[str, Any],
        error: str,
        warnings: list[str],
    ) -> ExtractedDocument:
        warnings.append(error)
        logger.debug("Kordoc 추출 실패: %s - %s", path, error)
        return ExtractedDocument(
            text="",
            metadata=metadata,
            tables=[],
            diagnostics=build_diagnostics("kordoc", text="", fallback_used=False, warnings=warnings),
            error=error,
        )