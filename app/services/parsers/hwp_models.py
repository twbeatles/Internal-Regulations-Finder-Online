from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MULTISPACE_RE = re.compile(r"[ \t\f\v]+")
_MULTI_BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    value = str(text or "")
    if not value:
        return ""
    value = value.replace("\ufeff", "").replace("\x00", "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = _CONTROL_CHARS_RE.sub("", value)
    lines = []
    for raw_line in value.split("\n"):
        line = _MULTISPACE_RE.sub(" ", raw_line).strip()
        lines.append(line)
    value = "\n".join(lines)
    value = _MULTI_BLANK_LINES_RE.sub("\n\n", value)
    return value.strip()


def is_usable_text(text: str, min_content_chars: int = 5) -> bool:
    value = normalize_text(text)
    if not value:
        return False
    printable = sum(1 for ch in value if ch.isprintable() or ch in "\n\t")
    content_chars = sum(1 for ch in value if ch.isalnum() or ("\uac00" <= ch <= "\ud7a3"))
    return printable / max(len(value), 1) >= 0.85 and content_chars >= min_content_chars


def compute_quality_score(text: str, warnings: list[str] | None = None) -> float:
    normalized = normalize_text(text)
    if not normalized:
        return 0.0
    warning_count = len(warnings or [])
    base = min(1.0, 0.25 + (len(normalized) / 4000.0))
    penalty = min(0.5, warning_count * 0.05)
    return round(max(0.0, base - penalty), 3)


def build_basic_metadata(path: str, source_format: str, **extra: Any) -> dict[str, Any]:
    filename = os.path.basename(path or "")
    title, _ = os.path.splitext(filename)
    metadata = {
        "title": title or filename,
        "file_name": filename,
        "source_path": path,
        "source_format": source_format,
    }
    metadata.update(extra)
    return metadata


def build_diagnostics(
    engine_used: str,
    *,
    text: str = "",
    fallback_used: bool = False,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    warning_items = [str(item) for item in (warnings or []) if str(item).strip()]
    return {
        "engine_used": str(engine_used or ""),
        "fallback_used": bool(fallback_used),
        "quality_score": compute_quality_score(text, warning_items),
        "warnings": warning_items,
    }


@dataclass
class ExtractedTableCell:
    row: int
    col: int
    text: str = ""
    row_span: int = 1
    col_span: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "col": self.col,
            "text": self.text,
            "row_span": self.row_span,
            "col_span": self.col_span,
        }


@dataclass
class ExtractedTable:
    index: int
    rows: int
    cols: int
    cells: list[ExtractedTableCell] = field(default_factory=list)

    def to_text_lines(self) -> list[str]:
        max_row = max((cell.row + cell.row_span) for cell in self.cells) if self.cells else 0
        max_col = max((cell.col + cell.col_span) for cell in self.cells) if self.cells else 0
        row_count = max(self.rows, max_row)
        col_count = max(self.cols, max_col)
        if row_count <= 0 or col_count <= 0:
            return []

        grid = [["" for _ in range(col_count)] for _ in range(row_count)]
        for cell in self.cells:
            row = max(0, int(cell.row))
            col = max(0, int(cell.col))
            if row >= row_count or col >= col_count:
                continue
            grid[row][col] = normalize_text(cell.text)

        lines = []
        for row_values in grid:
            if any(value for value in row_values):
                lines.append(" | ".join(row_values).strip())
        return [line for line in lines if line]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "rows": self.rows,
            "cols": self.cols,
            "cells": [cell.to_dict() for cell in self.cells],
            "text_rows": self.to_text_lines(),
        }


@dataclass
class ExtractedDocument:
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    tables: list[ExtractedTable] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_legacy_tuple(self) -> tuple[str, str | None]:
        return self.text, self.error

    def table_dicts(self) -> list[dict[str, Any]]:
        return [table.to_dict() for table in self.tables]
