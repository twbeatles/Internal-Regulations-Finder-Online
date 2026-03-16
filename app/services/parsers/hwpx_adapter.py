from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from typing import IO
from xml.etree import ElementTree as ET

from app.utils import logger

from .hwp_models import (
    ExtractedDocument,
    ExtractedTable,
    ExtractedTableCell,
    build_basic_metadata,
    build_diagnostics,
    normalize_text,
)


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    if ":" in tag:
        return tag.split(":", 1)[1]
    return tag


@dataclass
class _CellState:
    row_span: int = 1
    col_span: int = 1
    text: str = ""


@dataclass
class _TableState:
    rows: list[list[_CellState]] = field(default_factory=list)
    current_row: list[_CellState] = field(default_factory=list)
    current_cell: _CellState | None = None
    in_row: bool = False
    in_cell: bool = False
    cell_paragraph_depth: int = 0


class HwpxAdapter:
    def extract(self, path: str) -> ExtractedDocument:
        metadata = build_basic_metadata(path, "hwpx", is_archive=True)
        warnings: list[str] = []
        fallback_used = False

        try:
            with zipfile.ZipFile(path, "r") as zf:
                section_names, path_fallback_used = self._iter_section_names(zf)
                fallback_used = path_fallback_used
                metadata["archive_entries"] = len(zf.namelist())

                if not section_names:
                    error = "HWPX 본문 섹션(Contents/section*.xml)을 찾을 수 없습니다"
                    warnings.append(error)
                    return ExtractedDocument(
                        text="",
                        metadata=metadata,
                        tables=[],
                        diagnostics=build_diagnostics(
                            "hwpx-xml",
                            text="",
                            fallback_used=True,
                            warnings=warnings,
                        ),
                        error=error,
                    )

                paragraphs: list[str] = []
                tables: list[ExtractedTable] = []
                parsed_sections = 0

                for section_name in section_names:
                    try:
                        with zf.open(section_name, "r") as stream:
                            section_paragraphs, section_tables = self._extract_section_content_stream(
                                stream,
                                table_start_index=len(tables),
                            )
                    except ET.ParseError as exc:
                        fallback_used = True
                        warnings.append(f"{section_name} XML 파싱 실패: {exc}")
                        continue
                    except Exception as exc:
                        fallback_used = True
                        warnings.append(f"{section_name} 섹션 추출 실패: {exc}")
                        logger.debug("HWPX section parse error: %s - %s", section_name, exc)
                        continue

                    if section_paragraphs or section_tables:
                        parsed_sections += 1
                    paragraphs.extend(section_paragraphs)
                    tables.extend(section_tables)

                text_parts = [text for text in paragraphs if text]
                for table in tables:
                    text_parts.extend(table.to_text_lines())
                text = normalize_text("\n\n".join(text_parts))

                metadata["section_count"] = parsed_sections or len(section_names)
                metadata["paragraph_count"] = len(paragraphs)
                metadata["table_count"] = len(tables)

                if text:
                    return ExtractedDocument(
                        text=text,
                        metadata=metadata,
                        tables=tables,
                        diagnostics=build_diagnostics(
                            "hwpx-xml",
                            text=text,
                            fallback_used=fallback_used,
                            warnings=warnings,
                        ),
                        error=None,
                    )

                error = "HWPX 텍스트 추출 실패"
                if not warnings:
                    warnings.append(error)
                return ExtractedDocument(
                    text="",
                    metadata=metadata,
                    tables=tables,
                    diagnostics=build_diagnostics(
                        "hwpx-xml",
                        text="",
                        fallback_used=fallback_used,
                        warnings=warnings,
                    ),
                    error=error,
                )
        except zipfile.BadZipFile as exc:
            error = f"HWPX ZIP 오류: {exc}"
            warnings.append("HWPX 파일이 ZIP 컨테이너 형식이 아니거나 손상되었습니다")
            return ExtractedDocument(
                text="",
                metadata=metadata,
                tables=[],
                diagnostics=build_diagnostics(
                    "hwpx-xml",
                    text="",
                    fallback_used=False,
                    warnings=warnings,
                ),
                error=error,
            )
        except Exception as exc:
            logger.warning("HWPX 파일 처리 오류: %s - %s", path, exc)
            warnings.append(str(exc))
            return ExtractedDocument(
                text="",
                metadata=metadata,
                tables=[],
                diagnostics=build_diagnostics(
                    "hwpx-xml",
                    text="",
                    fallback_used=False,
                    warnings=warnings,
                ),
                error=f"HWPX 오류: {exc}",
            )

    def _iter_section_names(self, zf: zipfile.ZipFile) -> tuple[list[str], bool]:
        names = [
            name for name in zf.namelist()
            if name.lower().startswith("contents/section") and name.lower().endswith(".xml")
        ]
        fallback_used = False
        if not names:
            names = [
                name for name in zf.namelist()
                if re.search(r"section\d+\.xml$", name, flags=re.IGNORECASE)
            ]
            fallback_used = bool(names)

        section_pattern = re.compile(r"section(\d+)\.xml$", flags=re.IGNORECASE)

        def _sort_key(name: str) -> tuple[int, int, str]:
            match = section_pattern.search(name)
            if match is None:
                return (1, 0, name.lower())
            return (0, int(match.group(1)), name.lower())

        names.sort(key=_sort_key)
        return names, fallback_used

    def _extract_section_content_stream(
        self,
        source: IO[bytes],
        *,
        table_start_index: int,
    ) -> tuple[list[str], list[ExtractedTable]]:
        paragraphs: list[str] = []
        tables: list[ExtractedTable] = []
        table_stack: list[_TableState] = []

        in_paragraph = False
        paragraph_parts: list[str] = []
        in_cell_depth = 0

        for event, elem in ET.iterparse(source, events=("start", "end")):
            name = _local_name(str(elem.tag)).lower()

            if event == "start":
                if name == "tbl":
                    table_stack.append(_TableState())
                elif name == "tr" and table_stack:
                    state = table_stack[-1]
                    state.in_row = True
                    state.current_row = []
                elif name == "tc" and table_stack:
                    in_cell_depth += 1
                    state = table_stack[-1]
                    if state.in_row:
                        state.in_cell = True
                        state.current_cell = _CellState(
                            row_span=self._attr_int(elem.attrib, "rowspan", 1),
                            col_span=self._attr_int(elem.attrib, "colspan", 1),
                            text="",
                        )
                        state.cell_paragraph_depth = 0
                elif name == "p":
                    if in_cell_depth == 0:
                        in_paragraph = True
                        paragraph_parts = []
                    elif table_stack and table_stack[-1].in_cell:
                        table_stack[-1].cell_paragraph_depth += 1
                continue

            if name == "t":
                fragment = self._clean_fragment(elem.text or "")
                if fragment:
                    if in_paragraph:
                        paragraph_parts.append(fragment)
                    if table_stack:
                        state = table_stack[-1]
                        if state.in_cell and state.current_cell is not None:
                            state.current_cell.text += fragment
            elif name == "p":
                if in_paragraph:
                    paragraph_text = normalize_text("".join(paragraph_parts))
                    if paragraph_text:
                        paragraphs.append(paragraph_text)
                    in_paragraph = False
                    paragraph_parts = []
                elif table_stack and table_stack[-1].in_cell:
                    state = table_stack[-1]
                    if state.cell_paragraph_depth > 0:
                        state.cell_paragraph_depth -= 1
                    if state.current_cell and state.current_cell.text and not state.current_cell.text.endswith("\n"):
                        state.current_cell.text += "\n"
            elif name == "tc" and table_stack:
                if in_cell_depth > 0:
                    in_cell_depth -= 1
                state = table_stack[-1]
                if state.in_cell:
                    state.in_cell = False
                    if state.current_cell is not None:
                        state.current_cell.text = normalize_text(state.current_cell.text)
                        state.current_row.append(state.current_cell)
                    state.current_cell = None
            elif name == "tr" and table_stack:
                state = table_stack[-1]
                if state.in_row:
                    state.in_row = False
                    state.rows.append(state.current_row)
                    state.current_row = []
            elif name == "tbl" and table_stack:
                finished = table_stack.pop()
                tables.append(self._rows_to_table(finished.rows, table_start_index + len(tables)))

            elem.clear()

        return paragraphs, tables

    def _rows_to_table(self, rows: list[list[_CellState]], table_index: int) -> ExtractedTable:
        cells: list[ExtractedTableCell] = []
        occupied: dict[int, list[bool]] = {}
        max_col = 0
        max_row = 0

        for row_index, row_cells in enumerate(rows):
            occupancy = occupied.setdefault(row_index, [])
            col_cursor = 0
            for cell in row_cells:
                while True:
                    if col_cursor >= len(occupancy):
                        occupancy.extend([False] * (col_cursor - len(occupancy) + 1))
                    if not occupancy[col_cursor]:
                        break
                    col_cursor += 1

                col_index = col_cursor
                row_span = max(1, int(cell.row_span or 1))
                col_span = max(1, int(cell.col_span or 1))

                for rr in range(row_index, row_index + row_span):
                    row_occupancy = occupied.setdefault(rr, [])
                    if col_index + col_span > len(row_occupancy):
                        row_occupancy.extend([False] * (col_index + col_span - len(row_occupancy)))
                    for cc in range(col_index, col_index + col_span):
                        row_occupancy[cc] = True

                cells.append(
                    ExtractedTableCell(
                        row=row_index,
                        col=col_index,
                        row_span=row_span,
                        col_span=col_span,
                        text=normalize_text(cell.text),
                    )
                )
                max_col = max(max_col, col_index + col_span)
                max_row = max(max_row, row_index + row_span)
                col_cursor += col_span

        return ExtractedTable(
            index=table_index,
            rows=max(max_row, len(rows)),
            cols=max_col,
            cells=cells,
        )

    def _attr_int(self, attrs: dict[str, str], key: str, default: int) -> int:
        direct = attrs.get(key)
        if direct is not None:
            try:
                return max(1, int(direct))
            except Exception:
                return default
        for raw_key, raw_value in attrs.items():
            lowered = raw_key.lower()
            if lowered == key or lowered.endswith("}" + key) or lowered.endswith(":" + key):
                try:
                    return max(1, int(raw_value))
                except Exception:
                    return default
        return default

    def _clean_fragment(self, text: str) -> str:
        value = str(text or "")
        value = value.replace("\ufeff", "").replace("\x00", "")
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        return value
