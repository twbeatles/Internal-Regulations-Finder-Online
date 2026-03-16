from __future__ import annotations

from typing import Any

from app.utils import logger

from .hwp_models import (
    ExtractedDocument,
    build_basic_metadata,
    build_diagnostics,
    is_usable_text,
    normalize_text,
)


class HwpAdapter:
    def __init__(self, ole_module: Any | None):
        self._ole_module = ole_module

    def extract(self, path: str) -> ExtractedDocument:
        metadata = build_basic_metadata(path, "hwp", table_count=0)
        warnings: list[str] = []

        if not self._ole_module:
            error = "HWP 라이브러리 없음 (pip install olefile)"
            warnings.append("olefile 미설치로 HWP 본문 추출을 수행할 수 없습니다")
            return ExtractedDocument(
                text="",
                metadata=metadata,
                tables=[],
                diagnostics=build_diagnostics(
                    "hwp-unavailable",
                    text="",
                    fallback_used=True,
                    warnings=warnings,
                ),
                error=error,
            )

        ole = None
        try:
            ole = self._ole_module.OleFileIO(path)
            stream_entries = ole.listdir()
            metadata["stream_count"] = len(stream_entries)

            preview_text = ""
            if ole.exists("PrvText"):
                try:
                    raw_preview = ole.openstream("PrvText").read()
                    preview_text = self._decode_candidate_bytes(raw_preview, ("utf-16le", "utf-8", "cp949"))
                except Exception as exc:
                    warnings.append(f"PrvText 추출 실패: {exc}")

            body_texts: list[str] = []
            body_section_count = 0
            for entry in stream_entries:
                if not entry or str(entry[0]).lower() != "bodytext":
                    continue
                body_section_count += 1
                try:
                    data = ole.openstream(entry).read()
                    decoded = self._decode_candidate_bytes(data, ("utf-16le", "utf-8", "cp949"))
                    if decoded:
                        body_texts.append(decoded)
                except Exception as exc:
                    logger.debug("HWP BodyText 읽기 실패: %s - %s", "/".join(entry), exc)

            metadata["section_count"] = body_section_count

            text_parts: list[str] = []
            fallback_used = False
            if preview_text:
                text_parts.append(preview_text)

            unique_body_text = self._dedupe_texts(body_texts)
            if unique_body_text and (not preview_text or len(preview_text) < 120):
                text_parts.extend(unique_body_text)
                fallback_used = True
                if preview_text:
                    warnings.append("PrvText가 짧아 BodyText 휴리스틱 결과를 보완적으로 사용했습니다")
                else:
                    warnings.append("PrvText가 없어 BodyText 휴리스틱 결과를 사용했습니다")

            text = normalize_text("\n\n".join(self._dedupe_texts(text_parts)))
            metadata["used_preview_text"] = bool(preview_text)
            metadata["used_body_text_fallback"] = fallback_used

            if text:
                engine_used = "hwp-ole-bodytext" if fallback_used else "hwp-ole-preview"
                return ExtractedDocument(
                    text=text,
                    metadata=metadata,
                    tables=[],
                    diagnostics=build_diagnostics(
                        engine_used,
                        text=text,
                        fallback_used=fallback_used,
                        warnings=warnings,
                    ),
                    error=None,
                )

            error = "HWP 텍스트 추출 실패 (PrvText/BodyText에서 유효한 텍스트를 찾지 못했습니다)"
            warnings.append("텍스트로 판단 가능한 본문을 찾지 못했습니다")
            return ExtractedDocument(
                text="",
                metadata=metadata,
                tables=[],
                diagnostics=build_diagnostics(
                    "hwp-ole",
                    text="",
                    fallback_used=fallback_used,
                    warnings=warnings,
                ),
                error=error,
            )
        except Exception as exc:
            logger.warning("HWP 파일 처리 오류: %s - %s", path, exc)
            warnings.append(str(exc))
            return ExtractedDocument(
                text="",
                metadata=metadata,
                tables=[],
                diagnostics=build_diagnostics(
                    "hwp-ole",
                    text="",
                    fallback_used=False,
                    warnings=warnings,
                ),
                error=f"HWP 오류: {exc}",
            )
        finally:
            if ole is not None:
                try:
                    ole.close()
                except Exception as exc:
                    logger.debug("HWP OLE 파일 닫기 실패: %s", exc)

    def _decode_candidate_bytes(self, data: bytes, encodings: tuple[str, ...]) -> str:
        for encoding in encodings:
            try:
                decoded = normalize_text(data.decode(encoding, errors="ignore"))
            except Exception:
                continue
            if is_usable_text(decoded):
                return decoded
        return ""

    def _dedupe_texts(self, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = normalize_text(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result
