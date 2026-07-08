# -*- coding: utf-8 -*-
"""다양한 포맷 문서 텍스트 추출."""
import os
import re
import logging
import importlib
from typing import Any, Dict, List, Optional, Tuple

from app.utils import logger, FileUtils
from app.config import AppConfig
from app.constants import ErrorMessages, Patterns, Limits
from app.services.parsers.hwp_adapter import HwpAdapter
from app.services.parsers.hwpx_adapter import HwpxAdapter
from app.services.parsers.hwp_models import (
    ExtractedDocument,
    build_basic_metadata,
    build_diagnostics,
)
from app.exceptions import (
    DocumentNotFoundError, DocumentExtractionError,
    DocumentTypeError, DocumentEmptyError
)

class DocumentExtractor:
    def __init__(self):
        self._docx_module: Any | None = None
        self._docx_loaded = False
        self._pdf_module: Any | None = None
        self._pdf_loaded = False
        self._xlsx_module: Any | None = None
        self._xlsx_loaded = False
        self._hwp_module: Any | None = None
        self._hwp_loaded = False
        self._ocr_available: Optional[bool] = None
    
    @property
    def docx(self) -> Any | None:
        if not self._docx_loaded:
            self._docx_loaded = True
            try:
                docx_module = importlib.import_module('docx')
                self._docx_module = getattr(docx_module, 'Document', None)
            except ImportError:
                self._docx_module = None
        return self._docx_module
    
    @property
    def pdf(self) -> Any | None:
        if not self._pdf_loaded:
            self._pdf_loaded = True
            try:
                pypdf_module = importlib.import_module('pypdf')
                self._pdf_module = getattr(pypdf_module, 'PdfReader', None)
            except ImportError:
                self._pdf_module = None
        return self._pdf_module
    
    @property
    def xlsx(self) -> Any | None:
        """Excel 모듈 로드 (v2.0)"""
        if not self._xlsx_loaded:
            self._xlsx_loaded = True
            try:
                self._xlsx_module = importlib.import_module('openpyxl')
            except ImportError:
                self._xlsx_module = None
        return self._xlsx_module
    
    @property
    def hwp(self) -> Any | None:
        """HWP 모듈 로드 (v2.0) - olefile 기반 기본 추출"""
        if not self._hwp_loaded:
            self._hwp_loaded = True
            try:
                self._hwp_module = importlib.import_module('olefile')
            except ImportError:
                self._hwp_module = None
        return self._hwp_module
    
    @property
    def ocr_available(self):
        """OCR 가용성 확인 (v2.0)"""
        if self._ocr_available is None:
            try:
                pytesseract = importlib.import_module('pytesseract')
                importlib.import_module('PIL.Image')
                # Tesseract 설치 확인
                pytesseract.get_tesseract_version()
                self._ocr_available = True
            except Exception:
                self._ocr_available = False
        return self._ocr_available

    def _build_result(
        self,
        path: str,
        source_format: str,
        engine_used: str,
        text: str,
        error: str | None,
        *,
        tables: Optional[List[Any]] = None,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        fallback_used: bool = False,
    ) -> ExtractedDocument:
        base_metadata = build_basic_metadata(path, source_format)
        if metadata:
            base_metadata.update(metadata)
        warning_items = [str(item) for item in (warnings or []) if str(item).strip()]
        if error and error not in warning_items:
            warning_items.append(error)
        return ExtractedDocument(
            text=text or "",
            metadata=base_metadata,
            tables=list(tables or []),
            diagnostics=build_diagnostics(
                engine_used,
                text=text or "",
                fallback_used=fallback_used,
                warnings=warning_items,
            ),
            error=error,
        )

    def extract_with_details(self, path: str) -> ExtractedDocument:
        """문서에서 텍스트와 부가정보를 함께 추출"""
        if not path:
            return self._build_result(
                path,
                "unknown",
                "unavailable",
                "",
                ErrorMessages.FILE_NOT_FOUND,
            )

        if not os.path.exists(path):
            logger.debug(f"파일 없음: {path}")
            return self._build_result(
                path,
                os.path.splitext(path)[1].lower().lstrip('.') or "unknown",
                "unavailable",
                "",
                f"{ErrorMessages.FILE_NOT_FOUND}: {path}",
            )

        if not os.path.isfile(path):
            return self._build_result(
                path,
                os.path.splitext(path)[1].lower().lstrip('.') or "unknown",
                "unavailable",
                "",
                f"파일이 아님: {path}",
            )

        ext = os.path.splitext(path)[1].lower()
        source_format = ext.lstrip('.') or "unknown"

        if ext not in AppConfig.SUPPORTED_EXTENSIONS:
            return self._build_result(
                path,
                source_format,
                "unsupported",
                "",
                f"{ErrorMessages.FILE_TYPE_NOT_SUPPORTED}: {ext}",
            )

        try:
            if ext == '.txt':
                return self._extract_txt_document(path)
            if ext == '.docx':
                return self._extract_docx_document(path)
            if ext == '.pdf':
                return self._extract_pdf_document(path)
            if ext in ['.xlsx', '.xls']:
                return self._extract_xlsx_document(path)
            if ext == '.hwp':
                return self._extract_hwp_document(path)
            if ext == '.hwpx':
                return self._extract_hwpx_document(path)
            return self._build_result(
                path,
                source_format,
                "unsupported",
                "",
                f"{ErrorMessages.FILE_TYPE_NOT_SUPPORTED}: {ext}",
            )
        except Exception as e:
            logger.error(f"문서 추출 중 예상치 못한 오류: {path} - {e}")
            return self._build_result(
                path,
                source_format,
                "unexpected-error",
                "",
                f"추출 오류: {str(e)}",
            )

    def extract(self, path: str) -> Tuple[str, Optional[str]]:
        """기존 호출부 호환용 텍스트 추출 API"""
        return self.extract_with_details(path).to_legacy_tuple()

    def _extract_txt_document(self, path: str) -> ExtractedDocument:
        text, error = self._extract_txt(path)
        return self._build_result(path, "txt", "txt", text, error)

    def _extract_docx_document(self, path: str) -> ExtractedDocument:
        text, error = self._extract_docx(path)
        return self._build_result(path, "docx", "docx", text, error)

    def _extract_pdf_document(self, path: str) -> ExtractedDocument:
        text, error = self._extract_pdf(path)
        return self._build_result(path, "pdf", "pdf", text, error)

    def _extract_xlsx_document(self, path: str) -> ExtractedDocument:
        text, error = self._extract_xlsx(path)
        return self._build_result(path, "xlsx", "xlsx", text, error)

    def _extract_hwp_document(self, path: str) -> ExtractedDocument:
        return HwpAdapter(self.hwp).extract(path)

    def _extract_hwpx_document(self, path: str) -> ExtractedDocument:
        return HwpxAdapter().extract(path)
    
    def _extract_txt(self, path: str) -> Tuple[str, Optional[str]]:
        return FileUtils.safe_read(path)
    
    def _extract_docx(self, path: str) -> Tuple[str, Optional[str]]:
        if not self.docx:
            return "", "DOCX 라이브러리 없음 (pip install python-docx)"
        try:
            doc = self.docx(path)
            parts = []
            for para in doc.paragraphs:
                if para.text.strip():
                    parts.append(para.text.strip())
            for table in doc.tables:
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        parts.append(' | '.join(cells))
            return '\n\n'.join(parts), None
        except Exception as e:
            return "", f"DOCX 오류: {e}"
    
    def _extract_pdf(self, path: str) -> Tuple[str, Optional[str]]:
        if not self.pdf:
            return "", "PDF 라이브러리 없음 (pip install pypdf)"
        try:
            reader = self.pdf(path)
            if reader.is_encrypted:
                try:
                    reader.decrypt('')
                except Exception:
                    return "", "암호화된 PDF"
            texts = []
            for page in reader.pages:
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        texts.append(text.strip())
                except Exception:
                    continue
            
            # 텍스트가 없으면 OCR 시도 (v2.0)
            if not texts:
                if self.ocr_available:
                    return self._extract_pdf_ocr(path)
                return "", "텍스트 없음 (이미지 PDF - OCR 미설치)"
            return '\n\n'.join(texts), None
        except Exception as e:
            return "", f"PDF 오류: {e}"
    
    def _extract_pdf_ocr(self, path: str) -> Tuple[str, Optional[str]]:
        """OCR을 사용한 이미지 PDF 텍스트 추출 (v2.0)"""
        try:
            pytesseract = importlib.import_module('pytesseract')
            pdf2image = importlib.import_module('pdf2image')
            convert_from_path = getattr(pdf2image, 'convert_from_path')
            
            # PDF를 이미지로 변환
            images = convert_from_path(path, dpi=150)
            texts = []
            
            for i, image in enumerate(images):
                try:
                    # 한국어 + 영어 OCR
                    text = pytesseract.image_to_string(image, lang='kor+eng')
                    if text.strip():
                        texts.append(f"[페이지 {i+1}]\n{text.strip()}")
                except Exception as e:
                    logger.warning(f"OCR 페이지 {i+1} 실패: {e}")
                    continue
            
            if texts:
                return '\n\n'.join(texts), None
            return "", "OCR 텍스트 추출 실패"
        except Exception as e:
            return "", f"OCR 오류: {e}"
    
    def _extract_xlsx(self, path: str) -> Tuple[str, Optional[str]]:
        """Excel 파일 텍스트 추출 (v2.0)"""
        if not self.xlsx:
            return "", "Excel 라이브러리 없음 (pip install openpyxl)"
        try:
            wb = self.xlsx.load_workbook(path, read_only=True, data_only=True)
            texts = []
            
            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                sheet_texts = [f"[시트: {sheet_name}]"]
                
                for row in sheet.iter_rows():
                    row_values = []
                    for cell in row:
                        if cell.value is not None:
                            row_values.append(str(cell.value).strip())
                    if any(row_values):
                        sheet_texts.append(' | '.join(filter(None, row_values)))
                
                if len(sheet_texts) > 1:  # 시트 제목만 있으면 스킵
                    texts.append('\n'.join(sheet_texts))
            
            wb.close()
            return '\n\n'.join(texts), None
        except Exception as e:
            return "", f"Excel 오류: {e}"
    
    def _extract_hwp(self, path: str) -> Tuple[str, Optional[str]]:
        return self._extract_hwp_document(path).to_legacy_tuple()

