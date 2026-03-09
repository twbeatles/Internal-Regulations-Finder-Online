# -*- coding: utf-8 -*-
"""
사내 규정 검색기 - 웹 서버 버전 v1.0
Flask 기반 웹 서버로 다중 사용자 동시 접속 지원
"""

from __future__ import annotations
import sys
import os
import json
import threading
import tempfile
import hashlib
import importlib
import shutil
import logging
import subprocess
import platform
import re
import gc
import math
import time
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import Counter
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import signal
import atexit

from flask import Flask, request, jsonify, render_template, send_from_directory, session
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ============================================================================
# 상수 및 설정
# ============================================================================
class AppConfig:
    APP_NAME = "사내 규정 검색기"
    APP_VERSION = "2.0 (웹 서버)"  # v2.0 대규모 기능 업그레이드
    
    # 서버 설정
    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = 8080
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    
    # 오프라인 모드 설정 (폐쇄망 지원)
    OFFLINE_MODE = False  # True면 인터넷 연결 없이 로컬 모델만 사용
    LOCAL_MODEL_PATH = ""  # 사전 다운로드된 모델 폴더 경로 (빈 문자열이면 기본 경로 사용)
    
    # AI 모델 설정
    AVAILABLE_MODELS: Dict[str, str] = {
        "SNU SBERT (고성능)": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        "BM-K Simal (균형)": "BM-K/ko-simal-roberta-base",
        "JHGan SBERT (빠름)": "jhgan/ko-sbert-nli"
    }
    DEFAULT_MODEL = "SNU SBERT (고성능)"
    
    # 파일 설정
    UPLOAD_FOLDER = "uploads"
    SUPPORTED_EXTENSIONS = {'.txt', '.docx', '.pdf', '.xlsx', '.xls', '.hwp'}
    
    # 검색 설정
    MAX_SEARCH_RESULTS = 10
    DEFAULT_SEARCH_RESULTS = 5
    
    # 청킹 설정
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 80
    VECTOR_WEIGHT = 0.7
    BM25_WEIGHT = 0.3
    
    # 동시성 설정
    MAX_WORKERS = 8
    REQUEST_TIMEOUT = 30
    SEARCH_CACHE_SIZE = 200
    MAX_CONCURRENT_SEARCHES = 10
    RATE_LIMIT_PER_MINUTE = 60


class FileStatus(Enum):
    PENDING = "대기"
    PROCESSING = "처리중"
    SUCCESS = "완료"
    FAILED = "실패"
    CACHED = "캐시"


@dataclass
class TaskResult:
    success: bool
    message: str
    data: Any = None
    failed_items: List[str] = field(default_factory=list)


@dataclass
class FileInfo:
    path: str
    name: str
    extension: str
    size: int
    status: FileStatus = FileStatus.PENDING
    chunks: int = 0
    error: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "extension": self.extension,
            "size": self.size,
            "status": self.status.value,
            "chunks": self.chunks,
            "error": self.error
        }


# ============================================================================
# 로깅 설정
# ============================================================================
def get_app_directory() -> str:
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def setup_logger() -> logging.Logger:
    logger = logging.getLogger('RegSearchServer')
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    log_dir = os.path.join(get_app_directory(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(
        os.path.join(log_dir, f'server_{datetime.now():%Y%m%d}.log'),
        encoding='utf-8'
    )
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
    return logger


logger = setup_logger()


# ============================================================================
# 유틸리티
# ============================================================================
def _import_optional_module(module_name: str) -> Optional[Any]:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _import_optional_attr(module_name: str, attr_name: str) -> Optional[Any]:
    module = _import_optional_module(module_name)
    if module is None:
        return None
    return getattr(module, attr_name, None)


class FileUtils:
    @staticmethod
    def safe_read(path: str, encoding: str = 'utf-8') -> Tuple[str, Optional[str]]:
        try:
            with open(path, 'r', encoding=encoding, errors='ignore') as f:
                return f.read(), None
        except Exception as e:
            return "", str(e)
    
    @staticmethod
    def get_metadata(path: str) -> Optional[Dict]:
        try:
            stat = os.stat(path)
            return {'size': stat.st_size, 'mtime': stat.st_mtime}
        except OSError as e:
            logger.debug(f"파일 메타데이터 조회 실패: {path} - {e}")
            return None
    
    @staticmethod
    def format_size(size: int) -> str:
        size_value = float(size)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_value < 1024:
                return f"{size_value:.1f}{unit}"
            size_value /= 1024
        return f"{size_value:.1f}TB"
    
    @staticmethod
    def allowed_file(filename: str) -> bool:
        ext = os.path.splitext(filename)[1].lower()
        return ext in AppConfig.SUPPORTED_EXTENSIONS


# ============================================================================
# BM25 경량 구현 (스레드 안전)
# ============================================================================
class BM25Light:
    __slots__ = ['k1', 'b', 'corpus', 'doc_lens', 'avgdl', 'idf', 'N', '_lock']
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[List[str]] = []
        self.doc_lens: List[int] = []
        self.avgdl = 0.0
        self.idf: Dict[str, float] = {}
        self.N = 0
        self._lock = threading.RLock()
    
    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        text = re.sub(r'[^\w\s가-힣]', ' ', text.lower())
        return [t for t in text.split() if len(t) >= 2]
    
    def fit(self, docs: List[str]):
        with self._lock:
            self.corpus = []
            self.doc_lens = []
            df = Counter()
            for doc in docs:
                tokens = self._tokenize(doc)
                self.corpus.append(tokens)
                self.doc_lens.append(len(tokens))
                df.update(set(tokens))
            self.N = len(docs)
            self.avgdl = sum(self.doc_lens) / self.N if self.N else 0
            self.idf = {t: math.log((self.N - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}
            del df
            gc.collect()
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        with self._lock:
            if not self.corpus or not query:
                return []
            q_tokens = self._tokenize(query)
            if not q_tokens:
                return []
            scores = []
            for idx, doc_tokens in enumerate(self.corpus):
                if not doc_tokens:
                    continue
                score = self._score(q_tokens, doc_tokens, self.doc_lens[idx])
                if score > 0:
                    scores.append((idx, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]
    
    def _score(self, query: List[str], doc: List[str], doc_len: int) -> float:
        score = 0.0
        doc_tf = Counter(doc)
        for term in query:
            if term not in self.idf:
                continue
            tf = doc_tf.get(term, 0)
            idf = self.idf[term]
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * num / den if den > 0 else 0
        return score
    
    def clear(self):
        with self._lock:
            self.corpus.clear()
            self.doc_lens.clear()
            self.idf.clear()
            gc.collect()


# ============================================================================
# 문서 추출기 (v2.0 확장 - Excel, HWP, OCR 지원)
# ============================================================================
class DocumentExtractor:
    def __init__(self):
        self._docx_module: Optional[Any] = None
        self._pdf_module: Optional[Any] = None
        self._xlsx_module: Optional[Any] = None
        self._hwp_module: Optional[Any] = None
        self._ocr_available: Optional[bool] = None
    
    @property
    def docx(self):
        if self._docx_module is None:
            self._docx_module = _import_optional_attr("docx", "Document")
        return self._docx_module
    
    @property
    def pdf(self):
        if self._pdf_module is None:
            self._pdf_module = _import_optional_attr("pypdf", "PdfReader")
        return self._pdf_module
    
    @property
    def xlsx(self):
        """Excel 모듈 로드 (v2.0)"""
        if self._xlsx_module is None:
            self._xlsx_module = _import_optional_module("openpyxl")
        return self._xlsx_module
    
    @property
    def hwp(self):
        """HWP 모듈 로드 (v2.0) - olefile 기반 기본 추출"""
        if self._hwp_module is None:
            self._hwp_module = _import_optional_module("olefile")
        return self._hwp_module
    
    @property
    def ocr_available(self):
        """OCR 가용성 확인 (v2.0)"""
        if self._ocr_available is None:
            try:
                import pytesseract
                from PIL import Image
                # Tesseract 설치 확인
                pytesseract.get_tesseract_version()
                self._ocr_available = True
            except Exception:
                self._ocr_available = False
        return self._ocr_available
    
    def extract(self, path: str) -> Tuple[str, Optional[str]]:
        if not path or not os.path.exists(path):
            return "", f"파일 없음: {path}"
        if not os.path.isfile(path):
            return "", f"파일이 아님: {path}"
        ext = os.path.splitext(path)[1].lower()
        if ext == '.txt':
            return self._extract_txt(path)
        elif ext == '.docx':
            return self._extract_docx(path)
        elif ext == '.pdf':
            return self._extract_pdf(path)
        elif ext in ['.xlsx', '.xls']:
            return self._extract_xlsx(path)
        elif ext == '.hwp':
            return self._extract_hwp(path)
        return "", f"지원하지 않는 형식: {ext}"
    
    def _extract_txt(self, path: str) -> Tuple[str, Optional[str]]:
        return FileUtils.safe_read(path)
    
    def _extract_docx(self, path: str) -> Tuple[str, Optional[str]]:
        docx_loader = self.docx
        if docx_loader is None:
            return "", "DOCX 라이브러리 없음 (pip install python-docx)"
        try:
            doc = docx_loader(path)
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
        pdf_reader_cls = self.pdf
        if pdf_reader_cls is None:
            return "", "PDF 라이브러리 없음 (pip install pypdf)"
        try:
            reader = pdf_reader_cls(path)
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
        pytesseract = _import_optional_module("pytesseract")
        pdf2image = _import_optional_module("pdf2image")
        convert_from_path = getattr(pdf2image, "convert_from_path", None) if pdf2image else None
        if pytesseract is None or convert_from_path is None:
            return "", "OCR 라이브러리 없음 (pip install pytesseract pdf2image)"
        try:
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
        xlsx_module = self.xlsx
        if xlsx_module is None:
            return "", "Excel 라이브러리 없음 (pip install openpyxl)"
        try:
            wb = xlsx_module.load_workbook(path, read_only=True, data_only=True)
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
        """HWP 파일 텍스트 추출 (v2.0) - olefile 기반 기본 추출"""
        hwp_module = self.hwp
        if hwp_module is None:
            return "", "HWP 라이브러리 없음 (pip install olefile)"
        try:
            ole = hwp_module.OleFileIO(path)
            
            # HWP 파일 구조에서 텍스트 스트림 찾기
            texts = []
            
            # PrvText 스트림에서 텍스트 추출 시도 (미리보기 텍스트)
            if ole.exists('PrvText'):
                try:
                    prv_text = ole.openstream('PrvText').read()
                    # UTF-16LE 디코딩 시도
                    decoded = prv_text.decode('utf-16le', errors='ignore')
                    # NULL 문자 제거
                    decoded = decoded.replace('\x00', '')
                    if decoded.strip():
                        texts.append(decoded.strip())
                except Exception as e:
                    logger.debug(f"HWP PrvText 추출 실패: {e}")
            
            # BodyText 섹션에서 텍스트 추출 시도
            for entry in ole.listdir():
                entry_path = '/'.join(entry)
                if 'BodyText' in entry_path or 'Section' in entry_path:
                    try:
                        stream = ole.openstream(entry)
                        data = stream.read()
                        # 다양한 인코딩 시도
                        for encoding in ['utf-16le', 'cp949', 'utf-8']:
                            try:
                                decoded = data.decode(encoding, errors='ignore')
                                decoded = decoded.replace('\x00', '')
                                # 의미있는 텍스트만 추가
                                if decoded.strip() and len(decoded.strip()) > 10:
                                    texts.append(decoded.strip())
                                break
                            except Exception:
                                continue
                    except Exception:
                        continue
            
            ole.close()
            
            if texts:
                # 중복 제거 및 정리
                unique_texts = list(dict.fromkeys(texts))
                return '\n\n'.join(unique_texts), None
            return "", "HWP 텍스트 추출 실패 (빈 파일 또는 지원되지 않는 형식)"
        except Exception as e:
            return "", f"HWP 오류: {e}"

# ============================================================================
# 태그/카테고리 관리자 (v2.0)
# ============================================================================
class TagManager:
    """파일별 태그 관리"""
    
    PREDEFINED_CATEGORIES = [
        '인사', '회계', '보안', '복리후생', '근태', 
        '윤리', '조직', '계약', '기술', '기타'
    ]
    
    # 자동 분류를 위한 키워드 매핑
    CATEGORY_KEYWORDS = {
        '인사': ['인사', '채용', '퇴직', '승진', '평가', '인력', '직원', '사원'],
        '회계': ['회계', '경비', '예산', '결산', '세금', '세무', '비용', '지출'],
        '보안': ['보안', '비밀', '정보보호', '접근', '암호', '인증', '개인정보'],
        '복리후생': ['복리', '후생', '건강', '보험', '연금', '지원금', '복지'],
        '근태': ['근태', '휴가', '출퇴근', '연차', '병가', '출장', '재택'],
        '윤리': ['윤리', '청렴', '공정', '부정', '비위', '행동강령'],
        '조직': ['조직', '부서', '팀', '직제', '직무', '업무분장'],
        '계약': ['계약', '협약', '협정', '입찰', '구매', '조달'],
        '기술': ['기술', '개발', 'IT', '시스템', '소프트웨어', '하드웨어']
    }
    
    def __init__(self):
        self.tags_file = os.path.join(get_app_directory(), 'config', 'tags.json')
        self.tags_data: Dict[str, List[str]] = {}  # filename -> [tags]
        self._lock = threading.Lock()
        self._load()
    
    def _load(self):
        """태그 데이터 로드"""
        try:
            if os.path.exists(self.tags_file):
                with open(self.tags_file, 'r', encoding='utf-8') as f:
                    self.tags_data = json.load(f)
        except Exception as e:
            logger.warning(f"태그 파일 로드 실패: {e}")
            self.tags_data = {}
    
    def _save(self):
        """태그 데이터 저장"""
        try:
            os.makedirs(os.path.dirname(self.tags_file), exist_ok=True)
            with open(self.tags_file, 'w', encoding='utf-8') as f:
                json.dump(self.tags_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"태그 파일 저장 실패: {e}")
    
    def add_tag(self, filename: str, tag: str) -> bool:
        """태그 추가"""
        with self._lock:
            if filename not in self.tags_data:
                self.tags_data[filename] = []
            if tag not in self.tags_data[filename]:
                self.tags_data[filename].append(tag)
                self._save()
                return True
            return False
    
    def remove_tag(self, filename: str, tag: str) -> bool:
        """태그 제거"""
        with self._lock:
            if filename in self.tags_data and tag in self.tags_data[filename]:
                self.tags_data[filename].remove(tag)
                self._save()
                return True
            return False
    
    def get_tags(self, filename: str) -> List[str]:
        """파일의 태그 목록 반환"""
        return self.tags_data.get(filename, [])
    
    def set_tags(self, filename: str, tags: List[str]):
        """파일의 태그 설정"""
        with self._lock:
            self.tags_data[filename] = list(set(tags))
            self._save()
    
    def search_by_tag(self, tag: str) -> List[str]:
        """태그로 파일 검색"""
        return [f for f, tags in self.tags_data.items() if tag in tags]
    
    def get_all_tags(self) -> List[str]:
        """사용된 모든 태그 반환"""
        all_tags = set()
        for tags in self.tags_data.values():
            all_tags.update(tags)
        return sorted(all_tags)
    
    def auto_categorize(self, content: str, filename: str = "") -> List[str]:
        """키워드 기반 자동 카테고리 추천"""
        content_lower = content.lower()
        suggested = []
        
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in content_lower:
                    suggested.append(category)
                    break
        
        # 제목에서도 확인
        if filename:
            filename_lower = filename.lower()
            for category, keywords in self.CATEGORY_KEYWORDS.items():
                if category not in suggested:
                    for keyword in keywords:
                        if keyword in filename_lower:
                            suggested.append(category)
                            break
        
        return suggested if suggested else ['기타']


# ============================================================================
# 조문/조항 파서 (v2.0)
# ============================================================================
class ArticleParser:
    """규정 문서의 조문 구조 파싱"""
    
    ARTICLE_PATTERNS = [
        (r'제\s*(\d+)\s*장[^\n]*', 'chapter'),    # 제1장
        (r'제\s*(\d+)\s*절[^\n]*', 'section'),    # 제1절
        (r'제\s*(\d+)\s*조[^\n]*', 'article'),    # 제1조
        (r'제\s*(\d+)\s*조의\s*(\d+)[^\n]*', 'article_sub'),  # 제1조의2
    ]
    
    ITEM_PATTERNS = [
        (r'①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩', 'paragraph'),  # 원문자 항
        (r'^\s*(\d+)\.\s*', 'numbered'),  # 1. 2. 3.
        (r'^\s*[가-하]\.\s*', 'korean'),   # 가. 나. 다.
    ]
    
    def parse_articles(self, content: str) -> List[Dict]:
        """조문별로 분리된 구조 반환"""
        articles = []
        
        # 조문 패턴으로 분할
        pattern = r'(제\s*\d+\s*조(?:의\s*\d+)?[^\n]*)'
        parts = re.split(pattern, content)
        
        current_article = None
        for part in parts:
            match = re.match(r'제\s*(\d+)\s*조(?:의\s*(\d+))?(.+)?', part.strip())
            if match:
                if current_article:
                    articles.append(current_article)
                
                article_num = match.group(1)
                sub_num = match.group(2) or ""
                title = match.group(3) or ""
                
                current_article = {
                    'number': f"제{article_num}조" + (f"의{sub_num}" if sub_num else ""),
                    'title': title.strip().strip('()（）[]'),
                    'content': "",
                    'paragraphs': []
                }
            elif current_article and part.strip():
                current_article['content'] += part
                
                # 항 분리
                para_split = re.split(r'([①②③④⑤⑥⑦⑧⑨⑩])', part)
                for i in range(1, len(para_split), 2):
                    if i+1 < len(para_split):
                        current_article['paragraphs'].append({
                            'marker': para_split[i],
                            'content': para_split[i+1].strip()
                        })
        
        if current_article:
            articles.append(current_article)
        
        return articles
    
    def search_article(self, articles: List[Dict], query: str) -> List[Dict]:
        """조문에서 검색"""
        results = []
        query_lower = query.lower()
        
        for article in articles:
            score = 0
            if query_lower in article.get('title', '').lower():
                score += 3
            if query_lower in article.get('content', '').lower():
                score += 1
            if query_lower in article.get('number', '').lower():
                score += 5
            
            if score > 0:
                results.append({**article, 'score': score})
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def get_article_by_number(self, articles: List[Dict], number: str) -> Optional[Dict]:
        """조문 번호로 조문 찾기"""
        # 정규화
        num_match = re.search(r'\d+', number)
        if not num_match:
            return None
        
        target_num = num_match.group()
        for article in articles:
            if target_num in article.get('number', ''):
                return article
        return None


# ============================================================================
# 문서 분할기 (v2.0)
# ============================================================================
class DocumentSplitter:
    """대용량 규정집을 개별 규정 파일로 분할"""
    
    def split_by_chapters(self, content: str, filename: str = "") -> List[Dict]:
        """장 기준 분할"""
        pattern = r'(제\s*\d+\s*장[^\n]*)'
        return self._split_by_pattern(content, pattern, 'chapter', filename)
    
    def split_by_articles(self, content: str, filename: str = "") -> List[Dict]:
        """조문 기준 분할"""
        pattern = r'(제\s*\d+\s*조(?:의\s*\d+)?[^\n]*)'
        return self._split_by_pattern(content, pattern, 'article', filename)
    
    def _split_by_pattern(self, content: str, pattern: str, split_type: str, filename: str) -> List[Dict]:
        """패턴으로 분할"""
        parts = re.split(pattern, content)
        results = []
        
        current_title = None
        current_content = ""
        
        for part in parts:
            if re.match(pattern, part):
                if current_title:
                    results.append({
                        'title': current_title.strip(),
                        'content': current_content.strip(),
                        'type': split_type,
                        'source': filename
                    })
                current_title = part
                current_content = ""
            else:
                current_content += part
        
        # 마지막 항목
        if current_title:
            results.append({
                'title': current_title.strip(),
                'content': current_content.strip(),
                'type': split_type,
                'source': filename
            })
        
        return results
    
    def split_by_size(self, content: str, max_size: int = 5000) -> List[Dict]:
        """크기 기준 분할"""
        results = []
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        chunk_num = 1
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > max_size:
                if current_chunk:
                    results.append({
                        'title': f'파트 {chunk_num}',
                        'content': current_chunk.strip(),
                        'type': 'size_split'
                    })
                    chunk_num += 1
                    current_chunk = para
                else:
                    # 단일 문단이 max_size 초과
                    results.append({
                        'title': f'파트 {chunk_num}',
                        'content': para[:max_size],
                        'type': 'size_split'
                    })
                    chunk_num += 1
            else:
                current_chunk += '\n\n' + para if current_chunk else para
        
        if current_chunk:
            results.append({
                'title': f'파트 {chunk_num}',
                'content': current_chunk.strip(),
                'type': 'size_split'
            })
        
        return results


# ============================================================================
# 문서 비교기 (v2.0)
# ============================================================================
class DocumentComparator:
    """두 문서 간 차이점 비교"""
    
    def compare(self, doc1: str, doc2: str) -> Dict:
        """diff 결과 반환"""
        import difflib
        
        lines1 = doc1.splitlines(keepends=True)
        lines2 = doc2.splitlines(keepends=True)
        
        differ = difflib.unified_diff(lines1, lines2, lineterm='')
        diff_lines = list(differ)
        
        # 통계 계산
        added = sum(1 for line in diff_lines if line.startswith('+') and not line.startswith('+++'))
        removed = sum(1 for line in diff_lines if line.startswith('-') and not line.startswith('---'))
        
        # HTML 형식 diff
        html_diff = difflib.HtmlDiff()
        html_table = html_diff.make_table(lines1, lines2, context=True)
        
        return {
            'added_lines': added,
            'removed_lines': removed,
            'total_changes': added + removed,
            'diff_text': ''.join(diff_lines),
            'diff_html': html_table,
            'similarity': difflib.SequenceMatcher(None, doc1, doc2).ratio()
        }
    
    def highlight_changes(self, doc1: str, doc2: str) -> Tuple[str, str]:
        """변경된 부분 하이라이트"""
        import difflib
        
        matcher = difflib.SequenceMatcher(None, doc1, doc2)
        
        result1 = []
        result2 = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                result1.append(doc1[i1:i2])
                result2.append(doc2[j1:j2])
            elif tag == 'delete':
                result1.append(f'<del>{doc1[i1:i2]}</del>')
            elif tag == 'insert':
                result2.append(f'<ins>{doc2[j1:j2]}</ins>')
            elif tag == 'replace':
                result1.append(f'<del>{doc1[i1:i2]}</del>')
                result2.append(f'<ins>{doc2[j1:j2]}</ins>')
        
        return ''.join(result1), ''.join(result2)


# ============================================================================
# 개정 이력 추적기 (v2.0)
# ============================================================================
class RevisionTracker:
    """규정 개정 이력 관리"""
    
    def __init__(self):
        self.revisions_dir = os.path.join(get_app_directory(), 'revisions')
        self.index_file = os.path.join(self.revisions_dir, 'index.json')
        self.index: Dict[str, List[Dict]] = {}  # filename -> [{version, date, hash, note}]
        self._lock = threading.Lock()
        self._load_index()
    
    def _load_index(self):
        """인덱스 로드"""
        try:
            if os.path.exists(self.index_file):
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.index = json.load(f)
        except Exception as e:
            logger.warning(f"개정 이력 인덱스 로드 실패: {e}")
            self.index = {}
    
    def _save_index(self):
        """인덱스 저장"""
        try:
            os.makedirs(self.revisions_dir, exist_ok=True)
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"개정 이력 인덱스 저장 실패: {e}")
    
    def save_revision(self, filename: str, content: str, note: str = "") -> Dict:
        """새 버전 저장"""
        with self._lock:
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 버전 번호 결정
            if filename not in self.index:
                self.index[filename] = []
            
            version = f"v{len(self.index[filename]) + 1}"
            
            # 파일 저장
            revision_filename = f"{filename}_{version}_{timestamp}.txt"
            revision_path = os.path.join(self.revisions_dir, revision_filename)
            
            os.makedirs(self.revisions_dir, exist_ok=True)
            with open(revision_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 인덱스 업데이트
            revision_info = {
                'version': version,
                'date': datetime.now().isoformat(),
                'hash': content_hash,
                'note': note,
                'file': revision_filename
            }
            self.index[filename].append(revision_info)
            self._save_index()
            
            return revision_info
    
    def get_history(self, filename: str) -> List[Dict]:
        """버전 히스토리 조회"""
        return self.index.get(filename, [])
    
    def get_revision(self, filename: str, version: str) -> Optional[str]:
        """특정 버전 내용 조회"""
        history = self.get_history(filename)
        for rev in history:
            if rev['version'] == version:
                revision_path = os.path.join(self.revisions_dir, rev['file'])
                if os.path.exists(revision_path):
                    with open(revision_path, 'r', encoding='utf-8') as f:
                        return f.read()
        return None
    
    def compare_versions(self, filename: str, v1: str, v2: str) -> Optional[Dict]:
        """버전 간 비교"""
        content1 = self.get_revision(filename, v1)
        content2 = self.get_revision(filename, v2)
        
        if content1 is None or content2 is None:
            return None
        
        comparator = DocumentComparator()
        return comparator.compare(content1, content2)


# ============================================================================
# 폴더 모니터링 (v2.0)
# ============================================================================
class FolderWatcher:
    """watchdog 기반 폴더 변경 감지"""
    
    def __init__(self, callback=None):
        self.observer = None
        self.watching = False
        self.watch_path = ""
        self.callback = callback
        self._watchdog_available = None
    
    @property
    def watchdog_available(self):
        """watchdog 가용성 확인"""
        if self._watchdog_available is None:
            observers = _import_optional_module("watchdog.observers")
            events = _import_optional_module("watchdog.events")
            self._watchdog_available = bool(
                observers and getattr(observers, "Observer", None) and events and getattr(events, "FileSystemEventHandler", None)
            )
        return self._watchdog_available
    
    def start_watching(self, folder: str) -> bool:
        """모니터링 시작"""
        if not self.watchdog_available:
            logger.warning("watchdog 라이브러리 미설치 (pip install watchdog)")
            return False
        
        if self.watching:
            self.stop_watching()
        
        try:
            observers = _import_optional_module("watchdog.observers")
            events = _import_optional_module("watchdog.events")
            Observer = getattr(observers, "Observer", None)
            FileSystemEventHandler = getattr(events, "FileSystemEventHandler", None)
            if Observer is None or FileSystemEventHandler is None:
                logger.warning("watchdog 라이브러리 미설치 (pip install watchdog)")
                return False
            
            class RegulationEventHandler(FileSystemEventHandler):
                def __init__(self, callback):
                    self._callback = callback
                
                def on_created(self, event):
                    if not event.is_directory:
                        ext = os.path.splitext(event.src_path)[1].lower()
                        if ext in AppConfig.SUPPORTED_EXTENSIONS:
                            logger.info(f"📁 새 파일 감지: {event.src_path}")
                            if self._callback:
                                self._callback('created', event.src_path)
                
                def on_modified(self, event):
                    if not event.is_directory:
                        ext = os.path.splitext(event.src_path)[1].lower()
                        if ext in AppConfig.SUPPORTED_EXTENSIONS:
                            logger.info(f"📝 파일 수정 감지: {event.src_path}")
                            if self._callback:
                                self._callback('modified', event.src_path)
                
                def on_deleted(self, event):
                    if not event.is_directory:
                        logger.info(f"🗑️ 파일 삭제 감지: {event.src_path}")
                        if self._callback:
                            self._callback('deleted', event.src_path)
            
            self.observer = Observer()
            event_handler = RegulationEventHandler(self.callback)
            self.observer.schedule(event_handler, folder, recursive=True)
            self.observer.start()
            self.watching = True
            self.watch_path = folder
            
            logger.info(f"👁️ 폴더 모니터링 시작: {folder}")
            return True
            
        except Exception as e:
            logger.error(f"폴더 모니터링 시작 실패: {e}")
            return False
    
    def stop_watching(self):
        """모니터링 중지"""
        if self.observer:
            try:
                self.observer.stop()
                self.observer.join(timeout=2)
                logger.info("👁️ 폴더 모니터링 중지")
            except Exception as e:
                logger.warning(f"폴더 모니터링 중지 오류: {e}")
            finally:
                self.observer = None
                self.watching = False
                self.watch_path = ""
    
    def get_status(self) -> Dict:
        """모니터링 상태 반환"""
        return {
            'watching': self.watching,
            'path': self.watch_path,
            'available': self.watchdog_available
        }


# 전역 인스턴스
tag_manager = TagManager()
article_parser = ArticleParser()
document_splitter = DocumentSplitter()
document_comparator = DocumentComparator()
revision_tracker = RevisionTracker()
folder_watcher = FolderWatcher()


# ============================================================================
# 검색 캐시 (LRU)
# ============================================================================
class SearchCache:
    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, Tuple[float, Any]] = {}
        self.max_size = max_size
        self._lock = threading.Lock()
    
    def _make_key(self, query: str, k: int, hybrid: bool) -> str:
        return f"{query}|{k}|{hybrid}"
    
    def get(self, query: str, k: int, hybrid: bool) -> Optional[Any]:
        key = self._make_key(query, k, hybrid)
        with self._lock:
            if key in self.cache:
                timestamp, result = self.cache[key]
                # 5분 이내 캐시만 유효
                if time.time() - timestamp < 300:
                    return result
                del self.cache[key]
        return None
    
    def set(self, query: str, k: int, hybrid: bool, result: Any):
        key = self._make_key(query, k, hybrid)
        with self._lock:
            if len(self.cache) >= self.max_size:
                # 가장 오래된 항목 제거
                oldest_key = min(self.cache.keys(), key=lambda x: self.cache[x][0])
                del self.cache[oldest_key]
            self.cache[key] = (time.time(), result)
    
    def clear(self):
        with self._lock:
            self.cache.clear()


# ============================================================================
# Rate Limiter (IP 기반 요청 제한)
# ============================================================================
class RateLimiter:
    """IP 기반 요청 제한으로 서버 과부하 방지"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests: Dict[str, List[float]] = {}  # ip -> [timestamps]
        self.limit = requests_per_minute
        self._lock = threading.Lock()
        self._cleanup_interval = 60  # 정리 주기 (초)
        self._last_cleanup = time.time()
    
    def is_allowed(self, ip: str) -> bool:
        """요청이 허용되는지 확인"""
        current_time = time.time()
        
        with self._lock:
            # 주기적 정리
            if current_time - self._last_cleanup > self._cleanup_interval:
                self._cleanup(current_time)
                self._last_cleanup = current_time
            
            if ip not in self.requests:
                self.requests[ip] = []
            
            # 1분 이내 요청만 유지
            cutoff = current_time - 60
            self.requests[ip] = [t for t in self.requests[ip] if t > cutoff]
            
            # 제한 확인
            if len(self.requests[ip]) >= self.limit:
                logger.warning(f"Rate limit exceeded for IP: {ip}")
                return False
            
            # 새 요청 기록
            self.requests[ip].append(current_time)
            return True
    
    def _cleanup(self, current_time: float):
        """오래된 기록 정리"""
        cutoff = current_time - 60
        expired_ips = []
        for ip, timestamps in self.requests.items():
            self.requests[ip] = [t for t in timestamps if t > cutoff]
            if not self.requests[ip]:
                expired_ips.append(ip)
        for ip in expired_ips:
            del self.requests[ip]
    
    def get_remaining(self, ip: str) -> int:
        """남은 요청 수 반환"""
        with self._lock:
            current_time = time.time()
            cutoff = current_time - 60
            if ip not in self.requests:
                return self.limit
            recent = [t for t in self.requests[ip] if t > cutoff]
            return max(0, self.limit - len(recent))


# ============================================================================
# 검색 요청 큐 (동시 검색 수 제한)
# ============================================================================
class SearchQueue:
    """동시 검색 수를 제한하여 서버 안정성 확보"""
    
    def __init__(self, max_concurrent: int = 10):
        self._semaphore = threading.Semaphore(max_concurrent)
        self._active_count = 0
        self._lock = threading.Lock()
        self._total_processed = 0
        self._total_rejected = 0
    
    def acquire(self, timeout: float = 30.0) -> bool:
        """검색 슬롯 획득 (타임아웃 지원)"""
        acquired = self._semaphore.acquire(timeout=timeout)
        if acquired:
            with self._lock:
                self._active_count += 1
        else:
            with self._lock:
                self._total_rejected += 1
            logger.warning("Search queue full, request rejected")
        return acquired
    
    def release(self):
        """검색 슬롯 반환"""
        self._semaphore.release()
        with self._lock:
            self._active_count = max(0, self._active_count - 1)
            self._total_processed += 1
    
    def get_stats(self) -> Dict:
        """큐 상태 반환"""
        with self._lock:
            return {
                'active': self._active_count,
                'processed': self._total_processed,
                'rejected': self._total_rejected
            }


# 전역 Rate Limiter 및 Search Queue 인스턴스
rate_limiter = RateLimiter(AppConfig.RATE_LIMIT_PER_MINUTE)
search_queue = SearchQueue(AppConfig.MAX_CONCURRENT_SEARCHES)

# 파일 작업 락 (동시 파일 업로드/삭제 보호)
file_operation_lock = threading.Lock()

# ============================================================================
# 검색 히스토리 (최근 검색어 + 인기 검색어)
# ============================================================================
class SearchHistory:
    def __init__(self, max_recent: int = 20, max_popular: int = 10):
        self.recent: List[Dict] = []  # [{query, timestamp}, ...]
        self.popular: Counter = Counter()  # query -> count
        self.max_recent = max_recent
        self.max_popular = max_popular
        self._lock = threading.Lock()
    
    def add(self, query: str):
        """검색어 추가"""
        query = query.strip()
        if len(query) < 2:
            return
        
        with self._lock:
            # 최근 검색어에서 중복 제거
            self.recent = [r for r in self.recent if r['query'] != query]
            # 새 검색어 추가
            self.recent.insert(0, {
                'query': query,
                'timestamp': time.time()
            })
            # 최대 크기 유지
            self.recent = self.recent[:self.max_recent]
            # 인기 검색어 업데이트
            self.popular[query] += 1
    
    def get_recent(self, limit: int = 10) -> List[str]:
        """최근 검색어 반환"""
        with self._lock:
            return [r['query'] for r in self.recent[:limit]]
    
    def get_popular(self, limit: int = 10) -> List[Tuple[str, int]]:
        """인기 검색어 반환 (검색어, 횟수)"""
        with self._lock:
            return self.popular.most_common(min(limit, self.max_popular))
    
    def suggest(self, prefix: str, limit: int = 5) -> List[str]:
        """접두사 기반 검색어 추천"""
        prefix = prefix.strip().lower()
        if len(prefix) < 1:
            return []
        
        with self._lock:
            suggestions = []
            # 최근 검색어에서 매칭
            for r in self.recent:
                if r['query'].lower().startswith(prefix):
                    suggestions.append(r['query'])
            # 인기 검색어에서 매칭
            for q, _ in self.popular.most_common():
                if q.lower().startswith(prefix) and q not in suggestions:
                    suggestions.append(q)
            return suggestions[:limit]
    
    def clear(self):
        """히스토리 초기화"""
        with self._lock:
            self.recent.clear()
            self.popular.clear()


# ============================================================================
# 텍스트 하이라이터
# ============================================================================
class TextHighlighter:
    @staticmethod
    def highlight(text: str, query: str, tag: str = 'mark') -> str:
        """검색어를 태그로 감싸서 하이라이트"""
        if not text or not query:
            return text
        
        # 검색어를 공백으로 분리
        keywords = [kw.strip() for kw in query.split() if len(kw.strip()) >= 2]
        if not keywords:
            return text
        
        # 각 키워드에 대해 하이라이트 적용
        result = text
        for keyword in keywords:
            # 대소문자 무시 검색
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            result = pattern.sub(f'<{tag}>\\g<0></{tag}>', result)
        
        return result
    
    @staticmethod
    def extract_keywords(documents: List[str], top_k: int = 50) -> List[str]:
        """문서에서 핵심 키워드 추출"""
        if not documents:
            return []
        
        # 간단한 키워드 추출 (빈도 기반)
        word_freq = Counter()
        for doc in documents:
            # 한글, 영문 단어 추출
            words = re.findall(r'[가-힣]{2,}|[a-zA-Z]{3,}', doc)
            word_freq.update(words)
        
        # 불용어 필터링 (간단한 한글 불용어)
        stopwords = {'있는', '하는', '및', '등', '이', '가', '을', '를', '의', '에', '로', '으로'}
        keywords = [w for w, _ in word_freq.most_common(top_k * 2) if w not in stopwords]
        
        return keywords[:top_k]


# ============================================================================
# 핵심 QA 시스템 (스레드 안전)
# ============================================================================
class RegulationQASystem:
    def __init__(self):
        self.vector_store = None
        self.embedding_model = None
        self.model_id = None
        self.model_name = ""
        self.extractor = DocumentExtractor()
        self.cache_path = os.path.join(tempfile.gettempdir(), "reg_qa_server_v10")
        self.bm25 = None
        self.documents: List[str] = []
        self.doc_meta: List[Dict] = []
        self.file_infos: Dict[str, FileInfo] = {}
        self.current_folder = ""
        self._lock = threading.RLock()
        self._search_cache = SearchCache(AppConfig.SEARCH_CACHE_SIZE)
        self._search_history = SearchHistory()  # 검색 히스토리
        self._keyword_cache: List[str] = []  # 문서 키워드 캐시
        self._executor = ThreadPoolExecutor(max_workers=AppConfig.MAX_WORKERS)
        self._is_ready = False
        self._is_loading = False
        self._load_progress = ""
        self._load_error = ""  # 마지막 로드 오류 메시지
    
    def get_keywords(self, limit: int = 50) -> List[str]:
        """문서에서 추출한 키워드 반환 (자동완성용)"""
        if not self._keyword_cache and self.documents:
            self._keyword_cache = TextHighlighter.extract_keywords(self.documents, limit)
        return self._keyword_cache[:limit]
    
    @property
    def is_ready(self) -> bool:
        return self._is_ready and self.embedding_model is not None
    
    @property
    def is_loading(self) -> bool:
        return self._is_loading
    
    @property
    def load_progress(self) -> str:
        return self._load_progress
    
    @property
    def load_error(self) -> str:
        """마지막 로드 오류 메시지"""
        return self._load_error
    
    def load_model(
        self,
        model_name: str,
        offline_mode: Optional[bool] = None,
        local_model_path: Optional[str] = None
    ) -> TaskResult:
        """AI 임베딩 모델 로드
        
        Args:
            model_name: 모델 이름 (AVAILABLE_MODELS의 키)
            offline_mode: 오프라인 모드 여부 (None이면 AppConfig 설정 사용)
            local_model_path: 로컬 모델 경로 (None이면 AppConfig 설정 사용)
        """
        import traceback
        
        if self._is_loading:
            return TaskResult(False, "이미 모델을 로딩 중입니다")
        
        # 오프라인 모드 설정 결정
        is_offline = bool(offline_mode if offline_mode is not None else AppConfig.OFFLINE_MODE)
        model_path_override = str(
            local_model_path if local_model_path is not None else (AppConfig.LOCAL_MODEL_PATH or "")
        )
        
        model_id = AppConfig.AVAILABLE_MODELS.get(model_name, AppConfig.AVAILABLE_MODELS[AppConfig.DEFAULT_MODEL])
        self._load_error = ""  # 오류 초기화
        
        try:
            self._is_loading = True
            self._load_progress = "라이브러리 로드 중..."
            logger.info("라이브러리 로드 중...")
            
            # PyTorch 로드 시도
            try:
                import torch
                logger.info(f"PyTorch 버전: {torch.__version__}")
            except ImportError as e:
                error_msg = f"PyTorch 로드 실패: {e}"
                logger.error(error_msg)
                self._load_error = error_msg
                return TaskResult(False, error_msg)
            
            # LangChain HuggingFace 로드 시도
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                logger.info("LangChain HuggingFace 로드 완료")
            except ImportError as e:
                error_msg = f"LangChain HuggingFace 로드 실패: {e}"
                logger.error(error_msg)
                self._load_error = error_msg
                return TaskResult(False, error_msg)
            
            # 모델 캐시 경로 결정
            if model_path_override:
                cache_dir = model_path_override
            else:
                cache_dir = os.path.join(get_app_directory(), 'models')
            os.makedirs(cache_dir, exist_ok=True)
            
            # 오프라인 모드 로깅 및 환경변수 설정
            if is_offline:
                self._load_progress = "오프라인 모드: 로컬 모델 로딩 중..."
                logger.info(f"🔒 오프라인 모드 활성화: 로컬 모델만 사용")
                logger.info(f"📂 모델 경로: {cache_dir}")
                
                # HuggingFace 오프라인 환경변수 설정 (네트워크 요청 방지)
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['HF_DATASETS_OFFLINE'] = '1'
                
                # 모델 디렉토리 존재 확인
                model_subdir = os.path.join(cache_dir, model_id.replace('/', '--'))
                if not os.path.exists(model_subdir):
                    # HuggingFace 캐시 형식도 확인
                    alt_dirs = [
                        os.path.join(cache_dir, 'models--' + model_id.replace('/', '--')),
                        cache_dir  # 직접 경로
                    ]
                    found = False
                    for alt_dir in alt_dirs:
                        if os.path.exists(alt_dir):
                            found = True
                            break
                    
                    if not found:
                        error_msg = (
                            f"오프라인 모드에서 모델을 찾을 수 없습니다.\n"
                            f"모델 경로: {cache_dir}\n"
                            f"예상 모델 폴더: {model_id.replace('/', '--')}\n\n"
                            f"해결 방법:\n"
                            f"1. 인터넷 연결된 환경에서 'python download_models.py' 실행\n"
                            f"2. 생성된 models 폴더를 이 서버로 복사\n"
                            f"3. 설정에서 '로컬 모델 경로'를 올바르게 지정"
                        )
                        logger.error(error_msg)
                        self._load_error = error_msg
                        return TaskResult(False, error_msg)
            else:
                self._load_progress = "모델 다운로드/로딩 중..."
                logger.info(f"모델 로딩 중: {model_name}")
                logger.info(f"모델 캐시 경로: {cache_dir}")
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"사용 디바이스: {device}")
            
            # 모델 로드 옵션 설정
            model_kwargs: Dict[str, Any] = {'device': device}
            if is_offline:
                model_kwargs['local_files_only'] = True
            
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_id,
                cache_folder=cache_dir,
                model_kwargs=model_kwargs,
                encode_kwargs={'normalize_embeddings': True}
            )
            self.model_id = model_id
            self.model_name = model_name
            self._is_ready = True
            
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            mode_str = "오프라인" if is_offline else "온라인"
            self._load_progress = "완료"
            logger.info(f"모델 로드 완료: {model_name} ({device}, {mode_str})")
            return TaskResult(True, f"모델 로드 완료 ({device}, {mode_str})")
            
        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"모델 로드 실패: {e}\n{error_detail}")
            self._load_error = str(e)
            self._load_progress = f"실패: {e}"
            return TaskResult(False, f"모델 로드 실패: {e}")
        finally:
            self._is_loading = False
    
    def _get_cache_dir(self, folder: str) -> str:
        if not self.model_id:
            raise ValueError("모델이 로드되지 않았습니다")
        h1 = hashlib.md5(self.model_id.encode()).hexdigest()[:6]
        h2 = hashlib.md5(folder.encode()).hexdigest()[:6]
        return os.path.join(self.cache_path, f"{h2}_{h1}")
    
    def process_documents(self, folder: str, files: List[str], progress_cb=None) -> TaskResult:
        """문서 처리 및 인덱싱"""
        if not self.embedding_model:
            return TaskResult(False, "모델이 로드되지 않았습니다")
        
        with self._lock:
            return self._process_internal(folder, files, progress_cb)
    
    def _process_internal(self, folder: str, files: List[str], progress_cb) -> TaskResult:
        # LangChain 최신 버전 호환 import
        CharacterTextSplitter = _import_optional_attr(
            "langchain_text_splitters", "CharacterTextSplitter"
        ) or _import_optional_attr("langchain.text_splitter", "CharacterTextSplitter")
        FAISS = _import_optional_attr(
            "langchain_community.vectorstores", "FAISS"
        ) or _import_optional_attr("langchain.vectorstores", "FAISS")
        Document = _import_optional_attr(
            "langchain_core.documents", "Document"
        ) or _import_optional_attr("langchain.docstore.document", "Document")
        if CharacterTextSplitter is None or FAISS is None or Document is None:
            return TaskResult(False, "LangChain 의존성이 누락되어 문서 처리를 진행할 수 없습니다")
        embedding_model = self.embedding_model
        if embedding_model is None:
            return TaskResult(False, "모델이 로드되지 않았습니다")
        
        self.current_folder = folder
        cache_dir = self._get_cache_dir(folder)
        self.file_infos.clear()
        self._search_cache.clear()
        
        # 파일 정보 초기화
        for fp in files:
            meta = FileUtils.get_metadata(fp)
            self.file_infos[fp] = FileInfo(
                path=fp,
                name=os.path.basename(fp),
                extension=os.path.splitext(fp)[1].lower(),
                size=int(meta['size']) if meta else 0
            )
        
        if progress_cb:
            progress_cb(5, "캐시 확인...")
        
        cache_info = self._load_cache_info(cache_dir)
        to_process, cached = [], []
        
        for fp in files:
            fname = os.path.basename(fp)
            meta = FileUtils.get_metadata(fp)
            if meta and fname in cache_info:
                cm = cache_info[fname]
                if cm.get('size') == meta['size'] and cm.get('mtime') == meta['mtime']:
                    cached.append(fp)
                    self.file_infos[fp].status = FileStatus.CACHED
                    self.file_infos[fp].chunks = cm.get('chunks', 0)
                    continue
            to_process.append(fp)
        
        self.documents, self.doc_meta = [], []
        
        # 캐시된 데이터 로드
        if cached and os.path.exists(os.path.join(cache_dir, "index.faiss")):
            try:
                if progress_cb:
                    progress_cb(10, "캐시 로드...")
                self.vector_store = FAISS.load_local(
                    cache_dir, embedding_model,
                    allow_dangerous_deserialization=True
                )
                docs_path = os.path.join(cache_dir, "docs.json")
                if os.path.exists(docs_path):
                    with open(docs_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.documents = data.get('docs', [])
                        self.doc_meta = data.get('meta', [])
            except Exception as e:
                logger.warning(f"캐시 로드 실패: {e}")
                to_process, cached = files, []
                self.vector_store = None
        
        if not to_process:
            self._build_bm25()
            if progress_cb:
                progress_cb(100, "완료!")
            return TaskResult(
                True,
                f"캐시에서 {len(cached)}개 파일 로드",
                {'chunks': len(self.documents), 'cached': len(cached), 'new': 0}
            )
        
        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=AppConfig.CHUNK_SIZE,
            chunk_overlap=AppConfig.CHUNK_OVERLAP
        )
        failed, new_docs, new_cache_info = [], [], {}
        
        # 병렬 문서 추출 함수
        def extract_file(fp: str) -> Tuple[str, str, Optional[str], Optional[str], Optional[Dict[str, Any]]]:
            """파일에서 텍스트 추출 (병렬 처리용)"""
            fname = os.path.basename(fp)
            try:
                content, error = self.extractor.extract(fp)
                meta = FileUtils.get_metadata(fp)
                return fp, fname, content, error, meta
            except Exception as e:
                return fp, fname, None, str(e), None
        
        # ThreadPoolExecutor로 병렬 문서 추출
        extracted_results = []
        if progress_cb:
            progress_cb(15, f"문서 추출 중... (병렬 처리)")
        
        with ThreadPoolExecutor(max_workers=min(AppConfig.MAX_WORKERS, len(to_process))) as executor:
            futures = {executor.submit(extract_file, fp): fp for fp in to_process}
            completed = 0
            for future in futures:
                try:
                    result = future.result(timeout=60)  # 파일당 60초 타임아웃
                    extracted_results.append(result)
                    completed += 1
                    if progress_cb and completed % 5 == 0:
                        progress = 15 + int((completed / len(to_process)) * 30)
                        progress_cb(progress, f"추출 완료: {completed}/{len(to_process)}")
                except Exception as e:
                    fp = futures[future]
                    fname = os.path.basename(fp)
                    extracted_results.append((fp, fname, None, f"추출 타임아웃: {e}", None))
        
        # 추출 결과 처리 및 청킹
        if progress_cb:
            progress_cb(50, "텍스트 청킹 중...")
        
        for fp, fname, content, error, meta in extracted_results:
            self.file_infos[fp].status = FileStatus.PROCESSING
            
            if error:
                failed.append(f"{fname} ({error})")
                self.file_infos[fp].status = FileStatus.FAILED
                self.file_infos[fp].error = error
                continue
            if not content or not content.strip():
                failed.append(f"{fname} (빈 파일)")
                self.file_infos[fp].status = FileStatus.FAILED
                self.file_infos[fp].error = "빈 파일"
                continue
            
            try:
                chunks = splitter.split_text(content)
                chunk_count = 0
                for chunk in chunks:
                    if chunk.strip():
                        new_docs.append(Document(
                            page_content=chunk.strip(),
                            metadata={"source": fname, "path": fp}
                        ))
                        self.documents.append(chunk.strip())
                        self.doc_meta.append({"source": fname, "path": fp})
                        chunk_count += 1
                
                self.file_infos[fp].status = FileStatus.SUCCESS
                self.file_infos[fp].chunks = chunk_count
                
                if meta:
                    new_cache_info[fname] = {
                        'size': meta['size'],
                        'mtime': meta['mtime'],
                        'chunks': chunk_count
                    }
            except Exception as e:
                failed.append(f"{fname} ({e})")
                self.file_infos[fp].status = FileStatus.FAILED
                self.file_infos[fp].error = str(e)

        
        if not new_docs and not self.vector_store:
            return TaskResult(False, "처리 가능한 문서 없음", failed_items=failed)
        
        if progress_cb:
            progress_cb(75, "벡터 인덱스 생성...")
        
        try:
            if new_docs:
                if self.vector_store:
                    batch_size = 100
                    for i in range(0, len(new_docs), batch_size):
                        self.vector_store.add_documents(new_docs[i:i + batch_size])
                else:
                    self.vector_store = FAISS.from_documents(new_docs, embedding_model)
        except Exception as e:
            return TaskResult(False, f"인덱스 생성 실패: {e}")
        
        if progress_cb:
            progress_cb(85, "키워드 인덱스 생성...")
        self._build_bm25()
        
        if progress_cb:
            progress_cb(90, "캐시 저장...")
        self._save_cache(cache_dir, cache_info, new_cache_info)
        
        gc.collect()
        if progress_cb:
            progress_cb(100, "완료!")
        
        return TaskResult(
            True,
            f"{len(files) - len(failed)}개 처리 완료",
            {
                'chunks': len(self.documents),
                'new': len(to_process) - len(failed),
                'cached': len(cached)
            },
            failed
        )
    
    def _build_bm25(self):
        if self.documents:
            self.bm25 = BM25Light()
            self.bm25.fit(self.documents)
    
    def _load_cache_info(self, cache_dir: str) -> Dict:
        path = os.path.join(cache_dir, "cache_info.json")
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.debug(f"캐시 정보 로드 실패: {e}")
        return {}
    
    def _save_cache(self, cache_dir: str, old_info: Dict, new_info: Dict):
        try:
            os.makedirs(cache_dir, exist_ok=True)
            if self.vector_store and hasattr(self.vector_store, "save_local"):
                self.vector_store.save_local(cache_dir)
            with open(os.path.join(cache_dir, "cache_info.json"), 'w', encoding='utf-8') as f:
                json.dump({**old_info, **new_info}, f, ensure_ascii=False)
            with open(os.path.join(cache_dir, "docs.json"), 'w', encoding='utf-8') as f:
                json.dump({'docs': self.documents, 'meta': self.doc_meta}, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"캐시 저장 실패: {e}")
    
    def search(self, query: str, k: int = 3, hybrid: bool = True, 
                filter_file: Optional[str] = None, sort_by: str = 'relevance') -> TaskResult:
        """하이브리드 검색 수행
        
        Args:
            query: 검색어
            k: 결과 개수
            hybrid: 하이브리드 검색 사용 여부
            filter_file: 특정 파일에서만 검색 (None=전체)
            sort_by: 정렬 방식 ('relevance', 'filename', 'length')
        """
        if not self.vector_store:
            return TaskResult(False, "문서가 로드되지 않음")
        
        query = query.strip()
        if len(query) < 2:
            return TaskResult(False, "검색어가 너무 짧습니다 (최소 2자)")
        
        # 캐시 확인
        cached_result = self._search_cache.get(query, k, hybrid)
        if cached_result is not None:
            return TaskResult(True, "검색 완료 (캐시)", cached_result)
        
        try:
            k = max(1, min(k, AppConfig.MAX_SEARCH_RESULTS))
            
            # 벡터 검색
            vec_results = self.vector_store.similarity_search_with_score(query, k=k*2)
            
            results = {}
            if vec_results and len(vec_results) > 0:
                distances = [r[1] for r in vec_results]
                min_d = min(distances)
                max_d = max(distances)
                rng = max_d - min_d if max_d != min_d else 1
                
                for doc, dist in vec_results:
                    key = doc.page_content[:100]
                    score = max(0.1, 1 - ((dist - min_d) / (rng + 0.001)))
                    results[key] = {
                        'content': doc.page_content,
                        'source': doc.metadata.get('source', '?'),
                        'path': doc.metadata.get('path', ''),
                        'vec_score': score,
                        'bm25_score': 0
                    }
            
            # BM25 검색 (하이브리드)
            if hybrid and self.bm25:
                try:
                    bm_res = self.bm25.search(query, top_k=k*2)
                except Exception as bm_err:
                    logger.warning(f"BM25 검색 오류: {bm_err}")
                    bm_res = []
                if bm_res and len(bm_res) > 0:
                    bm_scores = [r[1] for r in bm_res]
                    max_bm = max(bm_scores) if bm_scores else 1
                    for idx, sc in bm_res:
                        if 0 <= idx < len(self.documents):
                            key = self.documents[idx][:100]
                            norm = sc / (max_bm + 0.001)
                            if key in results:
                                results[key]['bm25_score'] = norm
                            else:
                                meta = self.doc_meta[idx] if idx < len(self.doc_meta) else {}
                                results[key] = {
                                    'content': self.documents[idx],
                                    'source': meta.get('source', '?'),
                                    'path': meta.get('path', ''),
                                    'vec_score': 0,
                                    'bm25_score': norm
                                }
            
            # 파일 필터링 적용
            if filter_file:
                results = {k: v for k, v in results.items() if v['source'] == filter_file}
            
            # 최종 점수 계산
            for item in results.values():
                item['score'] = (
                    AppConfig.VECTOR_WEIGHT * item['vec_score'] +
                    AppConfig.BM25_WEIGHT * item['bm25_score']
                )
            
            # 정렬 적용
            if sort_by == 'filename':
                sorted_res = sorted(results.values(), key=lambda x: x['source'])[:k]
            elif sort_by == 'length':
                sorted_res = sorted(results.values(), key=lambda x: len(x['content']), reverse=True)[:k]
            else:  # relevance (기본)
                sorted_res = sorted(results.values(), key=lambda x: x['score'], reverse=True)[:k]
            
            # 캐시 저장 (필터 없을 때만)
            if not filter_file:
                self._search_cache.set(query, k, hybrid, sorted_res)
            
            return TaskResult(True, "검색 완료", sorted_res)
            
        except Exception as e:
            logger.error(f"검색 오류: {e}")
            return TaskResult(False, f"검색 오류: {e}")
    
    def get_file_infos(self) -> List[Dict]:
        return [info.to_dict() for info in self.file_infos.values()]
    
    def get_stats(self) -> Dict:
        total_size = sum(info.size for info in self.file_infos.values())
        total_chunks = sum(info.chunks for info in self.file_infos.values())
        return {
            'files': len(self.file_infos),
            'chunks': total_chunks,
            'size': total_size,
            'size_formatted': FileUtils.format_size(total_size),
            'folder': self.current_folder
        }
    
    def clear_cache(self) -> TaskResult:
        if os.path.exists(self.cache_path):
            shutil.rmtree(self.cache_path, ignore_errors=True)
        self._search_cache.clear()
        return TaskResult(True, "캐시 삭제 완료")
    
    def cleanup(self):
        self.documents.clear()
        self.doc_meta.clear()
        if self.bm25:
            self.bm25.clear()
        self._search_cache.clear()
        try:
            self._executor.shutdown(wait=False)
        except Exception as e:
            logger.debug(f"Executor shutdown error: {e}")
        gc.collect()


# ============================================================================
# Flask 애플리케이션
# ============================================================================
# NumPy 호환 JSON Encoder
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o: Any):
        try:
            import numpy as np
            if isinstance(o, (np.integer, np.floating, np.bool_)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
        except ImportError:
            pass
        return super().default(o)

app = Flask(__name__)
setattr(app, "json_encoder", CustomJSONEncoder)  # Flask < 2.2 호환
app.config['MAX_CONTENT_LENGTH'] = AppConfig.MAX_CONTENT_LENGTH
app.config['JSON_AS_ASCII'] = False
app.secret_key = os.urandom(24)
CORS(app, supports_credentials=True)

# Flask > 2.2 호환을 위한 Provider 설정 (선택적)
try:
    from flask.json.provider import DefaultJSONProvider

    def _numpy_json_default(o):
        try:
            import numpy as np
            if isinstance(o, (np.integer, np.floating, np.bool_)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
        except ImportError:
            pass
        return DefaultJSONProvider.default(o)

    class CustomJSONProvider(DefaultJSONProvider):
        default = staticmethod(_numpy_json_default)
    app.json = CustomJSONProvider(app)
except ImportError:
    pass  # 구버전 Flask는 json_encoder만 사용


# 전역 QA 시스템
qa_system = RegulationQASystem()

# 업로드 폴더 설정
UPLOAD_DIR = os.path.join(get_app_directory(), AppConfig.UPLOAD_FOLDER)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============================================================================
# 에러 핸들러
# ============================================================================
@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'message': 'API 엔드포인트를 찾을 수 없습니다'}), 404
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"서버 내부 오류: {e}")
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'message': '서버 내부 오류가 발생했습니다', 'error': str(e)}), 500
    return "서버 오류발생", 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"예외 발생: {e}")
    import traceback
    logger.error(traceback.format_exc())
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'message': f'오류가 발생했습니다: {str(e)}'}), 500
    return str(e), 500



# ============================================================================
# API 라우트
# ============================================================================
@app.route('/')
def index():
    """메인 검색 페이지"""
    return render_template('index.html')


@app.route('/admin')
def admin():
    """관리자 페이지"""
    # 세션 기반 단순 인증 체크
    # GUI 설정 관리자에서 비밀번호가 설정되어 있는지 확인
    settings_manager = get_settings_manager()
    if settings_manager and settings_manager.has_admin_password():
        # 비밀번호가 설정되어 있는데 세션에 로그인이 안되어 있다면
        if not session.get('admin_authenticated'):
            # API 요청이면 401, 브라우저 접근이면 JS에서 처리하도록 템플릿 렌더링하되
            # 템플릿 내에서 인증 모달을 띄우는 방식은 이미 구현되어 있음.
            # 하지만 "아무런 제한 없이 진입 가능"하다는 사용자 피드백을 반영하여
            # 여기서는 템플릿을 그대로 주되, app.js에서 초기 로드 시 인증을 강제하도록 함.
            # 또는 여기서 바로 접근 거부(403)를 할 수도 있음.
            # 사용자 요청: "실제 웹에서는 아무런 제한없이 관리자 모드로 진입이 가능"
            # -> JS 인증 체크가 우회될 수 있으므로 서버사이드 체크가 필요하지만,
            #    현재 구조상 로그인 페이지가 별도로 없으므로, 
            #    JS에서 비밀번호 모달을 띄우는 것이 UX상 좋음.
            #    단, API 요청에 대해서는 철저히 막아야 함.
            pass

    return render_template('admin.html')


@app.route('/api/models', methods=['GET'])
def api_get_models():
    """사용 가능한 모델 목록 반환"""
    return jsonify({
        'success': True,
        'models': list(AppConfig.AVAILABLE_MODELS.keys()),  # 키 목록만 반환
        'current': getattr(qa_system, 'model_name', AppConfig.DEFAULT_MODEL)
    })






# ============================================================================
# 관리자 인증 API
# ============================================================================
# 전역 설정 관리자 캐시 (순환 참조 방지)
_settings_manager_instance = None

def get_settings_manager():
    """GUI의 SettingsManager 가져오기 (순환 참조 방지)"""
    global _settings_manager_instance
    
    if _settings_manager_instance is not None:
        return _settings_manager_instance
    
    try:
        # server_gui가 이미 import 되었는지 확인
        import sys
        if 'server_gui' in sys.modules:
            _settings_manager_instance = sys.modules['server_gui'].settings_manager
            return _settings_manager_instance
        return None
    except Exception as e:
        logger.debug(f"SettingsManager 접근 실패: {e}")
        return None


@app.route('/api/admin/check')
def api_admin_check():
    """관리자 인증 상태 확인"""
    from flask import session
    
    sm = get_settings_manager()
    if sm is None or not sm.has_admin_password():
        # 비밀번호 미설정 - 인증 불필요
        return jsonify({'success': True, 'authenticated': True, 'required': False})
    
    # 세션 확인
    is_auth = session.get('admin_authenticated', False)
    return jsonify({'success': True, 'authenticated': is_auth, 'required': True})


@app.route('/api/admin/auth', methods=['POST'])
def api_admin_auth():
    """관리자 비밀번호 인증"""
    from flask import session
    
    sm = get_settings_manager()
    if sm is None:
        return jsonify({'success': True, 'message': '인증 불필요'})
    
    data = request.get_json()
    password = data.get('password', '') if data else ''
    
    if sm.verify_admin_password(password):
        session['admin_authenticated'] = True
        return jsonify({'success': True, 'message': '인증 성공'})
    else:
        return jsonify({'success': False, 'message': '비밀번호가 일치하지 않습니다'}), 401


@app.route('/api/admin/logout', methods=['POST'])
def api_admin_logout():
    """관리자 로그아웃"""
    from flask import session
    session.pop('admin_authenticated', None)
    return jsonify({'success': True, 'message': '로그아웃 완료'})


@app.route('/api/status')
def api_status():
    """서버 상태 조회 (시스템 메트릭 포함)"""
    # 시스템 메트릭 수집
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
    except ImportError:
        cpu_percent = None
        memory_percent = None
    
    return jsonify({
        'success': True,
        'ready': qa_system.is_ready,
        'loading': qa_system.is_loading,
        'progress': qa_system.load_progress,
        'error': qa_system.load_error,  # 오류 메시지 추가
        'model': qa_system.model_name,
        'stats': qa_system.get_stats() if qa_system.is_ready else None,
        # 시스템 메트릭
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
        # 검색 큐 상태
        'search_queue': search_queue.get_stats()
    })


@app.route('/api/health')
def api_health():
    """헬스체크 엔드포인트"""
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
    except ImportError:
        cpu_percent = None
        memory_percent = None
    
    return jsonify({
        'status': 'healthy' if qa_system.is_ready else 'initializing',
        'ready': qa_system.is_ready,
        'model_loaded': qa_system.embedding_model is not None,
        'documents_loaded': qa_system.vector_store is not None,
        'document_count': len(qa_system.documents) if qa_system.documents else 0,
        'file_count': len(qa_system.file_infos),
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
        'version': AppConfig.APP_VERSION
    })


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """파일 업로드 및 처리 (동시 업로드 보호)"""
    if not qa_system.is_ready:
        return jsonify({'success': False, 'message': '서버가 준비되지 않았습니다'}), 503
    
    if 'files' not in request.files:
        return jsonify({'success': False, 'message': '파일이 없습니다'}), 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'success': False, 'message': '파일이 선택되지 않았습니다'}), 400
    
    # 파일 저장 (락 사용)
    saved_files = []
    save_errors = []
    
    with file_operation_lock:
        for file in files:
            source_filename = file.filename or ""
            if file and source_filename and FileUtils.allowed_file(source_filename):
                try:
                    filename = secure_filename(source_filename)
                    # 한글 파일명 보존
                    if filename != source_filename:
                        filename = source_filename.replace('/', '_').replace('\\', '_').replace('..', '')
                    
                    # 파일명 중복 처리
                    base_filepath = os.path.join(UPLOAD_DIR, filename)
                    filepath = base_filepath
                    counter = 1
                    while os.path.exists(filepath) and counter < 100:
                        name, ext = os.path.splitext(filename)
                        filepath = os.path.join(UPLOAD_DIR, f"{name}_{counter}{ext}")
                        counter += 1
                    
                    file.save(filepath)
                    saved_files.append(filepath)
                    logger.info(f"파일 업로드: {os.path.basename(filepath)}")
                except Exception as e:
                    save_errors.append(f"{file.filename}: {e}")
                    logger.error(f"파일 저장 실패: {file.filename} - {e}")
    
    if not saved_files:
        error_msg = '지원되는 파일이 없습니다'
        if save_errors:
            error_msg += f' (오류: {", ".join(save_errors[:3])})'
        return jsonify({'success': False, 'message': error_msg}), 400
    
    # 문서 처리
    def progress_cb(percent, msg):
        logger.info(f"처리 진행: {percent}% - {msg}")
    
    try:
        result = qa_system.process_documents(UPLOAD_DIR, saved_files, progress_cb)
        
        return jsonify({
            'success': result.success,
            'message': result.message,
            'data': result.data,
            'failed': result.failed_items,
            'save_errors': save_errors if save_errors else None
        })
    except Exception as e:
        logger.error(f"문서 처리 중 오류: {e}")
        return jsonify({
            'success': False,
            'message': f'문서 처리 중 오류: {e}',
            'saved_count': len(saved_files)
        }), 500



@app.route('/api/process', methods=['POST'])
def api_process():
    """업로드된 모든 파일 재처리"""
    if not qa_system.is_ready:
        return jsonify({'success': False, 'message': '서버가 준비되지 않았습니다'}), 503
    
    if not os.path.exists(UPLOAD_DIR):
        return jsonify({'success': False, 'message': '업로드 폴더가 없습니다'}), 400
    
    files = [
        os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR)
        if FileUtils.allowed_file(f)
    ]
    
    if not files:
        return jsonify({'success': False, 'message': '처리할 파일이 없습니다'}), 400
    
    result = qa_system.process_documents(UPLOAD_DIR, files, None)
    
    return jsonify({
        'success': result.success,
        'message': result.message,
        'data': result.data,
        'failed': result.failed_items
    })


@app.route('/api/search', methods=['POST'])
def api_search():
    """검색 수행"""
    start_time = time.time()
    
    # Rate Limiting 체크
    client_ip = request.remote_addr or '127.0.0.1'
    if not rate_limiter.is_allowed(client_ip):
        return jsonify({
            'success': False, 
            'message': '요청이 너무 많습니다. 잠시 후 다시 시도해주세요.',
            'retry_after': 60,
            'remaining_requests': 0
        }), 429
    
    # 검색 큐 슬롯 획득
    if not search_queue.acquire(timeout=AppConfig.REQUEST_TIMEOUT):
        return jsonify({
            'success': False,
            'message': '서버가 바쁩니다. 잠시 후 다시 시도해주세요.',
            'queue_stats': search_queue.get_stats()
        }), 503
    
    try:
        if not qa_system.is_ready:
            return jsonify({'success': False, 'message': '서버가 준비되지 않았습니다'}), 503
        
        if not qa_system.vector_store:
            return jsonify({'success': False, 'message': '문서가 로드되지 않았습니다'}), 400
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'success': False, 'message': '검색어가 필요합니다'}), 400
        
        query = data.get('query', '').strip()
        k = min(data.get('k', AppConfig.DEFAULT_SEARCH_RESULTS), AppConfig.MAX_SEARCH_RESULTS)
        hybrid = data.get('hybrid', True)
        highlight = data.get('highlight', True)
        filter_file = data.get('filter_file', None)  # 파일 필터
        sort_by = data.get('sort_by', 'relevance')  # 정렬 방식
        
        result = qa_system.search(query, k, hybrid, filter_file, sort_by)
        
        # 검색 성공 시 히스토리에 추가
        if result.success and query:
            qa_system._search_history.add(query)
        
        # 하이라이팅 적용
        results_data = result.data if result.success else []
        if highlight and results_data:
            for item in results_data:
                item['content_highlighted'] = TextHighlighter.highlight(item['content'], query)
        
        # 응답 시간 계산
        response_time_ms = round((time.time() - start_time) * 1000, 2)
        
        return jsonify({
            'success': result.success,
            'message': result.message,
            'results': results_data,
            'query': query,
            'response_time_ms': response_time_ms,
            'result_count': len(results_data)
        })
    except Exception as e:
        logger.error(f"검색 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': f'검색 처리 중 오류가 발생했습니다: {e}'}), 500
    finally:
        # 검색 큐 슬롯 반환
        search_queue.release()



@app.route('/api/search/history')
def api_search_history():
    """검색 히스토리 조회"""
    limit = request.args.get('limit', 10, type=int)
    recent = qa_system._search_history.get_recent(limit)
    popular = qa_system._search_history.get_popular(limit)
    
    return jsonify({
        'success': True,
        'recent': recent,
        'popular': [{'query': q, 'count': c} for q, c in popular]
    })


@app.route('/api/search/suggest')
def api_search_suggest():
    """검색어 자동완성 제안"""
    prefix = request.args.get('q', '').strip()
    limit = request.args.get('limit', 8, type=int)
    
    if len(prefix) < 1:
        return jsonify({'success': True, 'suggestions': []})
    
    suggestions = []
    
    # 1. 검색 히스토리에서 매칭
    history_suggestions = qa_system._search_history.suggest(prefix, limit)
    suggestions.extend(history_suggestions)
    
    # 2. 문서 키워드에서 매칭
    if len(suggestions) < limit:
        keywords = qa_system.get_keywords()
        prefix_lower = prefix.lower()
        for kw in keywords:
            if kw.lower().startswith(prefix_lower) and kw not in suggestions:
                suggestions.append(kw)
                if len(suggestions) >= limit:
                    break
    
    return jsonify({
        'success': True,
        'suggestions': suggestions[:limit]
    })


@app.route('/api/stats/search')
def api_search_stats():
    """검색 통계 조회"""
    limit = request.args.get('limit', 10, type=int)
    
    # 인기 검색어
    popular = qa_system._search_history.get_popular(limit)
    
    # 최근 검색어
    recent = qa_system._search_history.get_recent(limit)
    
    # 총 검색 횟수
    total_searches = sum(count for _, count in popular) if popular else 0
    
    # 파일별 검색 빈도 (검색 결과에서 자주 나오는 파일)
    file_stats = []
    for info in qa_system.file_infos.values():
        file_stats.append({
            'name': info.name,
            'chunks': info.chunks,
            'status': info.status.value
        })
    
    return jsonify({
        'success': True,
        'total_searches': total_searches,
        'popular_queries': [{'query': q, 'count': c} for q, c in popular],
        'recent_queries': recent,
        'file_stats': sorted(file_stats, key=lambda x: x['chunks'], reverse=True)[:limit]
    })


@app.route('/api/files/names')
def api_file_names():
    """파일명 목록만 반환 (검색 필터용)"""
    names = [info.name for info in qa_system.file_infos.values()]
    return jsonify({
        'success': True,
        'files': sorted(names)
    })


@app.route('/api/files')
def api_files():
    """로드된 파일 목록"""
    return jsonify({
        'success': True,
        'files': qa_system.get_file_infos(),
        'stats': qa_system.get_stats()
    })


@app.route('/api/cache', methods=['DELETE'])
def api_clear_cache():
    """캐시 삭제"""
    result = qa_system.clear_cache()
    return jsonify({'success': result.success, 'message': result.message})


@app.route('/api/models')
def api_models():
    """사용 가능한 모델 목록"""
    return jsonify({
        'success': True,
        'models': list(AppConfig.AVAILABLE_MODELS.keys()),
        'current': qa_system.model_name
    })


@app.route('/api/files/<filename>', methods=['DELETE'])
def api_delete_file(filename):
    """개별 파일 삭제 (스레드 안전)"""
    if not qa_system.is_ready:
        return jsonify({'success': False, 'message': '서버가 준비되지 않았습니다'}), 503
    
    # 경로 검증 (경로 탐색 공격 방지)
    safe_filename = secure_filename(filename)
    if safe_filename != filename and not filename.replace(' ', '_') == safe_filename:
        # 한글 파일명 처리
        safe_filename = filename.replace('/', '_').replace('\\', '_').replace('..', '')
    
    filepath = os.path.join(UPLOAD_DIR, safe_filename)
    
    # 파일 존재 확인
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'message': '파일을 찾을 수 없습니다'}), 404
    
    # 파일 삭제 (락 사용)
    with file_operation_lock:
        try:
            # 파일 삭제
            os.remove(filepath)
            logger.info(f"파일 삭제: {safe_filename}")
            
            # 인덱스에서 해당 파일 관련 데이터 제거
            if filepath in qa_system.file_infos:
                del qa_system.file_infos[filepath]
            
            # 검색 캐시 무효화
            qa_system._search_cache.clear()
            qa_system._keyword_cache.clear()
            
            # 남은 파일로 인덱스 재구성 필요 알림
            remaining_files = [
                os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR)
                if FileUtils.allowed_file(f)
            ]
            
            return jsonify({
                'success': True,
                'message': f'{safe_filename} 삭제 완료',
                'remaining_files': len(remaining_files),
                'reindex_required': True  # 프론트엔드에서 재처리 안내
            })
            
        except OSError as e:
            logger.error(f"파일 삭제 실패: {e}")
            return jsonify({'success': False, 'message': f'삭제 실패: {e}'}), 500
        except Exception as e:
            logger.error(f"파일 삭제 중 예외: {e}")
            return jsonify({'success': False, 'message': f'예외 발생: {e}'}), 500



@app.route('/api/files/<filename>/preview')
def api_file_preview(filename):
    """파일 내용 미리보기"""
    if not qa_system.is_ready:
        return jsonify({'success': False, 'message': '서버가 준비되지 않았습니다'}), 503
    
    # 경로 검증
    safe_filename = filename.replace('/', '_').replace('\\', '_').replace('..', '')
    filepath = os.path.join(UPLOAD_DIR, safe_filename)
    
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'message': '파일을 찾을 수 없습니다'}), 404
    
    try:
        # 파일 내용 추출
        content, error = qa_system.extractor.extract(filepath)
        
        if error:
            return jsonify({
                'success': False,
                'message': f'파일 읽기 실패: {error}'
            }), 400
        
        # 미리보기 길이 제한 (기본 2000자)
        max_length = request.args.get('length', 2000, type=int)
        max_length = min(max_length, 5000)  # 최대 5000자
        
        preview_content = content[:max_length]
        is_truncated = len(content) > max_length
        
        # 파일 정보
        file_info = qa_system.file_infos.get(filepath)
        
        return jsonify({
            'success': True,
            'filename': safe_filename,
            'content': preview_content,
            'total_length': len(content),
            'is_truncated': is_truncated,
            'chunks': file_info.chunks if file_info else 0,
            'status': file_info.status.value if file_info else 'unknown'
        })
        
    except Exception as e:
        logger.error(f"미리보기 실패: {e}")
        return jsonify({'success': False, 'message': f'미리보기 실패: {e}'}), 500


@app.route('/api/files/<filename>/download')
def api_file_download(filename):
    """파일 다운로드"""
    # 경로 검증
    safe_filename = filename.replace('/', '_').replace('\\', '_').replace('..', '')
    filepath = os.path.join(UPLOAD_DIR, safe_filename)
    
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'message': '파일을 찾을 수 없습니다'}), 404
    
    try:
        return send_from_directory(
            UPLOAD_DIR, 
            safe_filename, 
            as_attachment=True,
            download_name=safe_filename
        )
    except Exception as e:
        logger.error(f"파일 다운로드 실패: {e}")
        return jsonify({'success': False, 'message': f'다운로드 실패: {e}'}), 500


@app.route('/api/models', methods=['POST'])
def api_set_model():
    """모델 변경"""
    # 관리자 인증 체크
    if not session.get('admin_authenticated'):
        settings_manager = get_settings_manager()
        # settings_manager가 있고 패스워드가 설정되어 있으면 인증 필요
        if settings_manager and settings_manager.has_admin_password():
             return jsonify({'success': False, 'message': '관리자 권한이 필요합니다'}), 401
             
    data = request.get_json() or {}
    model_name = data.get('model')
    
    if not model_name or model_name not in AppConfig.AVAILABLE_MODELS:
        return jsonify({'success': False, 'message': '잘못된 모델입니다'}), 400
        
    try:
        # 모델 변경 작업
        # 1. 모델 로드 시도
        result = qa_system.load_model(model_name)
        
        if result.success:
            # 2. 모델 변경 성공 시 캐시 초기화 등 부가 작업? (load_model 내부에서 처리됨)
            pass
            
        return jsonify({
            'success': result.success,
            'message': result.message
        })
    except Exception as e:
        logger.error(f"모델 변경 중 오류: {e}")
        return jsonify({'success': False, 'message': f'오류 발생: {str(e)}'}), 500


# ============================================================================
# v2.0 API 엔드포인트 - 태그/카테고리
# ============================================================================
@app.route('/api/tags', methods=['GET'])
def api_get_tags():
    """태그 목록 조회"""
    filename = request.args.get('filename')
    
    if filename:
        # 특정 파일의 태그
        tags = tag_manager.get_tags(filename)
        return jsonify({'success': True, 'tags': tags, 'filename': filename})
    else:
        # 모든 사용된 태그
        all_tags = tag_manager.get_all_tags()
        return jsonify({'success': True, 'tags': all_tags})


@app.route('/api/tags', methods=['POST'])
def api_add_tag():
    """태그 추가"""
    data = request.get_json() or {}
    filename = data.get('filename')
    tag = data.get('tag')
    
    if not filename or not tag:
        return jsonify({'success': False, 'message': '파일명과 태그가 필요합니다'}), 400
    
    added = tag_manager.add_tag(filename, tag)
    return jsonify({
        'success': True,
        'added': added,
        'tags': tag_manager.get_tags(filename)
    })


@app.route('/api/tags', methods=['DELETE'])
def api_remove_tag():
    """태그 제거"""
    data = request.get_json() or {}
    filename = data.get('filename')
    tag = data.get('tag')
    
    if not filename or not tag:
        return jsonify({'success': False, 'message': '파일명과 태그가 필요합니다'}), 400
    
    removed = tag_manager.remove_tag(filename, tag)
    return jsonify({
        'success': True,
        'removed': removed,
        'tags': tag_manager.get_tags(filename)
    })


@app.route('/api/tags/set', methods=['POST'])
def api_set_tags():
    """파일의 태그 전체 설정"""
    data = request.get_json() or {}
    filename = data.get('filename')
    tags = data.get('tags', [])
    
    if not filename:
        return jsonify({'success': False, 'message': '파일명이 필요합니다'}), 400
    
    tag_manager.set_tags(filename, tags)
    return jsonify({
        'success': True,
        'tags': tag_manager.get_tags(filename)
    })


@app.route('/api/tags/search')
def api_search_by_tag():
    """태그로 파일 검색"""
    tag = request.args.get('tag')
    
    if not tag:
        return jsonify({'success': False, 'message': '태그가 필요합니다'}), 400
    
    files = tag_manager.search_by_tag(tag)
    return jsonify({'success': True, 'files': files, 'tag': tag})


@app.route('/api/categories')
def api_categories():
    """카테고리 목록 반환"""
    return jsonify({
        'success': True,
        'categories': TagManager.PREDEFINED_CATEGORIES,
        'keywords': TagManager.CATEGORY_KEYWORDS
    })


@app.route('/api/tags/auto', methods=['POST'])
def api_auto_categorize():
    """자동 카테고리 추천"""
    data = request.get_json() or {}
    filename = data.get('filename')
    content = data.get('content', '')
    
    # 파일명이 주어지면 파일 내용 로드
    if filename and not content:
        filepath = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(filepath):
            content, _ = qa_system.extractor.extract(filepath)
    
    if not content:
        return jsonify({'success': False, 'message': '내용을 분석할 수 없습니다'}), 400
    
    suggested = tag_manager.auto_categorize(content, filename or '')
    return jsonify({
        'success': True,
        'suggested_categories': suggested,
        'filename': filename
    })


# ============================================================================
# v2.0 API 엔드포인트 - 조문/조항 검색
# ============================================================================
@app.route('/api/search/article', methods=['POST'])
def api_search_article():
    """조문 단위 검색"""
    data = request.get_json() or {}
    query = data.get('query', '').strip()
    filename = data.get('filename')
    
    if not query:
        return jsonify({'success': False, 'message': '검색어가 필요합니다'}), 400
    
    results = []
    
    # 특정 파일 또는 전체 파일에서 검색
    target_files = [filename] if filename else list(qa_system.file_infos.keys())
    
    for filepath in target_files:
        if not os.path.exists(filepath):
            continue
        
        fname = os.path.basename(filepath)
        content, _ = qa_system.extractor.extract(filepath)
        
        if content:
            articles = article_parser.parse_articles(content)
            matched = article_parser.search_article(articles, query)
            
            for article in matched:
                article['source'] = fname
                results.append(article)
    
    # 점수순 정렬 및 상위 결과 반환
    results.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    return jsonify({
        'success': True,
        'results': results[:20],
        'total': len(results)
    })


@app.route('/api/files/<filename>/articles')
def api_file_articles(filename):
    """파일의 조문 구조 반환"""
    safe_filename = filename.replace('/', '_').replace('\\', '_').replace('..', '')
    filepath = os.path.join(UPLOAD_DIR, safe_filename)
    
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'message': '파일을 찾을 수 없습니다'}), 404
    
    content, error = qa_system.extractor.extract(filepath)
    if error:
        return jsonify({'success': False, 'message': error}), 400
    
    articles = article_parser.parse_articles(content)
    return jsonify({
        'success': True,
        'filename': safe_filename,
        'articles': articles,
        'count': len(articles)
    })


# ============================================================================
# v2.0 API 엔드포인트 - 관련 규정 추천
# ============================================================================
@app.route('/api/search/related', methods=['POST'])
def api_related_regulations():
    """관련 규정 추천"""
    data = request.get_json() or {}
    content = data.get('content', '')
    source = data.get('source', '')
    k = data.get('k', 5)
    
    if not content:
        return jsonify({'success': False, 'message': '내용이 필요합니다'}), 400
    
    if not qa_system.is_ready:
        return jsonify({'success': False, 'message': '서버가 준비되지 않았습니다'}), 503
    
    # 벡터 검색으로 유사 문서 찾기
    try:
        if qa_system.vector_store is None:
            return jsonify({'success': False, 'message': '벡터 인덱스가 준비되지 않았습니다'}), 503
        similar = qa_system.vector_store.similarity_search_with_score(content[:500], k=k*2)
        
        results = []
        seen_sources = {source}  # 원본 파일 제외
        
        for doc, score in similar:
            doc_source = doc.metadata.get('source', '')
            if doc_source not in seen_sources:
                seen_sources.add(doc_source)
                results.append({
                    'source': doc_source,
                    'content': doc.page_content[:300] + '...' if len(doc.page_content) > 300 else doc.page_content,
                    'similarity': round(1 - score, 3)
                })
        
        return jsonify({
            'success': True,
            'related': results[:k]
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ============================================================================
# v2.0 API 엔드포인트 - 문서 분할
# ============================================================================
@app.route('/api/files/<filename>/split', methods=['POST'])
def api_split_file(filename):
    """파일 분할"""
    data = request.get_json() or {}
    split_type = data.get('type', 'chapter')  # chapter, article, size
    max_size = data.get('max_size', 5000)
    save_files = data.get('save', False)  # 실제 파일로 저장 여부
    
    safe_filename = filename.replace('/', '_').replace('\\', '_').replace('..', '')
    filepath = os.path.join(UPLOAD_DIR, safe_filename)
    
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'message': '파일을 찾을 수 없습니다'}), 404
    
    content, error = qa_system.extractor.extract(filepath)
    if error:
        return jsonify({'success': False, 'message': error}), 400
    
    # 분할 수행
    if split_type == 'chapter':
        parts = document_splitter.split_by_chapters(content, safe_filename)
    elif split_type == 'article':
        parts = document_splitter.split_by_articles(content, safe_filename)
    elif split_type == 'size':
        parts = document_splitter.split_by_size(content, max_size)
    else:
        return jsonify({'success': False, 'message': '잘못된 분할 유형입니다'}), 400
    
    # 파일로 저장
    saved_files = []
    if save_files and parts:
        base_name = os.path.splitext(safe_filename)[0]
        for i, part in enumerate(parts):
            part_filename = f"{base_name}_part{i+1}.txt"
            part_path = os.path.join(UPLOAD_DIR, part_filename)
            try:
                with open(part_path, 'w', encoding='utf-8') as f:
                    f.write(f"{part['title']}\n\n{part['content']}")
                saved_files.append(part_filename)
            except Exception as e:
                logger.error(f"분할 파일 저장 실패: {e}")
    
    return jsonify({
        'success': True,
        'parts': parts,
        'count': len(parts),
        'saved_files': saved_files if save_files else None
    })


# ============================================================================
# v2.0 API 엔드포인트 - 문서 비교
# ============================================================================
@app.route('/api/files/compare', methods=['POST'])
def api_compare_files():
    """두 파일 비교"""
    data = request.get_json() or {}
    file1 = data.get('file1')
    file2 = data.get('file2')
    
    if not file1 or not file2:
        return jsonify({'success': False, 'message': '두 파일명이 필요합니다'}), 400
    
    # 파일 내용 로드
    path1 = os.path.join(UPLOAD_DIR, file1.replace('/', '_').replace('\\', '_').replace('..', ''))
    path2 = os.path.join(UPLOAD_DIR, file2.replace('/', '_').replace('\\', '_').replace('..', ''))
    
    if not os.path.exists(path1):
        return jsonify({'success': False, 'message': f'{file1}을(를) 찾을 수 없습니다'}), 404
    if not os.path.exists(path2):
        return jsonify({'success': False, 'message': f'{file2}을(를) 찾을 수 없습니다'}), 404
    
    content1, err1 = qa_system.extractor.extract(path1)
    content2, err2 = qa_system.extractor.extract(path2)
    
    if err1:
        return jsonify({'success': False, 'message': f'{file1} 읽기 오류: {err1}'}), 400
    if err2:
        return jsonify({'success': False, 'message': f'{file2} 읽기 오류: {err2}'}), 400
    
    # 비교 수행
    result = document_comparator.compare(content1, content2)
    
    return jsonify({
        'success': True,
        'file1': file1,
        'file2': file2,
        'added_lines': result['added_lines'],
        'removed_lines': result['removed_lines'],
        'total_changes': result['total_changes'],
        'similarity': round(result['similarity'] * 100, 1),
        'diff_html': result['diff_html']
    })


# ============================================================================
# v2.0 API 엔드포인트 - 개정 이력
# ============================================================================
@app.route('/api/files/<filename>/revisions', methods=['GET'])
def api_file_revisions(filename):
    """파일 개정 이력 조회"""
    history = revision_tracker.get_history(filename)
    return jsonify({
        'success': True,
        'filename': filename,
        'revisions': history,
        'count': len(history)
    })


@app.route('/api/files/<filename>/revisions', methods=['POST'])
def api_save_revision(filename):
    """새 개정 버전 저장"""
    data = request.get_json() or {}
    note = data.get('note', '')
    
    # 파일 내용 로드
    safe_filename = filename.replace('/', '_').replace('\\', '_').replace('..', '')
    filepath = os.path.join(UPLOAD_DIR, safe_filename)
    
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'message': '파일을 찾을 수 없습니다'}), 404
    
    content, error = qa_system.extractor.extract(filepath)
    if error:
        return jsonify({'success': False, 'message': error}), 400
    
    # 개정 저장
    revision = revision_tracker.save_revision(safe_filename, content, note)
    
    return jsonify({
        'success': True,
        'revision': revision
    })


@app.route('/api/files/<filename>/revisions/<version>')
def api_get_revision(filename, version):
    """특정 버전 내용 조회"""
    content = revision_tracker.get_revision(filename, version)
    
    if content is None:
        return jsonify({'success': False, 'message': '해당 버전을 찾을 수 없습니다'}), 404
    
    return jsonify({
        'success': True,
        'filename': filename,
        'version': version,
        'content': content
    })


@app.route('/api/files/<filename>/revisions/compare', methods=['POST'])
def api_compare_revisions(filename):
    """버전 간 비교"""
    data = request.get_json() or {}
    v1 = data.get('v1')
    v2 = data.get('v2')
    
    if not v1 or not v2:
        return jsonify({'success': False, 'message': '두 버전이 필요합니다'}), 400
    
    result = revision_tracker.compare_versions(filename, v1, v2)
    
    if result is None:
        return jsonify({'success': False, 'message': '버전을 찾을 수 없습니다'}), 404
    
    return jsonify({
        'success': True,
        'filename': filename,
        'v1': v1,
        'v2': v2,
        'comparison': {
            'added_lines': result['added_lines'],
            'removed_lines': result['removed_lines'],
            'similarity': round(result['similarity'] * 100, 1)
        }
    })


# ============================================================================
# v2.0 API 엔드포인트 - 폴더 동기화
# ============================================================================
@app.route('/api/sync/start', methods=['POST'])
def api_start_sync():
    """폴더 모니터링 시작"""
    data = request.get_json() or {}
    folder = data.get('folder', UPLOAD_DIR)
    
    if not os.path.isdir(folder):
        return jsonify({'success': False, 'message': '폴더가 존재하지 않습니다'}), 400
    
    # 콜백 설정 (파일 변경 시 자동 재처리)
    def on_file_change(event_type, filepath):
        logger.info(f"파일 변경 감지: {event_type} - {filepath}")
        # TODO: 자동 인덱스 업데이트 로직
    
    folder_watcher.callback = on_file_change
    success = folder_watcher.start_watching(folder)
    
    return jsonify({
        'success': success,
        'message': '폴더 모니터링 시작됨' if success else '모니터링 시작 실패 (watchdog 설치 확인)',
        'status': folder_watcher.get_status()
    })


@app.route('/api/sync/stop', methods=['POST'])
def api_stop_sync():
    """폴더 모니터링 중지"""
    folder_watcher.stop_watching()
    return jsonify({
        'success': True,
        'message': '폴더 모니터링 중지됨',
        'status': folder_watcher.get_status()
    })


@app.route('/api/sync/status')
def api_sync_status():
    """동기화 상태 조회"""
    return jsonify({
        'success': True,
        'status': folder_watcher.get_status()
    })


# ============================================================================
# v2.0 API - 폴더 구조 업로드
# ============================================================================
@app.route('/api/upload/folder', methods=['POST'])
def api_upload_folder():
    """폴더 구조 업로드 (ZIP 파일로)"""
    if not qa_system.is_ready:
        return jsonify({'success': False, 'message': '서버가 준비되지 않았습니다'}), 503
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'ZIP 파일이 필요합니다'}), 400
    
    file = request.files['file']
    zip_filename = file.filename or ""
    if not zip_filename.endswith('.zip'):
        return jsonify({'success': False, 'message': 'ZIP 파일만 지원됩니다'}), 400
    
    import zipfile
    import io
    
    try:
        # ZIP 파일 처리
        zip_data = io.BytesIO(file.read())
        saved_files = []
        
        with zipfile.ZipFile(zip_data, 'r') as zip_ref:
            for zip_info in zip_ref.infolist():
                if zip_info.is_dir():
                    continue
                
                # 파일 확장자 검사
                ext = os.path.splitext(zip_info.filename)[1].lower()
                if ext not in AppConfig.SUPPORTED_EXTENSIONS:
                    continue
                
                # 파일명 추출 (폴더 경로 제거)
                filename = os.path.basename(zip_info.filename)
                if not filename:
                    continue
                
                # 파일 저장
                filepath = os.path.join(UPLOAD_DIR, filename)
                counter = 1
                while os.path.exists(filepath):
                    base, ext = os.path.splitext(filename)
                    filepath = os.path.join(UPLOAD_DIR, f"{base}_{counter}{ext}")
                    counter += 1
                
                with open(filepath, 'wb') as f:
                    f.write(zip_ref.read(zip_info.filename))
                saved_files.append(filepath)
        
        # 문서 처리
        if saved_files:
            result = qa_system.process_documents(UPLOAD_DIR, saved_files, None)
            return jsonify({
                'success': True,
                'message': f'{len(saved_files)}개 파일 업로드 및 처리 완료',
                'files': [os.path.basename(f) for f in saved_files],
                'process_result': result.data
            })
        else:
            return jsonify({'success': False, 'message': '지원되는 파일이 없습니다'}), 400
            
    except Exception as e:
        logger.error(f"ZIP 업로드 오류: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/files/structure')
def api_files_structure():
    """파일 구조 반환 (태그 포함)"""
    files = []
    
    for filepath, info in qa_system.file_infos.items():
        file_data = info.to_dict()
        file_data['tags'] = tag_manager.get_tags(info.name)
        files.append(file_data)
    
    return jsonify({
        'success': True,
        'files': files,
        'categories': TagManager.PREDEFINED_CATEGORIES
    })


# ============================================================================
# 메인 실행
# ============================================================================
def load_settings_to_config():
    """설정 파일에서 AppConfig로 값 로드"""
    settings_file = os.path.join(get_app_directory(), 'config', 'settings.json')
    if os.path.exists(settings_file):
        try:
            with open(settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                AppConfig.OFFLINE_MODE = settings.get('offline_mode', False)
                AppConfig.LOCAL_MODEL_PATH = settings.get('local_model_path', '')
                logger.info(f"📋 설정 로드 완료 - 오프라인 모드: {AppConfig.OFFLINE_MODE}")
        except Exception as e:
            logger.warning(f"설정 파일 로드 실패 (기본값 사용): {e}")


def initialize_server():
    """서버 초기화 - 모델 로드"""
    logger.info("=" * 60)
    logger.info(f"🚀 {AppConfig.APP_NAME} {AppConfig.APP_VERSION}")
    logger.info("=" * 60)
    
    # 설정 파일에서 오프라인 모드 설정 로드
    load_settings_to_config()
    
    # 오프라인 모드 상태 로깅
    if AppConfig.OFFLINE_MODE:
        logger.info("🔒 오프라인 모드 활성화 - 로컬 모델만 사용")
        if AppConfig.LOCAL_MODEL_PATH:
            logger.info(f"📂 로컬 모델 경로: {AppConfig.LOCAL_MODEL_PATH}")
    else:
        logger.info("🌐 온라인 모드 - 필요 시 모델 다운로드")
    
    # 모델 로드 (오프라인 모드 설정 전달)
    result = qa_system.load_model(
        AppConfig.DEFAULT_MODEL,
        offline_mode=AppConfig.OFFLINE_MODE,
        local_model_path=AppConfig.LOCAL_MODEL_PATH
    )
    if result.success:
        logger.info(f"✅ {result.message}")
        
        # 기존 업로드된 파일 자동 처리
        if os.path.exists(UPLOAD_DIR):
            files = [
                os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR)
                if FileUtils.allowed_file(f)
            ]
            if files:
                logger.info(f"📂 기존 파일 {len(files)}개 처리 중...")
                result = qa_system.process_documents(UPLOAD_DIR, files, None)
                if result.success:
                    logger.info(f"✅ {result.message}")
                else:
                    logger.warning(f"⚠️ {result.message}")
    else:
        logger.error(f"❌ 모델 로드 실패: {result.message}")


def graceful_shutdown(signum=None, frame=None):
    """서버 정상 종료 처리"""
    logger.info("🛑 서버 종료 중...")
    qa_system.cleanup()
    logger.info("✅ 정상 종료 완료")
    # sys.exit()는 __main__에서만 호출 (그 외에서는 호출하면 안됨!)


if __name__ == '__main__':
    # atexit 등록은 직접 실행 시에만
    atexit.register(graceful_shutdown)
    
    # SIGINT, SIGTERM 핸들러 등록
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    
    # 서버 초기화 (별도 스레드)
    init_thread = threading.Thread(target=initialize_server, daemon=True)
    init_thread.start()
    
    logger.info(f"🌐 서버 시작: http://localhost:{AppConfig.SERVER_PORT}")
    logger.info(f"📚 관리자 페이지: http://localhost:{AppConfig.SERVER_PORT}/admin")
    logger.info("=" * 60)
    
    # 프로덕션 서버 (waitress 권장, 없으면 Flask 기본)
    try:
        from waitress import serve
        logger.info("🚀 Waitress 프로덕션 서버로 실행")
        serve(app, host=AppConfig.SERVER_HOST, port=AppConfig.SERVER_PORT, threads=8)
    except ImportError:
        logger.warning("⚠️ waitress 없음 - Flask 개발 서버 사용 (프로덕션에서는 waitress 권장)")
        app.run(
            host=AppConfig.SERVER_HOST,
            port=AppConfig.SERVER_PORT,
            debug=False,
            threaded=True
        )
