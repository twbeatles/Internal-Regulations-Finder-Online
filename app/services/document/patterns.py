# -*- coding: utf-8 -*-
"""문서 처리 공통 정규식 패턴."""
import re

_RE_ARTICLE_SPLIT = re.compile(r'(제\s*\d+\s*조(?:의\s*\d+)?[^\n]*)')
_RE_ARTICLE_MATCH = re.compile(r'제\s*(\d+)\s*조(?:의\s*(\d+))?(.+)?')
_RE_PARAGRAPH_SPLIT = re.compile(r'([①②③④⑤⑥⑦⑧⑨⑩])')
_RE_NUMBER_EXTRACT = re.compile(r'\d+')
_RE_CHAPTER_SPLIT = re.compile(r'(제\s*\d+\s*장[^\n]*)')
_RE_ARTICLE_SPLIT_FULL = re.compile(r'(제\s*\d+\s*조(?:의\s*\d+)?[^\n]*)')
_RE_KEYWORD_EXTRACT = re.compile(r'[가-힣]{2,}|[a-zA-Z]{3,}')
