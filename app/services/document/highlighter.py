# -*- coding: utf-8 -*-
"""검색어 하이라이트."""
import re
from typing import List

from app.services.document.patterns import _RE_KEYWORD_EXTRACT

class TextHighlighter:
    # 정규식 패턴 캐시 (v2.6.1 성능 최적화)
    _pattern_cache = {}
    _query_pattern_cache = {}
    _CACHE_MAX_SIZE = 100
    
    @classmethod
    def _get_cached_pattern(cls, keyword: str):
        """캐싱된 정규식 패턴 반환"""
        if keyword not in cls._pattern_cache:
            # 캐시 크기 제한
            if len(cls._pattern_cache) >= cls._CACHE_MAX_SIZE:
                # 오래된 항목 절반 제거
                keys_to_remove = list(cls._pattern_cache.keys())[:cls._CACHE_MAX_SIZE // 2]
                for key in keys_to_remove:
                    del cls._pattern_cache[key]
            cls._pattern_cache[keyword] = re.compile(re.escape(keyword), re.IGNORECASE)
        return cls._pattern_cache[keyword]

    @classmethod
    def _get_cached_query_pattern(cls, query: str):
        """쿼리 전체에 대한 단일 정규식 패턴 캐시"""
        keywords = sorted(
            {kw.strip() for kw in str(query or '').split() if len(kw.strip()) >= 2},
            key=len,
            reverse=True
        )
        if not keywords:
            return None

        cache_key = "\x1f".join(keywords)
        if cache_key not in cls._query_pattern_cache:
            if len(cls._query_pattern_cache) >= cls._CACHE_MAX_SIZE:
                # 오래된 항목 절반 제거
                keys_to_remove = list(cls._query_pattern_cache.keys())[:cls._CACHE_MAX_SIZE // 2]
                for key in keys_to_remove:
                    del cls._query_pattern_cache[key]
            alternation = "|".join(re.escape(kw) for kw in keywords)
            cls._query_pattern_cache[cache_key] = re.compile(f"({alternation})", re.IGNORECASE)
        return cls._query_pattern_cache[cache_key]
    
    @staticmethod
    def highlight(text: str, query: str, tag: str = 'mark') -> str:
        """검색어 하이라이트 (성능 최적화: 쿼리 단일 패턴 + 캐싱)"""
        if not text or not query:
            return text

        pattern = TextHighlighter._get_cached_query_pattern(query)
        if pattern is None:
            return text

        return pattern.sub(f'<{tag}>\\g<0></{tag}>', text)
    
    @staticmethod
    def extract_keywords(documents: List[str], top_k: int = 50) -> List[str]:
        if not documents:
            return []
        
        word_freq = Counter()
        for doc in documents:
            # 사전 컴파일된 패턴 사용 (성능 최적화)
            words = _RE_KEYWORD_EXTRACT.findall(doc)
            word_freq.update(words)
        
        stopwords = {'있는', '하는', '및', '등', '이', '가', '을', '를', '의', '에', '로', '으로'}
        keywords = [w for w, _ in word_freq.most_common(top_k * 2) if w not in stopwords]
        
        return keywords[:top_k]
