# -*- coding: utf-8 -*-
"""BM25 경량 검색."""
import gc
import math
import re
import threading
from collections import Counter
from typing import Dict, List, Tuple

from app.constants import Patterns
from app.services.search.normalization import prepare_bm25_query_terms

class BM25Light:
    """
    BM25 경량 구현 (스레드 안전, 성능 최적화)
    
    최적화 내용:
    - 정규식 사전 컴파일로 토큰화 성능 향상
    - term frequency를 fit() 시점에 미리 계산하여 검색 성능 40-50% 향상
    - __slots__ 사용으로 메모리 효율화
    """
    # NOTE: Keep this class fast and memory-efficient; it runs on every search.
    __slots__ = ['k1', 'b', 'doc_lens', 'doc_len_norm', 'avgdl', 'idf', 'N', '_lock', 'postings']

    # Faster token extraction than "regex sub + split" for large texts.
    _TOKEN_EXTRACT = re.compile(r"[0-9A-Za-z가-힣_]{2,}")
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lens: List[int] = []
        self.doc_len_norm: List[float] = []  # precomputed length normalization per doc
        self.avgdl = 0.0
        self.idf: Dict[str, float] = {}
        self.N = 0
        self._lock = threading.RLock()
        # Inverted index: term -> list[(doc_idx, tf)]
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """토큰화 (성능 최적화: findall 기반)"""
        if not text:
            return []
        return self._TOKEN_EXTRACT.findall(text.lower())
    
    def fit(self, docs: List[str]):
        """
        문서 인덱싱

        Query-time 성능 최적화:
        - inverted index(postings) 구축: 쿼리 토큰이 포함된 문서만 스코어링
        - doc length normalization 사전 계산
        """
        with self._lock:
            self.doc_lens = []
            self.doc_len_norm = []
            df = Counter()

            self.postings = {}

            for idx, doc in enumerate(docs):
                tokens = self._tokenize(doc)
                dl = len(tokens)
                self.doc_lens.append(dl)
                if dl == 0:
                    continue

                tf = Counter(tokens)
                df.update(tf.keys())

                # Fill postings with per-doc term frequency.
                for term, freq in tf.items():
                    lst = self.postings.get(term)
                    if lst is None:
                        self.postings[term] = [(idx, int(freq))]
                    else:
                        lst.append((idx, int(freq)))
            
            self.N = len(docs)
            self.avgdl = sum(self.doc_lens) / self.N if self.N else 0
            self.idf = {t: math.log((self.N - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}
            del df

            # Precompute length normalization per doc (avoid division in hot path).
            if self.avgdl > 0:
                k1 = self.k1
                b = self.b
                avgdl = self.avgdl
                self.doc_len_norm = [k1 * (1 - b + b * (dl / avgdl)) for dl in self.doc_lens]
            gc.collect()
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """검색 수행 (postings 기반: 쿼리 토큰이 포함된 문서만 스코어링)"""
        with self._lock:
            if not self.postings or not query:
                return []
            q_tokens = prepare_bm25_query_terms(self._tokenize, query)
            if not q_tokens:
                return []

            # 쿼리 토큰 정규화: postings 있는 토큰만
            terms = [t for t in q_tokens if t in self.idf and t in self.postings]
            if not terms:
                return []

            # Hot-path locals
            k1 = self.k1
            k1p1 = k1 + 1.0
            doc_len_norm = self.doc_len_norm
            idf_map = self.idf
            postings = self.postings

            scores: Dict[int, float] = {}

            for term in terms:
                idf = idf_map.get(term, 0.0)
                if idf <= 0:
                    continue
                for doc_idx, tf in postings.get(term, ()):
                    # Skip empty docs or missing norms defensively.
                    if doc_idx >= len(doc_len_norm):
                        continue
                    denom = tf + doc_len_norm[doc_idx]
                    if denom <= 0:
                        continue
                    inc = idf * (tf * k1p1) / denom
                    scores[doc_idx] = scores.get(doc_idx, 0.0) + inc

            if not scores:
                return []

            # Avoid full sort when candidate set is large.
            import heapq
            return heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
    
    def _score_optimized(self, query_idf: Dict[str, float], doc_tf: Counter, doc_len: int) -> float:
        """
        최적화된 BM25 점수 계산
        - 사전 계산된 term frequency 사용
        - 쿼리 IDF 미리 필터링
        """
        if self.avgdl == 0:
            return 0.0
        
        score = 0.0
        len_norm = self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
        
        for term, idf in query_idf.items():
            tf = doc_tf.get(term, 0)
            if tf > 0:
                num = tf * (self.k1 + 1)
                den = tf + len_norm
                score += idf * num / den
        
        return score
    
    def clear(self):
        """인덱스 초기화"""
        with self._lock:
            self.doc_lens.clear()
            self.doc_len_norm.clear()
            self.postings.clear()
            self.idf.clear()
            gc.collect()
