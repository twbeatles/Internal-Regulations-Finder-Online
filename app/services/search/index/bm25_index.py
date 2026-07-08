# -*- coding: utf-8 -*-
"""BM25 키워드 인덱스 구축."""
from __future__ import annotations

from app.services.search.bm25 import BM25Light


class Bm25IndexService:
    def build(self, qa) -> None:
        if qa.documents:
            qa.bm25 = BM25Light()
            qa.bm25.fit(qa.documents)
        else:
            if qa.bm25:
                qa.bm25.clear()
            qa.bm25 = None


bm25_index_service = Bm25IndexService()