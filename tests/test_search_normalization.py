# -*- coding: utf-8 -*-
"""한국어 검색 정규화 단위 테스트."""

from __future__ import annotations

from app.services.search.bm25 import BM25Light
from app.services.search.normalization import (
    canonicalize_search_text,
    expand_search_text_variants,
    normalize_search_text,
    prepare_vector_query,
)


def test_canonicalize_article_spacing() -> None:
    assert canonicalize_search_text("제 27 조") == "제27조"
    assert canonicalize_search_text("제 3 의 2 조") == "제3의2조"


def test_canonicalize_money_korean_digit() -> None:
    assert canonicalize_search_text("3 천만원") == "3000만원"
    assert "3000만원" in expand_search_text_variants("3 천만원")


def test_normalize_search_text_strips_noise_terms() -> None:
    variants = expand_search_text_variants("승진 최저 소요 연수 알려줘")
    joined = " ".join(variants)
    assert "승진" in joined
    assert any("최저소요연수" in v.replace(" ", "") for v in variants)


def test_promotion_minimum_years_synonym() -> None:
    normalized = normalize_search_text("최소 소요 기간")
    assert "최저소요연수" in normalized or any(
        "최저소요연수" in v.replace(" ", "") for v in expand_search_text_variants("최소 소요 기간")
    )


def test_prepare_vector_query() -> None:
    assert prepare_vector_query("  제  5  조  ") == "제5조"


def test_bm25_uses_expanded_query_tokens() -> None:
    bm25 = BM25Light()
    docs = [
        "승진 최저소요연수 4년",
        "휴가 신청 절차",
    ]
    bm25.fit(docs)
    hits = bm25.search("최소 소요 기간", top_k=2)
    assert hits
    assert hits[0][0] == 0