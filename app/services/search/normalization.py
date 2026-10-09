# -*- coding: utf-8 -*-
"""한국어 검색 쿼리·인덱스 정규화 (kcsc-mcp search_normalization 이식)."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

# Ported from kcsc-mcp (2026-07)

QUERY_NOISE_TERMS = (
    "관련",
    "내용",
    "정보",
    "사항",
    "기준",
    "알려줘",
    "알려줄래",
    "알려주세요",
    "찾아줘",
    "검색해줘",
)

QUERY_REPLACEMENTS = (
    ("최소 소요 기간", "최저소요연수"),
    ("최소 소요기간", "최저소요연수"),
    ("최소기간", "최저연수"),
    ("최소 기간", "최저기간"),
    ("소요 기간", "소요연수"),
    ("소요기간", "소요연수"),
    ("최저 기간", "최저연수"),
)

_KOREAN_DIGITS = {
    "일": 1,
    "이": 2,
    "삼": 3,
    "사": 4,
    "오": 5,
    "육": 6,
    "칠": 7,
    "팔": 8,
    "구": 9,
}
_CIRCLED_PARAGRAPHS = "①②③④⑤⑥⑦⑧⑨⑩"


def _normalize_text_compact(value: str | None) -> str:
    """KCSC normalize_text 호환: 공백 제거 + 소문자."""
    if not value:
        return ""
    return re.sub(r"\s+", "", value).lower()


def normalize_search_text(value: str | None) -> str:
    """어휘 검색용 compact canonical form."""
    if not value:
        return ""
    canonical = canonicalize_search_text(value)
    without_punctuation = _strip_search_punctuation(canonical)
    return re.sub(r"\s+", "", without_punctuation).lower()


def build_search_index_text(*parts: str | None) -> str:
    """청크 인덱스용 정규화 텍스트 (향후 인덱싱 확장용)."""
    raw_parts = [part for part in parts if part and str(part).strip()]
    variants: list[str] = []
    for candidate in (" ".join(raw_parts), *raw_parts):
        variants.extend(expand_search_text_variants(candidate, include_noise_variants=False))
        variants.append(_normalize_text_compact(candidate))
        variants.append(normalize_search_text(candidate))
    return " ".join(_unique(normalize_search_text(variant) for variant in variants))


def expand_search_text_variants(query: str, *, include_noise_variants: bool = True) -> list[str]:
    """쿼리 동의어·표기 변형 확장."""
    variants: list[str] = []

    def add(candidate: str | None) -> None:
        compact = " ".join((candidate or "").split()).strip()
        if compact:
            variants.append(compact)

    add(query)
    canonical = canonicalize_search_text(query)
    add(canonical)
    add(_strip_search_punctuation(canonical))
    add(normalize_search_text(canonical))

    if include_noise_variants:
        cleaned = query
        for term in QUERY_NOISE_TERMS:
            cleaned = cleaned.replace(term, " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        add(cleaned)

    bases = list(variants)
    for base in bases:
        add(base.replace("직급별", "직급 "))
        add(base.replace("직급별", " "))
        for source, target in QUERY_REPLACEMENTS:
            if source in base:
                add(base.replace(source, target))
        for money_variant in _money_variants(base):
            add(money_variant)
        for grade_variant in _grade_year_variants(base):
            add(grade_variant)
        for annex_variant in _annex_variants(base):
            add(annex_variant)

    normalized_query = normalize_search_text(query)
    if "승진" in normalized_query and any(
        token in normalized_query
        for token in (
            "소요기간",
            "소요연수",
            "최소기간",
            "최저기간",
            "최소소요기간",
            "최저소요기간",
            "최저소요연수",
        )
    ):
        add("승진최저소요연수")
        add("승진 최저소요연수")

    return _unique(variants)


def canonicalize_search_text(value: str | None) -> str:
    """조문·금액·별표 등 표기 통일."""
    if not value:
        return ""
    text = unicodedata.normalize("NFKC", value)
    text = text.replace("，", ",")
    text = re.sub(r"제\s*(?P<article>\d+)\s*의\s*(?P<sub>\d+)\s*조", r"제\g<article>의\g<sub>조", text)
    text = re.sub(r"제\s*(?P<article>\d+(?:의\d+)?)\s*조", r"제\g<article>조", text)
    text = re.sub(r"제\s*(?P<paragraph>\d+)\s*항", r"제\g<paragraph>항", text)
    text = re.sub(r"제\s*(?P<clause>\d+)\s*호", r"제\g<clause>호", text)
    text = re.sub(r"(?P<grade>\d+)\s*급", r"\g<grade>급", text)
    text = re.sub(r"(?P<years>\d+)\s*년", r"\g<years>년", text)
    text = re.sub(r"별지\s*제?\s*(?P<no>\d+)\s*호\s*서식", r"별지 제\g<no>호 서식", text)
    text = re.sub(r"(?P<kind>별표|별첨|첨부)\s*(?P<no>\d+)", r"\g<kind> \g<no>", text)
    text = re.sub(r"부\s*칙", "부칙", text)
    text = _canonicalize_money(text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_vector_query(query: str) -> str:
    """벡터 검색용 canonical 쿼리."""
    canonical = canonicalize_search_text(query)
    return canonical or query.strip()


def prepare_bm25_query_terms(tokenize, query: str) -> list[str]:
    """BM25용 확장 쿼리 토큰 (중복 제거)."""
    terms: list[str] = []
    seen: set[str] = set()
    for variant in expand_search_text_variants(query):
        for token in tokenize(variant):
            if token in seen:
                continue
            seen.add(token)
            terms.append(token)
    return terms


def _canonicalize_money(value: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d)", "", value)

    def numeric_unit(match: re.Match[str], multiplier: int) -> str:
        number = int(match.group("number").replace(",", ""))
        return f"{number * multiplier}만원"

    text = re.sub(r"(?P<number>\d[\d,]*)\s*천\s*만\s*원", lambda item: numeric_unit(item, 1000), text)
    text = re.sub(r"(?P<number>\d[\d,]*)\s*백\s*만\s*원", lambda item: numeric_unit(item, 100), text)
    text = re.sub(r"(?P<number>\d[\d,]*)\s*십\s*만\s*원", lambda item: numeric_unit(item, 10), text)
    text = re.sub(
        r"(?P<number>\d[\d,]*)\s*만\s*원",
        lambda item: f"{int(item.group('number').replace(',', ''))}만원",
        text,
    )
    text = re.sub(
        r"(?P<number>\d[\d,]*)\s*억\s*원",
        lambda item: f"{int(item.group('number').replace(',', ''))}억원",
        text,
    )
    text = re.sub(
        rf"(?P<number>[{''.join(_KOREAN_DIGITS)}])\s*천\s*만\s*원",
        lambda item: f"{_KOREAN_DIGITS[item.group('number')] * 1000}만원",
        text,
    )
    text = re.sub(
        rf"(?P<number>[{''.join(_KOREAN_DIGITS)}])\s*억\s*원",
        lambda item: f"{_KOREAN_DIGITS[item.group('number')]}억원",
        text,
    )
    return text


def _money_variants(value: str) -> list[str]:
    canonical = canonicalize_search_text(value)
    variants: list[str] = []
    for match in re.finditer(r"(?P<number>\d+)억원", canonical):
        number = int(match.group("number"))
        variants.append(canonical[: match.start()] + f"{number * 10000}만원" + canonical[match.end() :])
    for match in re.finditer(r"(?P<number>\d+)만원", canonical):
        number = int(match.group("number"))
        if number >= 10000 and number % 10000 == 0:
            variants.append(canonical[: match.start()] + f"{number // 10000}억원" + canonical[match.end() :])
        if number >= 1000 and number % 1000 == 0:
            variants.append(canonical[: match.start()] + f"{number // 1000}천만원" + canonical[match.end() :])
    return variants


def _grade_year_variants(value: str) -> list[str]:
    canonical = canonicalize_search_text(value)
    variants: list[str] = []
    for match in re.finditer(r"(?P<from>\d+)급.*?(?P<to>\d+)급.*?(?P<years>\d+)년", canonical):
        from_grade = match.group("from")
        to_grade = match.group("to")
        years = match.group("years")
        variants.extend(
            [
                f"{from_grade}급{to_grade}급{years}년",
                f"{from_grade}급에서{to_grade}급승진{years}년",
                f"{from_grade}급에서{to_grade}급으로승진{years}년",
                f"{from_grade}급에서{to_grade}급으로의승진{years}년",
            ]
        )
    return variants


def _annex_variants(value: str) -> list[str]:
    canonical = canonicalize_search_text(value)
    variants: list[str] = []
    for match in re.finditer(r"(?P<kind>별표|별첨|첨부)\s*(?P<no>\d+)", canonical):
        kind = match.group("kind")
        no = match.group("no")
        variants.append(canonical[: match.start()] + f"{kind}{no}" + canonical[match.end() :])
        variants.append(canonical[: match.start()] + f"{kind} {no}" + canonical[match.end() :])
    for match in re.finditer(r"부\s*칙", value):
        variants.append(value[: match.start()] + "부칙" + value[match.end() :])
    return variants


def _strip_search_punctuation(value: str) -> str:
    kept: list[str] = []
    for char in value:
        if char in _CIRCLED_PARAGRAPHS:
            kept.append(char)
            continue
        category = unicodedata.category(char)
        if category.startswith(("P", "S")):
            continue
        kept.append(char)
    return "".join(kept)


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        candidate = " ".join(str(value or "").split()).strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
    return result