# -*- coding: utf-8 -*-
"""RAG 조문 청킹 테스트."""

from rag.pipeline.chunking import build_chunks_from_text, extract_article_number


SAMPLE_REGULATION = """
제1조(목적) 이 규정은 연차휴가에 관한 사항을 정한다.

제2조(연차휴가) ① 직원의 연차유급휴가는 15일로 한다.
② 1년 미만 근무자는 월 1일을 부여한다.
"""


class TestArticleChunking:
    def test_extract_article_number(self):
        assert extract_article_number("제15조 연차는?") == "제15조"
        assert extract_article_number("일반 질문") is None

    def test_build_chunks_from_articles(self):
        chunks = build_chunks_from_text(SAMPLE_REGULATION, chunk_size=800, chunk_overlap=80)
        assert len(chunks) >= 2
        assert any(c.get("article_no") == "제1조" for c in chunks)
        assert any(c.get("chunk_type") == "article" for c in chunks)