# -*- coding: utf-8 -*-
from rag.pipeline.guardrails import Guardrails
from rag.pipeline.verification import extract_claims, verify_answer_against_chunks
from rag.schemas import RetrievedChunk


def test_extract_claims_from_bullets() -> None:
    claims = extract_claims("- 연차는 15일입니다.\n- 신청은 전일까지 합니다.")
    assert len(claims) >= 1


def test_verify_supported_claim() -> None:
    report = verify_answer_against_chunks(
        "연차유급휴가는 15일을 부여합니다.",
        ["제2조 연차유급휴가 15일을 부여한다."],
    )
    assert report.score >= 0.35
    assert report.supported_claims >= 1


def test_guardrails_includes_verification_score() -> None:
    chunks = [
        RetrievedChunk(0, "연차 15일", "휴가규정.docx", "", "f1", 0.8, article_no="제2조"),
    ]
    result = Guardrails().apply("연차는 15일입니다 [1]", chunks)
    assert result.verification_score >= 0.0
    assert "verification_score" in result.to_dict()