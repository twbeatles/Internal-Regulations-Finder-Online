# -*- coding: utf-8 -*-
"""2026-09 모델 전면 개편 검증: 프리셋 정합성 + E5 prefix + Qwen3 torch 강제."""

from __future__ import annotations

import pytest

from typing import Any


def test_default_model_is_registered() -> None:
    from app.config import AppConfig

    assert AppConfig.DEFAULT_MODEL in AppConfig.AVAILABLE_MODELS


def test_every_model_has_preset() -> None:
    from app.config import AppConfig

    presets = AppConfig.EMBED_MODEL_PRESETS
    for name, model_id in AppConfig.AVAILABLE_MODELS.items():
        assert model_id in presets, f"프리셋 누락: {name} ({model_id})"


def test_e5_models_require_prefixes() -> None:
    from app.services.embeddings_backends.factory import get_model_preset

    for model_id in (
        "dragonkue/multilingual-e5-small-ko",
        "intfloat/multilingual-e5-small",
    ):
        preset = get_model_preset(model_id)
        assert preset["query_prefix"] == "query: "
        assert preset["passage_prefix"] == "passage: "


def test_qwen3_forces_torch_backend() -> None:
    from app.services.embeddings_backends.factory import get_model_preset

    preset = get_model_preset("Qwen/Qwen3-Embedding-0.6B")
    assert preset.get("force_backend") == "torch"


def test_preset_matches_local_cache_dir() -> None:
    from app.services.embeddings_backends.factory import get_model_preset

    preset = get_model_preset("/app/models/dragonkue--multilingual-e5-small-ko")
    assert preset["query_prefix"] == "query: "


def test_unknown_model_returns_empty_preset() -> None:
    from app.services.embeddings_backends.factory import get_model_preset

    assert get_model_preset("someone/unknown-model") == {}


def test_prefix_wrapper_applies_e5_prefixes() -> None:
    from app.services.embeddings_backends.torch_backend import PrefixEmbeddingWrapper

    seen_queries: list = []
    seen_docs: list = []

    class _FakeInner:
        def embed_query(self, text: str):
            seen_queries.append(text)
            return [0.1]

        def embed_documents(self, texts: list):
            seen_docs.append(list(texts))
            return [[0.1] for _ in texts]

    wrapper = PrefixEmbeddingWrapper(
        _FakeInner(), query_prefix="query: ", passage_prefix="passage: "
    )
    wrapper.embed_query("휴가 규정")
    wrapper.embed_query("query: 중복 방지")
    wrapper.embed_documents(["휴가 규정", "passage: 중복 방지"])

    assert seen_queries == ["query: 휴가 규정", "query: 중복 방지"]
    assert seen_docs == [["passage: 휴가 규정", "passage: 중복 방지"]]


def test_onnx_prefix_applied_before_encode(tmp_path, monkeypatch) -> None:
    np = pytest.importorskip("numpy")

    from app.services.embeddings_backends import onnx_backend

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"fake")

    captured: dict[str, object] = {}

    def _fake_encode(self, texts):
        captured["texts"] = list(texts)
        return np.zeros((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr(onnx_backend.OnnxEmbeddings, "_encode", _fake_encode)

    embeddings = onnx_backend.create_onnx_embeddings(
        model_path=str(model_dir),
        query_prefix="query: ",
        passage_prefix="passage: ",
    )
    embeddings.embed_query("휴가 규정")
    assert captured["texts"] == ["query: 휴가 규정"]

    embeddings.embed_documents(["휴가 규정"])
    assert captured["texts"] == ["passage: 휴가 규정"]


def test_factory_routes_qwen3_to_torch(monkeypatch, tmp_path) -> None:
    from app.services.embeddings_backends import factory

    calls: dict[str, object] = {}

    def _fake_torch(**kwargs):
        calls["torch"] = kwargs
        return object()

    def _fake_onnx(**kwargs):
        calls["onnx"] = kwargs
        return object()

    monkeypatch.setattr(factory, "_create_torch_embeddings", _fake_torch)
    monkeypatch.setattr(factory, "_create_onnx_with_fallback", _fake_onnx)
    monkeypatch.setattr(factory.AppConfig, "EMBED_BACKEND", "onnx_fp32", raising=False)
    monkeypatch.setattr(factory.AppConfig, "OFFLINE_MODE", False, raising=False)
    monkeypatch.setattr(factory.AppConfig, "LOCAL_MODEL_PATH", "", raising=False)
    monkeypatch.setattr(
        factory, "get_app_directory", lambda: str(tmp_path), raising=True
    )

    factory.create_embeddings(
        model_name="Qwen3-0.6B (최신·고성능)",
        model_id_or_path="Qwen/Qwen3-Embedding-0.6B",
        backend="onnx_fp32",
    )
    assert "torch" in calls
    assert "onnx" not in calls


def test_factory_passes_e5_prefix_to_torch(monkeypatch, tmp_path) -> None:
    from app.services.embeddings_backends import factory

    calls: dict[str, Any] = {}

    def _fake_torch(**kwargs):
        calls["torch"] = kwargs
        return object()

    monkeypatch.setattr(factory, "_create_torch_embeddings", _fake_torch)
    monkeypatch.setattr(factory.AppConfig, "OFFLINE_MODE", False, raising=False)
    monkeypatch.setattr(factory.AppConfig, "LOCAL_MODEL_PATH", "", raising=False)
    monkeypatch.setattr(
        factory, "get_app_directory", lambda: str(tmp_path), raising=True
    )

    factory.create_embeddings(
        model_name="E5-Small-KO (경량·기본)",
        model_id_or_path="dragonkue/multilingual-e5-small-ko",
        backend="torch",
    )
    assert calls["torch"]["query_prefix"] == "query: "
    assert calls["torch"]["passage_prefix"] == "passage: "


def test_download_models_uses_single_source() -> None:
    import download_models
    from app.config import AppConfig

    assert download_models.AVAILABLE_MODELS == dict(AppConfig.AVAILABLE_MODELS)
