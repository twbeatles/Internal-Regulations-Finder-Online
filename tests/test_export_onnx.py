# -*- coding: utf-8 -*-
"""ONNX export 스크립트 헬퍼 + ONNX 배치 인코딩 검증."""

from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import export_onnx  # pyright: ignore[reportMissingImports]  (scripts/는 sys.path로 추가)


def test_resolve_model_id_by_display_name() -> None:
    from app.config import AppConfig

    model_id = export_onnx.resolve_model_id(AppConfig.DEFAULT_MODEL)
    assert model_id == AppConfig.AVAILABLE_MODELS[AppConfig.DEFAULT_MODEL]


def test_resolve_model_id_passthrough() -> None:
    assert export_onnx.resolve_model_id("BAAI/bge-m3") == "BAAI/bge-m3"


def test_resolve_model_id_unknown() -> None:
    with pytest.raises(ValueError):
        export_onnx.resolve_model_id("존재하지 않는 모델")


def test_qwen3_export_refused() -> None:
    with pytest.raises(ValueError):
        export_onnx.ensure_exportable("Qwen/Qwen3-Embedding-0.6B")


def test_encoder_model_exportable() -> None:
    export_onnx.ensure_exportable("dragonkue/multilingual-e5-small-ko")
    export_onnx.ensure_exportable("BAAI/bge-m3")


def test_default_output_dir_escapes_id() -> None:
    out = export_onnx.default_output_dir("dragonkue/multilingual-e5-small-ko")
    assert out.endswith("dragonkue--multilingual-e5-small-ko")


def _block_imports(monkeypatch, prefix: str):
    real_import = builtins.__import__

    def _fake(name, *args, **kwargs):
        if name == prefix or name.startswith(prefix + "."):
            raise ImportError(f"blocked: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake)


def test_export_requires_optimum(monkeypatch, tmp_path) -> None:
    _block_imports(monkeypatch, "optimum")
    with pytest.raises(RuntimeError, match="optimum"):
        export_onnx.export_fp32("intfloat/multilingual-e5-small", str(tmp_path))


def test_quantize_requires_onnxruntime(monkeypatch, tmp_path) -> None:
    _block_imports(monkeypatch, "onnxruntime")
    with pytest.raises(RuntimeError, match="onnxruntime"):
        export_onnx.quantize_int8(str(tmp_path / "model.onnx"))


def test_verify_requires_deps(monkeypatch, tmp_path) -> None:
    _block_imports(monkeypatch, "sentence_transformers")
    with pytest.raises(RuntimeError):
        export_onnx.verify_onnx("intfloat/multilingual-e5-small", str(tmp_path))


def test_onnx_encode_batches_large_inputs(tmp_path, monkeypatch) -> None:
    np = pytest.importorskip("numpy")

    from app.config import AppConfig
    from app.services.embeddings_backends import onnx_backend

    batch_sizes: list[int] = []

    class _FakeTokenizer:
        def __call__(self, texts, padding=True, truncation=True, return_tensors="np"):
            n = len(texts)
            return {
                "input_ids": np.ones((n, 4), dtype=np.int64),
                "attention_mask": np.ones((n, 4), dtype=np.float32),
            }

    class _FakeSession:
        def run(self, _, inputs):
            n = inputs["input_ids"].shape[0]
            batch_sizes.append(n)
            return [np.ones((n, 4, 3), dtype=np.float32)]

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"fake")

    monkeypatch.setattr(AppConfig, "EMBED_BATCH_SIZE", 2, raising=False)
    embeddings = onnx_backend.OnnxEmbeddings(model_path=str(model_dir))
    embeddings._tokenizer = _FakeTokenizer()
    embeddings._session = _FakeSession()
    embeddings._input_names = ["input_ids", "attention_mask"]
    embeddings._output_names = ["last_hidden_state"]
    embeddings._has_last_hidden_state = True

    result = embeddings.embed_documents([f"문서 {i}" for i in range(5)])

    assert batch_sizes == [2, 2, 1]
    assert len(result) == 5
    assert all(len(vec) == 3 for vec in result)
    # 동일 입력이므로 전 배치 결과가 일치해야 함
    assert all(vec == result[0] for vec in result)


def test_onnx_encode_single_batch_when_small(tmp_path, monkeypatch) -> None:
    np = pytest.importorskip("numpy")

    from app.config import AppConfig
    from app.services.embeddings_backends import onnx_backend

    batch_sizes: list[int] = []

    class _FakeTokenizer:
        def __call__(self, texts, padding=True, truncation=True, return_tensors="np"):
            n = len(texts)
            return {
                "input_ids": np.ones((n, 4), dtype=np.int64),
                "attention_mask": np.ones((n, 4), dtype=np.float32),
            }

    class _FakeSession:
        def run(self, _, inputs):
            n = inputs["input_ids"].shape[0]
            batch_sizes.append(n)
            return [np.ones((n, 4, 3), dtype=np.float32)]

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"fake")

    monkeypatch.setattr(AppConfig, "EMBED_BATCH_SIZE", 32, raising=False)
    embeddings = onnx_backend.OnnxEmbeddings(model_path=str(model_dir))
    embeddings._tokenizer = _FakeTokenizer()
    embeddings._session = _FakeSession()
    embeddings._input_names = ["input_ids", "attention_mask"]
    embeddings._output_names = ["last_hidden_state"]
    embeddings._has_last_hidden_state = True

    result = embeddings.embed_documents(["문서 1", "문서 2"])

    assert batch_sizes == [2]
    assert len(result) == 2
