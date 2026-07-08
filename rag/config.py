# -*- coding: utf-8 -*-
"""RAG 런타임 설정 (settings.json rag 섹션)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from app.services.settings_store import get_settings_store

DEFAULT_RAG_SETTINGS: dict[str, Any] = {
    "llm": {
        "provider": "ollama",
        "model": "exaone3.5:7.8b",
        "base_url": "http://127.0.0.1:11434",
        "temperature": 0.1,
        "max_tokens": 2048,
    },
    "llm_fallback": {
        "enabled": False,
        "provider": "openai",
        "model": "gpt-4o-mini",
    },
    "retrieval": {
        "top_k": 8,
        "rerank_top_k": 4,
        "hybrid": True,
        "hyde": True,
        "article_fast_path": True,
    },
    "guardrails": {
        "require_citation": True,
        "min_confidence": 0.35,
        "disclaimer": "본 답변은 사내 규정 참고용이며 법적 효력이 없습니다.",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class RagConfig:
    llm: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_RAG_SETTINGS["llm"]))
    llm_fallback: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_RAG_SETTINGS["llm_fallback"]))
    retrieval: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_RAG_SETTINGS["retrieval"]))
    guardrails: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_RAG_SETTINGS["guardrails"]))

    @classmethod
    def from_settings(cls, settings: dict[str, Any] | None = None) -> "RagConfig":
        if settings is None:
            settings = get_settings_store().load()
        rag_raw = settings.get("rag") if isinstance(settings.get("rag"), dict) else {}
        merged = _deep_merge(DEFAULT_RAG_SETTINGS, rag_raw)
        return cls(
            llm=merged.get("llm", {}),
            llm_fallback=merged.get("llm_fallback", {}),
            retrieval=merged.get("retrieval", {}),
            guardrails=merged.get("guardrails", {}),
        )


def get_search_mode(settings: dict[str, Any] | None = None) -> str:
    if settings is None:
        settings = get_settings_store().load()
    mode = str(settings.get("search_mode", "rag")).strip().lower()
    return mode if mode in ("rag", "legacy") else "rag"


def save_search_mode(mode: str) -> bool:
    mode = str(mode).strip().lower()
    if mode not in ("rag", "legacy"):
        return False
    store = get_settings_store()
    settings = store.load()
    settings["search_mode"] = mode
    if "rag" not in settings:
        settings["rag"] = copy.deepcopy(DEFAULT_RAG_SETTINGS)
    return store.save(settings)


def save_rag_config(rag_config: dict[str, Any]) -> bool:
    store = get_settings_store()
    settings = store.load()
    current = settings.get("rag") if isinstance(settings.get("rag"), dict) else {}
    settings["rag"] = _deep_merge(DEFAULT_RAG_SETTINGS, {**current, **rag_config})
    return store.save(settings)