# -*- coding: utf-8 -*-
"""LLM 프로바이더 팩토리."""

from __future__ import annotations

from typing import Any

from app.utils import logger
from rag.config import RagConfig
from rag.llm.anthropic import AnthropicProvider
from rag.llm.base import LLMProvider
from rag.llm.gemini import GeminiProvider
from rag.llm.ollama import OllamaProvider
from rag.llm.openai_provider import OpenAIProvider


def _build_provider(spec: dict[str, Any]) -> LLMProvider | None:
    provider = str(spec.get("provider", "ollama")).strip().lower()
    model = str(spec.get("model", "")).strip()
    if not model:
        return None
    defaults = {
        "temperature": spec.get("temperature", 0.1),
        "max_tokens": spec.get("max_tokens", 2048),
    }
    if provider == "ollama":
        return OllamaProvider(
            model=model,
            base_url=str(spec.get("base_url", "http://127.0.0.1:11434")),
            **defaults,
        )
    if provider == "openai":
        return OpenAIProvider(model=model, base_url=spec.get("base_url"), **defaults)
    if provider == "anthropic":
        return AnthropicProvider(model=model, **defaults)
    if provider == "gemini":
        return GeminiProvider(model=model, **defaults)
    logger.warning(f"지원하지 않는 LLM 프로바이더: {provider}")
    return None


def create_llm_provider(config: RagConfig | None = None) -> LLMProvider | None:
    config = config or RagConfig.from_settings()
    primary = _build_provider(config.llm)
    if primary and primary.health():
        return primary
    fallback_cfg = config.llm_fallback or {}
    if fallback_cfg.get("enabled"):
        fallback = _build_provider(fallback_cfg)
        if fallback and fallback.health():
            logger.info("기본 LLM 사용 불가 — 폴백 프로바이더 사용")
            return fallback
    return primary or _build_provider(config.llm)


def get_llm_health(config: RagConfig | None = None) -> dict[str, Any]:
    config = config or RagConfig.from_settings()
    primary = _build_provider(config.llm)
    fallback = None
    if config.llm_fallback.get("enabled"):
        fallback = _build_provider(config.llm_fallback)
    return {
        "primary": {
            "provider": config.llm.get("provider"),
            "model": config.llm.get("model"),
            "healthy": bool(primary and primary.health()),
        },
        "fallback": {
            "enabled": bool(config.llm_fallback.get("enabled")),
            "provider": config.llm_fallback.get("provider"),
            "model": config.llm_fallback.get("model"),
            "healthy": bool(fallback and fallback.health()),
        },
    }