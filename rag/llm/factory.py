# -*- coding: utf-8 -*-
"""LLM 프로바이더 팩토리."""

from __future__ import annotations

import threading
import time
from typing import Any

from app.config import AppConfig
from app.utils import logger
from rag.config import RagConfig
from rag.llm.anthropic import AnthropicProvider
from rag.llm.base import LLMProvider
from rag.llm.gemini import GeminiProvider
from rag.llm.ollama import OllamaProvider
from rag.llm.openai_provider import OpenAIProvider

_health_lock = threading.Lock()
_health_cache: dict[tuple[Any, ...], tuple[float, bool]] = {}


def is_provider_healthy(provider: LLMProvider | None) -> bool:
    if provider is None:
        return False
    return _provider_health_cached(provider)


def _provider_health_cached(provider: LLMProvider) -> bool:
    """TTL 캐시된 health 체크 (매 요청 HTTP ping 완화)."""
    ttl = float(getattr(AppConfig, "LLM_HEALTH_CACHE_TTL", 30) or 30)
    key = (
        getattr(provider, "provider_name", type(provider).__name__),
        getattr(provider, "model", ""),
        getattr(provider, "base_url", ""),
    )
    now = time.time()
    with _health_lock:
        entry = _health_cache.get(key)
        if entry and (now - entry[0]) < ttl:
            return entry[1]
    try:
        ok = bool(provider.health())
    except Exception:
        ok = False
    with _health_lock:
        _health_cache[key] = (now, ok)
    return ok


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
    if primary and _provider_health_cached(primary):
        return primary
    fallback_cfg = config.llm_fallback or {}
    if fallback_cfg.get("enabled"):
        fallback = _build_provider(fallback_cfg)
        if fallback and _provider_health_cached(fallback):
            logger.info("기본 LLM 사용 불가 — 폴백 프로바이더 사용")
            return fallback
    # health 실패 시에도 primary 핸들을 반환 — 호출측에서 stream/complete 재시도·폴백
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
            "healthy": bool(primary and _provider_health_cached(primary)),
        },
        "fallback": {
            "enabled": bool(config.llm_fallback.get("enabled")),
            "provider": config.llm_fallback.get("provider"),
            "model": config.llm_fallback.get("model"),
            "healthy": bool(fallback and _provider_health_cached(fallback)),
        },
    }