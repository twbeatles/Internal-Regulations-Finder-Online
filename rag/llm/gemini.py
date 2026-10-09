# -*- coding: utf-8 -*-
import json
import os
from typing import Any, Iterator

import httpx

from rag.llm.base import LLMProvider


class GeminiProvider(LLMProvider):
    provider_name = "gemini"

    def __init__(self, model: str, api_key: str | None = None, **defaults: Any):
        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
        self.defaults = defaults

    def _to_contents(self, messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, Any]]]:
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        contents = []
        for m in messages:
            if m.get("role") not in ("user", "assistant"):
                continue
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        return ("\n".join(system_parts) or None), contents

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        system, contents = self._to_contents(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", self.defaults.get("temperature", 0.1)),
                "maxOutputTokens": kwargs.get("max_tokens", self.defaults.get("max_tokens", 2048)),
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )
        with httpx.Client(timeout=kwargs.get("timeout", 120.0)) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts).strip()

    def stream(self, messages: list[dict[str, str]], **kwargs: Any) -> Iterator[str]:
        # Gemini 스트리밍은 환경별 차이가 있어 complete 폴백
        text = self.complete(messages, **kwargs)
        if text:
            yield text

    def health(self) -> bool:
        return bool(self.api_key)