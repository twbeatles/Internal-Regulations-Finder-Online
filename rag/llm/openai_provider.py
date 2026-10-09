# -*- coding: utf-8 -*-
import json
import os
from typing import Any, Iterator

import httpx

from rag.llm.base import LLMProvider


class OpenAIProvider(LLMProvider):
    provider_name = "openai"

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None, **defaults: Any):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.defaults = defaults

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.defaults.get("temperature", 0.1)),
            "max_tokens": kwargs.get("max_tokens", self.defaults.get("max_tokens", 2048)),
        }
        with httpx.Client(timeout=kwargs.get("timeout", 120.0)) as client:
            resp = client.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=payload)
            resp.raise_for_status()
            data = resp.json()
            return str(data["choices"][0]["message"]["content"]).strip()

    def stream(self, messages: list[dict[str, str]], **kwargs: Any) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.defaults.get("temperature", 0.1)),
            "max_tokens": kwargs.get("max_tokens", self.defaults.get("max_tokens", 2048)),
            "stream": True,
        }
        with httpx.Client(timeout=None) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        yield delta

    def health(self) -> bool:
        return bool(self.api_key)