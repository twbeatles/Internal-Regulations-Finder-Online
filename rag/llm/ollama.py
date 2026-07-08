# -*- coding: utf-8 -*-
import json
from typing import Any, Iterator

import httpx

from rag.llm.base import LLMProvider


class OllamaProvider(LLMProvider):
    provider_name = "ollama"

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", **defaults: Any):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.defaults = defaults

    def _payload(self, messages: list[dict[str, str]], stream: bool, **kwargs: Any) -> dict[str, Any]:
        options = {
            "temperature": kwargs.get("temperature", self.defaults.get("temperature", 0.1)),
            "num_predict": kwargs.get("max_tokens", self.defaults.get("max_tokens", 2048)),
        }
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": options,
        }

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        payload = self._payload(messages, stream=False, **kwargs)
        with httpx.Client(timeout=kwargs.get("timeout", 120.0)) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return str(data.get("message", {}).get("content", "")).strip()

    def stream(self, messages: list[dict[str, str]], **kwargs: Any) -> Iterator[str]:
        payload = self._payload(messages, stream=True, **kwargs)
        with httpx.Client(timeout=None) as client:
            with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("done"):
                        break
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token

    def health(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False