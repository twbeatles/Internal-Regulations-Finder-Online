# -*- coding: utf-8 -*-
import json
import os
from typing import Any, Iterator

import httpx

from rag.llm.base import LLMProvider


class AnthropicProvider(LLMProvider):
    provider_name = "anthropic"

    def __init__(self, model: str, api_key: str | None = None, **defaults: Any):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.defaults = defaults

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _split_messages(self, messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        chat_messages = [
            {"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]}
            for m in messages
            if m.get("role") in ("user", "assistant")
        ]
        return "\n".join(system_parts), chat_messages

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        system, chat_messages = self._split_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.defaults.get("max_tokens", 2048)),
            "temperature": kwargs.get("temperature", self.defaults.get("temperature", 0.1)),
            "messages": chat_messages,
        }
        if system:
            payload["system"] = system
        with httpx.Client(timeout=kwargs.get("timeout", 120.0)) as client:
            resp = client.post("https://api.anthropic.com/v1/messages", headers=self._headers(), json=payload)
            resp.raise_for_status()
            data = resp.json()
            parts = data.get("content", [])
            return "".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()

    def stream(self, messages: list[dict[str, str]], **kwargs: Any) -> Iterator[str]:
        system, chat_messages = self._split_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.defaults.get("max_tokens", 2048)),
            "temperature": kwargs.get("temperature", self.defaults.get("temperature", 0.1)),
            "messages": chat_messages,
            "stream": True,
        }
        if system:
            payload["system"] = system
        with httpx.Client(timeout=None) as client:
            with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers=self._headers(),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if not chunk:
                        continue
                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    if data.get("type") == "content_block_delta":
                        text = data.get("delta", {}).get("text", "")
                        if text:
                            yield text

    def health(self) -> bool:
        return bool(self.api_key)