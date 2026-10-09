# -*- coding: utf-8 -*-
"""LLM 프로바이더 추상화."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator


class LLMProvider(ABC):
    provider_name: str = "base"

    @abstractmethod
    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    def stream(self, messages: list[dict[str, str]], **kwargs: Any) -> Iterator[str]:
        raise NotImplementedError

    def health(self) -> bool:
        try:
            reply = self.complete(
                [{"role": "user", "content": "ping"}],
                max_tokens=8,
                temperature=0,
            )
            return bool(reply and reply.strip())
        except Exception:
            return False