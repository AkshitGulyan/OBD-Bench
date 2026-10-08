from __future__ import annotations

import os
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


class LLMClient:
    def __init__(self, provider: str = "openai", model: str | None = None, api_key: str | None = None, base_url: str | None = None):
        self.provider = provider.lower()
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

    async def generate(self, messages: list[dict[str, str]], model: str | None = None, temperature: float | int | None = None, max_tokens: int | None = None, **kwargs: Any):
        if self.provider != "openai":
            raise ValueError(f"Unsupported provider: {self.provider!r}")
        if OpenAI is None:
            raise RuntimeError("The 'openai' package is not installed. Add it to requirements.txt.")
        if not self.api_key:
            raise RuntimeError("No OpenAI API key was configured. Set OPENAI_API_KEY or provide api_key.")

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature if temperature is not None else 0,
            max_tokens=max_tokens,
            **kwargs,
        )

        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return {
            "content": content,
            "model": response.model,
            "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
        }
