"""Local LLM client for smart voice agents.

Talks to any OpenAI-compatible chat completions server. Defaults target
Ollama (http://localhost:11434/v1) which exposes the OpenAI API; LM Studio,
llama.cpp server, and vLLM work the same way.
"""

import logging
import time

import requests

logger = logging.getLogger("rajvoicecloner.server")


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 45.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._available: bool | None = None
        self._available_checked_at = 0.0

    def configure(self, base_url: str, model: str, api_key: str | None) -> None:
        """Apply new connection settings and reset the availability cache."""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._available = None
        self._available_checked_at = 0.0

    def list_models(self) -> list[str]:
        """Model ids exposed by the endpoint. Raises on connection errors."""
        res = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=5)
        res.raise_for_status()
        data = res.json().get("data", [])
        return [m["id"] for m in data if isinstance(m, dict) and "id" in m]

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def available(self, ttl_seconds: float = 30.0) -> bool:
        """Cheap reachability check, cached for ``ttl_seconds``."""
        now = time.monotonic()
        if self._available is not None and now - self._available_checked_at < ttl_seconds:
            return self._available
        try:
            res = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=2)
            self._available = res.status_code == 200
        except requests.RequestException:
            self._available = False
        self._available_checked_at = now
        if not self._available:
            logger.info("LLM not reachable at %s (agents fall back to scripted mode)", self.base_url)
        return self._available

    def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 150) -> str:
        res = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout=self.timeout,
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()
