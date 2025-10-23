"""
AgentsRunner: thin wrapper around OpenAI Agents (Responses API).
Centralizes request building, retries, and error handling for both
textual and JSON outputs used by demo agents.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI
from openai.types import Response

DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 1.5


def _build_content_block(text: str) -> Dict[str, Any]:
    """Helpers to wrap plain text for the Responses API."""
    return {"type": "text", "text": text}


class AgentsRunner:
    """
    Wrapper to interact with OpenAI Agents (Responses API).

    Provides helpers for:
      * Free-form text generation (`run_text`)
      * JSON-formatted outputs (`run_json`)
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.4,
        max_output_tokens: int = 512,
        max_retries: int = DEFAULT_RETRIES,
        backoff: float = DEFAULT_BACKOFF,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required (set OPENAI_API_KEY or pass api_key).")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.backoff = backoff

    def run_text(
        self,
        *,
        system_prompt: str,
        user_content: str,
        extra_input: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Generate free-form text using the Agents Responses API.
        """
        response = self._create_response(
            system_prompt=system_prompt,
            user_content=user_content,
            extra_input=extra_input,
            response_format=None,
        )
        return response.output_text.strip()

    def run_json(
        self,
        *,
        system_prompt: str,
        user_content: str,
        extra_input: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate JSON output using the Agents Responses API.
        """
        response = self._create_response(
            system_prompt=system_prompt,
            user_content=user_content,
            extra_input=extra_input,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(response.output_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON returned by model: {exc}") from exc

    def _create_response(
        self,
        *,
        system_prompt: str,
        user_content: str,
        extra_input: Optional[List[Dict[str, Any]]],
        response_format: Optional[Dict[str, Any]],
    ) -> Response:
        """
        Invoke OpenAI Responses API with retry and backoff policy.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [_build_content_block(system_prompt)]},
            {"role": "user", "content": [_build_content_block(user_content)]},
        ]

        if extra_input:
            messages.extend(extra_input)

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.responses.create(
                    model=self.model,
                    input=messages,
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                    response_format=response_format,
                )
            except Exception as exc:  # broad catch: SDK raises various subclasses
                last_exc = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.backoff ** attempt)

        raise RuntimeError(f"Agents API call failed after {self.max_retries} attempts") from last_exc


__all__ = ["AgentsRunner"]
