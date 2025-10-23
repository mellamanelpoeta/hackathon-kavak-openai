"""
AgentsRunner: thin wrapper around OpenAI Agents (Responses API).
Centralizes request building, retries, and error handling for both
textual and JSON outputs used by demo agents.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI, APIError

DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 1.5


def _extract_text_from_response(response: Any) -> str:
    """
    Extract text from OpenAI Responses API response object.
    The response has structure: response.output[0].content[0].text
    """
    if hasattr(response, 'output_text'):
        # Fallback for older SDK versions
        return response.output_text

    # Extract from new structure
    if response.output and len(response.output) > 0:
        message = response.output[0]
        if message.content and len(message.content) > 0:
            return message.content[0].text

    raise ValueError("Unable to extract text from response")


def _build_content_block(text: str) -> Dict[str, Any]:
    """Helpers to wrap plain text for the Responses API."""
    return {"type": "input_text", "text": text}


logger = logging.getLogger(__name__)


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
        if not hasattr(self.client, "responses"):
            raise AttributeError(
                "OpenAI client is missing `responses`. "
                "Upgrade to openai>=1.3.0 to use the Responses API."
            )
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
        return _extract_text_from_response(response).strip()

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
            response_format=None,
        )
        try:
            return json.loads(_extract_text_from_response(response))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON returned by model: {exc}") from exc

    def _create_response(
        self,
        *,
        system_prompt: str,
        user_content: str,
        extra_input: Optional[List[Dict[str, Any]]],
        response_format: Optional[Dict[str, Any]],
    ) -> Any:
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
                # Build kwargs for API call
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "input": messages,
                    "temperature": self.temperature,
                }

                if self.max_output_tokens:
                    kwargs["max_output_tokens"] = self.max_output_tokens

                # Attach structured response directives when requested
                if response_format:
                    kwargs["response_format"] = response_format

                return self.client.responses.create(**kwargs)
            except Exception as exc:  # broad catch: SDK raises various subclasses
                last_exc = exc

                status_code = getattr(exc, "status_code", None)
                request_id = getattr(exc, "request_id", None)
                error_payload = None
                response_obj = getattr(exc, "response", None)
                if response_obj is not None:
                    try:
                        error_payload = response_obj.json()
                    except Exception:  # pragma: no cover - best effort
                        error_payload = str(response_obj)

                logger.warning(
                    "Agents API call failed (attempt %d/%d, model=%s, status=%s, request_id=%s, error=%s)",
                    attempt,
                    self.max_retries,
                    self.model,
                    status_code,
                    request_id,
                    error_payload or exc,
                )
                if attempt == self.max_retries:
                    break
                time.sleep(self.backoff ** attempt)

        raise RuntimeError(
            f"Agents API call failed after {self.max_retries} attempts: {last_exc}"
        ) from last_exc


__all__ = ["AgentsRunner"]
