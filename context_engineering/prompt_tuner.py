"""
Meta-agent that analyzes post-conversation records and suggests prompt adjustments.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from app.factories.agents_runner import AgentsRunner

from .prompts import PROMPT_TUNER_SYSTEM_PROMPT


class PromptTunerAgent:
    """Generates improved guidelines based on conversation outcomes."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1",
        temperature: float = 0.3,
    ):
        self.runner = AgentsRunner(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_output_tokens=600,
        )

    def run(
        self,
        *,
        run_records: List[Dict],
        current_prompt_notes: Optional[str] = None,
    ) -> Dict:
        """Return JSON suggestions for prompt and strategy adjustments."""
        payload_lines = ["## REGISTROS RECIENTES"]
        for record in run_records:
            payload_lines.append(str(record))

        if current_prompt_notes:
            payload_lines.append("\n## PROMPT ACTUAL")
            payload_lines.append(current_prompt_notes)

        payload_lines.append("\nDevuelve el JSON con las recomendaciones.")
        payload = "\n".join(payload_lines)

        return self.runner.run_json(
            system_prompt=PROMPT_TUNER_SYSTEM_PROMPT,
            user_content=payload,
        )


__all__ = ["PromptTunerAgent"]
