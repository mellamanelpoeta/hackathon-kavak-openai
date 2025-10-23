"""
Customer agent factory.

Given a structured customer profile, this module uses OpenAI Agents (Responses API)
to synthesize a customer persona prompt that can be used to simulate 1:1 conversations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from app.factories.agents_runner import AgentsRunner


CUSTOMER_SYSTEM_SEED = """
Eres un cliente de Kavak. Responde desde la primera persona.
Mantén la personalidad y el contexto que se te proporciona.
Sé consistente con tu historial y emociones.
No inventes hechos fuera del contexto.
Responde en español natural, máximo 80 palabras por intervención.
"""

CUSTOMER_PROMPT_BUILDER = """
Construye una ficha breve para un cliente de Kavak.
Debes devolver instrucciones en formato JSON con esta forma:
{
  "persona_prompt": "<texto para sistema>",
  "starting_message": "<mensaje inicial opcional>"
}

La ficha debe resumir:
- datos demográficos relevantes
- historial reciente con Kavak
- sentimiento actual (feliz/enojado)
- objetivos u obstáculos

Usa SOLO la información proporcionada.
"""


@dataclass
class CustomerAgent:
    """In-memory representation of a customer simulation agent."""

    customer_id: str
    system_prompt: str
    starting_message: Optional[str]
    profile: Dict


class CustomerAgentFactory:
    """
    Factory that generates customer agents on the fly.
    Uses an LLM to craft a persona-specific system prompt for each customer profile.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.4,
    ):
        self.runner = AgentsRunner(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_output_tokens=400,
        )

    def create_agent(self, profile: Dict) -> CustomerAgent:
        """
        Generate a persona-specific customer agent configuration.

        Args:
            profile: dictionary describing the customer (matches JSON schema)

        Returns:
            CustomerAgent dataclass with system prompt and optional opening message.
        """
        user_context = self._build_context_string(profile)

        result = self.runner.run_json(
            system_prompt=CUSTOMER_PROMPT_BUILDER,
            user_content=user_context,
        )

        persona_prompt = result.get("persona_prompt", "").strip()
        starting_message = result.get("starting_message")

        if not persona_prompt:
            persona_prompt = self._fallback_persona_prompt(profile)

        system_prompt = f"{CUSTOMER_SYSTEM_SEED.strip()}\n\nContexto cliente:\n{persona_prompt}"

        return CustomerAgent(
            customer_id=profile.get("customer_id", "unknown"),
            system_prompt=system_prompt,
            starting_message=starting_message,
            profile=profile,
        )

    def _build_context_string(self, profile: Dict) -> str:
        """Serialize profile into a text block the LLM can understand."""
        persona = profile.get("persona", {})
        purchase = profile.get("purchase", {})
        history = profile.get("history", {})
        risk = profile.get("risk_signals", {})
        cohort_info = profile.get("cohort")

        lines = [
            f"Customer ID: {profile.get('customer_id')}",
            f"Cohorte: {cohort_info}",
            f"Persona: {persona}",
            f"Compra: {purchase}",
            f"Historial: {history}",
            f"Riesgos: {risk}",
        ]
        return "\n".join(lines)

    def _fallback_persona_prompt(self, profile: Dict) -> str:
        """Fallback system instructions if the LLM output is empty/invalid."""
        persona = profile.get("persona", {})
        bio = persona.get("bio", "Cliente de Kavak.")
        mood = "satisfecho" if profile.get("cohort", {}).get("satisfied") else "insatisfecho"
        return f"{bio} Actualmente se siente {mood}."


__all__ = ["CustomerAgentFactory", "CustomerAgent"]
