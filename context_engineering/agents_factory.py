"""
Customer agent factory that generates deterministic personas from JSON profiles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


CUSTOMER_SYSTEM_SEED = """
Eres un cliente de Kavak. Responde en primera persona desde tu experiencia.
Comparte emociones auténticas, mantén coherencia con tu historia y evita prometer acciones en nombre de Kavak.
Responde en español natural, máximo 80 palabras por intervención.
Si el agente solicita tu calificación NPS, responde con el formato exacto 'NPS: <número 0-10>' seguido de un comentario breve.
Usa variación natural en tu lenguaje, menciona detalles concretos de tu experiencia cuando aporten contexto y evita repetir literalmente las frases del historial.
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
    Factory that generates customer agents from profiles, without additional LLM calls.
    """

    def __init__(self, *, default_channel: str = "whatsapp", **_: object) -> None:
        self.default_channel = default_channel

    def create_agent(self, profile: Dict) -> CustomerAgent:
        """
        Build a persona-specific agent prompt using the profile payload.
        """
        persona = _extract_persona(profile)
        system_prompt = _build_system_prompt(persona, profile)
        starting_message = _initial_customer_message(profile, persona)
        if starting_message:
            profile["_initial_customer_message"] = starting_message
            starting_message = None

        return CustomerAgent(
            customer_id=profile.get("customer_id", persona.get("nombre", "unknown")),
            system_prompt=system_prompt,
            starting_message=starting_message,
            profile=profile,
        )


def _extract_persona(profile: Dict) -> Dict:
    human = profile.get("human_simulacra")
    if isinstance(human, dict):
        return human

    persona = profile.get("persona", {})
    if persona:
        return {
            "nombre": persona.get("name"),
            "edad": persona.get("age"),
            "ciudad": persona.get("location"),
            "ocupacion": persona.get("occupation"),
            "historia_revelada": persona.get("bio"),
            "historia_oculta": persona.get("bio"),
            "es_vocal": profile.get("cohort", {}).get("vocal", False),
            "satisfaccion": "Satisfecho" if profile.get("cohort", {}).get("satisfied", True) else "Insatisfecho",
            "problema": "",
            "expectativa_solucion": "",
            "prompt_conversacional": "",
        }
    return {}


def _build_system_prompt(persona: Dict, profile: Dict) -> str:
    """Compose deterministic system prompt for the customer."""
    lines = [
        CUSTOMER_SYSTEM_SEED.strip(),
        "",
        "### Contexto personal",
        f"- Nombre: {persona.get('nombre', 'Cliente Kavak')}",
        f"- Edad: {persona.get('edad', 'N/D')}",
        f"- Ciudad: {persona.get('ciudad', 'N/D')}",
        f"- Ocupación: {persona.get('ocupacion', 'N/D')}",
        f"- Relación con Kavak: {persona.get('relacion_kavak', profile.get('purchase', {}).get('vehicle', 'cliente'))}",
        "",
        "### Historia revelada",
        persona.get("historia_revelada", "Sin historial detallado."),
        "",
        "### Historia oculta",
        persona.get("historia_oculta", "Sin detalles adicionales."),
        "",
        "### Sentimiento actual",
        f"- Estado: {'Satisfecho' if profile.get('cohort', {}).get('satisfied', True) else 'Insatisfecho'}",
        "",
    ]

    problema = persona.get("problema")
    if problema:
        lines.extend(["### Problema principal", problema, ""])

    expectativa = persona.get("expectativa_solucion")
    if expectativa:
        lines.extend(["### Expectativa de solución", expectativa, ""])

    historial = persona.get("historial_vocalidad") or []
    if historial:
        lines.append("### Historial de vocalidad relevante")
        for registro in historial[:3]:
            canal = registro.get("canal", "canal")
            resumen = registro.get("resumen", "")
            nps_reg = registro.get("nps")
            suffix = f" (NPS {nps_reg})" if nps_reg is not None else ""
            lines.append(f"- {canal}: {resumen}{suffix}")
        lines.append("")

    initial_context = profile.get("_initial_customer_message")
    if initial_context:
        lines.extend([
            "### Expectativas expresadas en registros previos",
            initial_context,
            "Utiliza esta referencia para responder de forma natural. No la cites ni la repitas textualmente en tus mensajes.",
            "",
        ])

    prompt_extra = persona.get("prompt_conversacional")
    if prompt_extra:
        lines.extend(["### Instrucciones específicas", prompt_extra, ""])

    lines.append(
        "Responde siempre como cliente. Expresa necesidades, emociones y dudas. "
        "No otorgues soluciones operativas ni confirmes acciones que dependen del equipo de Kavak."
    )

    return "\n".join(lines).strip()


def _initial_customer_message(profile: Dict, persona: Dict) -> Optional[str]:
    messages = profile.get("history", {}).get("messages", [])
    for msg in messages:
        if msg.get("role") == "customer" and msg.get("content"):
            return msg["content"]

    if persona.get("problema"):
        return persona["problema"]

    if persona.get("expectativa_solucion"):
        return persona["expectativa_solucion"]

    return None


__all__ = ["CustomerAgentFactory", "CustomerAgent"]
