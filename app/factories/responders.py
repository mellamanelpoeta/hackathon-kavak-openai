"""
Responder and Outreach agents for message generation.
Uses OpenAI Agents (Responses API) to produce customer-facing messages.
"""
from typing import Optional

from app.factories.agents_runner import AgentsRunner
from app.models import Context


RESPONDER_SYSTEM_PROMPT = """
Eres un agente de atención al cliente de Kavak.

OBJETIVO: Maximizar NPS esperado, reducir churn y aumentar engagement.

Usa UNA de las plantillas disponibles y rellena los slots con el contexto del cliente.

REGLAS:
- Tono respetuoso y empático
- NO culpar al cliente NUNCA
- NO hacer promesas legales o garantías imposibles
- NO dar instrucciones técnicas complejas sin soporte
- Claridad en próxima acción (next step concreto)
- Si corresponde, sugiere escalar a mecánica/finanzas/logística con SLA concreto
- Máximo 150 palabras

Si el cliente está enojado:
- Reconocer su frustración explícitamente
- Disculpa breve y sincera
- Acción inmediata o escalación con timeline

Devuelve SOLO el texto final para el cliente (sin JSON, sin metadatos, sin explicaciones).
El mensaje debe estar listo para enviar por el canal preferido del cliente.
"""


OUTREACH_SYSTEM_PROMPT = """
Eres un agente de outreach proactivo de Kavak.

OBJETIVO: Maximizar apertura y respuesta en contacto one-shot (una única oportunidad).

Usa UNA de las plantillas para crear un mensaje personalizado que:
- Capte atención en las primeras líneas
- Sea relevante al contexto del cliente (recencia, monto, posibles issues)
- Ofrezca valor claro (información, soporte, seguimiento)
- Tenga un call-to-action específico y fácil

REGLAS:
- Tono profesional pero cercano
- NO ser invasivo o insistente
- Personalizar con datos concretos (días desde compra, vehículo, etc.)
- Claridad sobre el motivo del contacto
- Next step claro y de bajo esfuerzo para el cliente
- Máximo 120 palabras

Devuelve SOLO el texto final para enviar (sin JSON, sin metadatos).
El mensaje debe estar listo para el canal preferido del cliente.
"""


class ResponderAgent:
    """Generates responses to vocal customers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.4,
        max_tokens: int = 300,
    ):
        """
        Initialize Responder Agent.

        Args:
            api_key: OpenAI API key
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum output tokens for response
        """
        self.runner = AgentsRunner(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    def run(self, context: Context, template_text: str) -> str:
        """
        Generate response message.

        Args:
            context: Customer context
            template_text: Template prompt from TemplateFactory

        Returns:
            Final message text
        """
        try:
            return self.runner.run_text(
                system_prompt=RESPONDER_SYSTEM_PROMPT,
                user_content=template_text,
            )
        except Exception as exc:
            print(f"Responder error: {exc}")
            return self._fallback_message(context)

    def _fallback_message(self, context: Context) -> str:
        """Fallback message when LLM fails."""
        return f"""
Estimado cliente,

Hemos recibido tu mensaje y estamos trabajando en resolverlo.

Un miembro de nuestro equipo se pondrá en contacto contigo en las próximas 24 horas por {context.channel_pref}.

Gracias por tu paciencia.

Atentamente,
Equipo Kavak
""".strip()


class OutreachAgent:
    """Generates proactive outreach messages to non-vocal customers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.5,
        max_tokens: int = 250,
    ):
        """
        Initialize Outreach Agent.

        Args:
            api_key: OpenAI API key
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum output tokens for response
        """
        self.runner = AgentsRunner(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    def run(self, context: Context, template_text: str) -> str:
        """
        Generate outreach message.

        Args:
            context: Customer context
            template_text: Template prompt from TemplateFactory

        Returns:
            Final message text
        """
        try:
            return self.runner.run_text(
                system_prompt=OUTREACH_SYSTEM_PROMPT,
                user_content=template_text,
            )
        except Exception as exc:
            print(f"Outreach error: {exc}")
            return self._fallback_message(context)

    def _fallback_message(self, context: Context) -> str:
        """Fallback message when LLM fails."""
        return f"""
Hola,

Esperamos que estés disfrutando tu vehículo.

Queremos asegurarnos de que todo esté bien. Si tienes alguna duda o necesitas soporte, estamos aquí para ayudarte.

Responde a este mensaje o contáctanos por {context.channel_pref}.

Saludos,
Equipo Kavak
""".strip()
