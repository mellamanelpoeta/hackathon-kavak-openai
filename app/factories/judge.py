"""
Judge: LLM-based evaluator using OpenAI Agents (Responses API).
Returns structured scores for NPS, engagement, churn, and sentiment.
"""
from typing import Optional

from app.factories.agents_runner import AgentsRunner
from app.models import Context, Score


JUDGE_SYSTEM_PROMPT = """
Eres un evaluador experto de mensajes de atención al cliente para Kavak.

Tu tarea es evaluar un mensaje dado el perfil y el historial del cliente.

Devuelve JSON ESTRICTO con esta estructura exacta:
{
  "NPS_expected": <float 0-10>,
  "EngagementProb": <float 0-1>,
  "ChurnProb": <float 0-1>,
  "AspectSentiment": {
    "finanzas": <float -1 a 1>,
    "mecanica": <float -1 a 1>,
    "logistica": <float -1 a 1>,
    "atencion": <float -1 a 1>
  },
  "rationale": "<string máximo 280 caracteres>"
}

CRITERIOS DE EVALUACIÓN:

NPS_expected (0-10):
- 0-6: Cliente muy insatisfecho, mensaje frío o poco útil
- 7-8: Cliente neutral/pasivo, mensaje correcto pero sin destacar
- 9-10: Cliente promotor, mensaje excelente que resuelve y empatiza

EngagementProb (0-1):
- Probabilidad de que el cliente responda o tome acción
- Considera: claridad del next step, personalización, tono apropiado
- Penaliza: mensajes genéricos, sin acción clara, tono inadecuado

ChurnProb (0-1):
- Probabilidad de que el cliente abandone Kavak
- Considera: contexto (cliente enojado, NPS bajo, issue grave)
- Penaliza: mensajes que no resuelven, culpan al cliente, son fríos con clientes enojados
- Bonifica: empatía, soluciones concretas, SLA claro

AspectSentiment (-1 a 1):
- Para cada aspecto, evalúa el sentimiento transmitido por el mensaje
- -1: muy negativo, 0: neutral, 1: muy positivo
- Considera cómo el mensaje aborda cada área

Rationale (<=280 chars):
- Explica brevemente la evaluación
- Menciona fortalezas y áreas de mejora

IMPORTANTE:
- Penaliza frialdad con clientes enojados
- Valora claridad y acción concreta
- Bonifica SLA específico cuando aplique
- Ajusta expectativas según contexto (cliente vocal vs outreach)
"""


class Judge:
    """LLM-based message evaluator."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.3,
        max_retries: int = 3,
        max_tokens: int = 500,
    ):
        """
        Initialize Judge.

        Args:
            api_key: OpenAI API key (or from env)
            model: Model to use for evaluation
            temperature: Sampling temperature
            max_retries: Number of retries for failed requests
            max_tokens: Maximum output tokens for evaluation
        """
        self.runner = AgentsRunner(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            max_retries=max_retries,
        )

    def run(self, context: Context, message: str) -> Score:
        """
        Evaluate a message given customer context.

        Args:
            context: Customer context
            message: Generated message to evaluate

        Returns:
            Score object with evaluation metrics
        """
        user_prompt = self._build_prompt(context, message)

        try:
            score_dict = self.runner.run_json(
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_content=user_prompt,
            )
            return Score(**score_dict)
        except Exception as exc:
            print(f"Judge evaluation failed: {exc}")
            return self._default_score()

    def _build_prompt(self, context: Context, message: str) -> str:
        """Build evaluation prompt from context and message."""
        prompt = f"""
CONTEXTO DEL CLIENTE:
- ID: {context.customer_id}
- Segmento: {context.segment}
- NPS previo: {context.past_NPS}/10
- Riesgo de churn: {context.churn_risk_est:.0%}
- Historia: {context.mini_story}
- Días desde compra: {context.last_purchase_days}
- Precio: ${context.price:,.0f}
- Categoría: {context.issue_bucket}
- Canal preferido: {context.channel_pref}
"""

        if context.first_message:
            prompt += f"\n- Mensaje inicial del cliente: \"{context.first_message}\"\n"

        prompt += f"""
MENSAJE A EVALUAR:
\"\"\"
{message}
\"\"\"

Evalúa este mensaje y devuelve el JSON con los scores.
"""

        return prompt

    def _default_score(self) -> Score:
        """Return default score when evaluation fails."""
        return Score(
            NPS_expected=5.0,
            EngagementProb=0.5,
            ChurnProb=0.5,
            AspectSentiment={
                "finanzas": 0.0,
                "mecanica": 0.0,
                "logistica": 0.0,
                "atencion": 0.0
            },
            rationale="Evaluación por defecto (error en Judge)"
        )
