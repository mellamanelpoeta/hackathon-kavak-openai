"""
Template Factory: Fills template slots with customer context.
Provides base drafts for agents to refine.
"""
from app.models import Context, Template
from app.templates import get_template


class TemplateFactory:
    """Fills template slots with customer-specific information."""

    @staticmethod
    def fill(template_id: str, context: Context) -> str:
        """
        Fill template with context information.

        Args:
            template_id: ID of template to use
            context: Customer context

        Returns:
            Draft message with template structure
        """
        template = get_template(template_id)

        # Create a prompt that describes the template and context
        # This will be refined by the agent
        prompt = f"""
Usa la plantilla '{template.name}' (ID: {template.id}) para responder.

CONTEXTO DEL CLIENTE:
- ID: {context.customer_id}
- Segmento: {context.segment}
- Historia: {context.mini_story}
- Días desde compra: {context.last_purchase_days}
- Precio pagado: ${context.price:,.0f}
- NPS previo: {context.past_NPS}/10
- Riesgo de churn: {context.churn_risk_est:.0%}
- Canal preferido: {context.channel_pref}
- Categoría: {context.issue_bucket}
"""

        if context.first_message:
            prompt += f"\n- Mensaje del cliente: \"{context.first_message}\"\n"

        prompt += f"""
ESTRUCTURA DE LA PLANTILLA:
Slots requeridos: {', '.join(template.slots)}

GUARDRAILS:
{chr(10).join(f'- {g}' for g in template.guardrails)}

INSTRUCCIONES:
Rellena todos los slots de manera natural y personalizada según el contexto.
Mantén el mensaje breve (máx 150 palabras), empático y orientado a acción.
"""

        return prompt
