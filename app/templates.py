"""
Template definitions for customer service messages.
Based on the 5 template types defined in AGENTS.md.
"""
from app.models import Template


TEMPLATES = [
    Template(
        id="empatiko",
        name="Empático breve",
        slots=["saludo", "reconocimiento_issue", "accion", "next_step", "firma"],
        guardrails=[
            "no culpar al cliente",
            "ofrece disculpa breve",
            "claridad en próxima acción"
        ],
        template_text="""
{saludo},

{reconocimiento_issue}

{accion}

{next_step}

{firma}
"""
    ),
    Template(
        id="tecnico",
        name="Técnico solución",
        slots=["saludo", "diagnostico", "accion_tecnica", "next_step", "firma"],
        guardrails=[
            "no hacer promesas técnicas imposibles",
            "ser específico con pasos técnicos",
            "incluir SLA cuando aplique"
        ],
        template_text="""
{saludo},

{diagnostico}

{accion_tecnica}

{next_step}

{firma}
"""
    ),
    Template(
        id="cupon",
        name="Compensación/cupón",
        slots=["saludo", "reconocimiento_issue", "oferta_compensacion", "next_step", "firma"],
        guardrails=[
            "no exceder límites de compensación",
            "ser claro sobre términos",
            "mantener tono profesional"
        ],
        template_text="""
{saludo},

{reconocimiento_issue}

{oferta_compensacion}

{next_step}

{firma}
"""
    ),
    Template(
        id="escalar",
        name="Escalación interna",
        slots=["saludo", "reconocimiento_issue", "derivacion_area", "sla", "firma"],
        guardrails=[
            "ser específico sobre el área de escalación",
            "proporcionar SLA concreto",
            "mantener expectativas realistas"
        ],
        template_text="""
{saludo},

{reconocimiento_issue}

{derivacion_area}

{sla}

{firma}
"""
    ),
    Template(
        id="seguimiento",
        name="Seguimiento proactivo",
        slots=["saludo", "resumen", "next_touch", "canal", "firma"],
        guardrails=[
            "resumir situación actual",
            "ser específico sobre próximo contacto",
            "confirmar canal preferido"
        ],
        template_text="""
{saludo},

{resumen}

{next_touch}

{canal}

{firma}
"""
    )
]


def get_template(template_id: str) -> Template:
    """Get template by ID."""
    for template in TEMPLATES:
        if template.id == template_id:
            return template
    raise ValueError(f"Template {template_id} not found")


def get_template_ids() -> list[str]:
    """Get list of all template IDs."""
    return [t.id for t in TEMPLATES]
