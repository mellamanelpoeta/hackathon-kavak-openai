"""
Schemas and helpers for realistic Kavak customer personas.

Generators are imported lazily to avoid requiring the OpenAI SDK
when only the schemas are needed.
"""

from .schemas import Persona, RegistroVocalidad

__all__ = ["Persona", "RegistroVocalidad", "PersonaGenerator", "generate_all", "PROMPTS"]


def __getattr__(name):  # pragma: no cover - lazy import
    if name in {"PersonaGenerator", "generate_all", "PROMPTS"}:
        from .generator import PersonaGenerator, PROMPTS, generate_all

        globals().update(
            {
                "PersonaGenerator": PersonaGenerator,
                "generate_all": generate_all,
                "PROMPTS": PROMPTS,
            }
        )
        return globals()[name]
    raise AttributeError(name)
