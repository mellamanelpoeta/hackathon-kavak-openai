"""
Persona generation pipeline leveraging the OpenAI Responses API.
Generates personas that comply with the Persona schema defined in schemas.py.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from dotenv import load_dotenv
from openai import APIError, AsyncOpenAI
from pydantic import ValidationError

from .schemas import Persona


DEFAULT_MODEL = "gpt-4.1-mini"
PERSONA_SCHEMA = Persona

BASE_SYSTEM_PROMPT = """Eres un generador de datos sintéticos para simulaciones de atención al cliente de Kavak.
Devuelves exclusivamente un JSON válido que cumpla exactamente con el esquema Persona documentado.
Asegúrate de que todos los campos requeridos están presentes, que los datos sean plausibles para México y respeten las reglas de validación."""

load_dotenv()


@dataclass(frozen=True)
class PersonaPrompt:
    """Prompt definition for a specific persona archetype."""

    name: str
    instructions: str

    def build_messages(self) -> list[dict[str, object]]:
        return [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": BASE_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": self.instructions}],
            },
        ]


PROMPTS: dict[str, PersonaPrompt] = {
    "vocal_satisfecho": PersonaPrompt(
        name="Cliente vocal satisfecho",
        instructions="""Genera un JSON de una persona que mantiene relación de compra vigente con Kavak, es vocal y actualmente satisfecha.
Requisitos explícitos:
- es_vocal debe ser true y satisfaccion "Satisfecho".
- Historia revelada debe incluir fechas específicas recientes (en los últimos 90 días) y detalles de vehículo.
- Historial de vocalidad debe contener al menos un registro en canal app o whatsapp con NPS alto (>=8).
- Expectativa de solución debe ser positiva, orientada a continuidad.""",
    ),
    "vocal_insatisfecho": PersonaPrompt(
        name="Cliente vocal insatisfecho",
        instructions="""Genera un JSON de una persona con relación compra_venta que es vocal e insatisfecha por un problema de logística o mecánica.
Requisitos explícitos:
- es_vocal true, satisfaccion "Insatisfecho", problema describiendo el dolor actual.
- Historial de vocalidad con al menos dos registros en distintos canales, incluyendo uno con nps <=5.
- Historia oculta debe revelar factores emocionales que expliquen su molestia.
- Prompt conversacional debe orientar al agente a contener la frustración y acordar próximos pasos concretos.""",
    ),
    "no_vocal_alta_ltv": PersonaPrompt(
        name="Cliente no vocal de alto valor",
        instructions="""Genera un JSON de una persona con relación compra, no vocal pero con alto LTV (>=450000) y señales de posible upsell.
Requisitos explícitos:
- es_vocal false, satisfaccion null, historial_vocalidad vacío.
- Historia revelada debe mostrar compras pasadas exitosas y preferencia de comunicación.
- Expectativa de solución debe describir lo que valoraría para reactivar contacto proactivo.""",
    ),
    "no_vocal_riesgo": PersonaPrompt(
        name="Cliente no vocal en riesgo",
        instructions="""Genera un JSON de una persona con relación venta que no es vocal, pero cuyo historial sugiere riesgo latente.
Requisitos explícitos:
- es_vocal false, satisfaccion null.
- Historia oculta debe incluir un problema interno que aún no comunica a Kavak.
- Historia revelada debe mostrar seguimiento incompleto o tareas pendientes.
- Expectativa de solución debe enfatizar qué necesitaría para confiar en Kavak nuevamente.""",
    ),
}


class PersonaGenerationError(Exception):
    """Raised when persona generation fails after retries."""


class PersonaGenerator:
    """Async persona generator leveraging OpenAI Responses API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        concurrent_requests: int = 8,
        client_factory: Callable[[], AsyncOpenAI] | None = None,
    ) -> None:
        self.model = model
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        self._client_factory = client_factory

    def _client(self) -> AsyncOpenAI:
        if not hasattr(self, "_client_instance"):
            factory = self._client_factory or AsyncOpenAI
            setattr(self, "_client_instance", factory())
        return getattr(self, "_client_instance")

    async def _generate_single(self, prompt: PersonaPrompt, retry: int = 3) -> Persona:
        for attempt in range(1, retry + 1):
            try:
                async with self.semaphore:
                    client = self._client()
                    response = await client.responses.parse(
                        model=self.model,
                        input=prompt.build_messages(),
                        text_format=Persona
                    )
                parsed = response.output_parsed
                if isinstance(parsed, Persona):
                    persona = parsed
                elif isinstance(parsed, str):
                    data = json.loads(parsed)
                    persona = Persona.model_validate(data)
                else:
                    persona = Persona.model_validate(parsed)
                return persona
            except (APIError, ValidationError, json.JSONDecodeError, KeyError, IndexError) as exc:
                if attempt == retry:
                    raise PersonaGenerationError(str(exc)) from exc
                await asyncio.sleep(2**attempt * 0.2)
        raise PersonaGenerationError("Exhausted retries")

    async def generate_batch(self, prompt_key: str, total: int) -> list[Persona]:
        prompt = PROMPTS[prompt_key]
        tasks = [asyncio.create_task(self._generate_single(prompt)) for _ in range(total)]
        personas: list[Persona] = []
        for task in asyncio.as_completed(tasks):
            personas.append(await task)
        return personas


async def generate_all(
    output_root: str | Path,
    per_prompt: int = 250,
    prompts: Iterable[str] | None = None,
    model: str = DEFAULT_MODEL,
    concurrent_requests: int = 8,
) -> None:
    """Generate personas for the given prompt keys and persist as JSON."""
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    generator = PersonaGenerator(model=model, concurrent_requests=concurrent_requests)
    targets = list(prompts) if prompts else list(PROMPTS.keys())

    for prompt_key in targets:
        personas = await generator.generate_batch(prompt_key, per_prompt)
        folder = out_root / prompt_key
        folder.mkdir(parents=True, exist_ok=True)
        for idx, persona in enumerate(personas, start=1):
            payload = persona.model_dump(mode="json")
            with (folder / f"{prompt_key}_{idx:04d}.json").open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate HumanSimulacra personas.")
    parser.add_argument("--output", default="personas_output", help="Directorio raíz para los JSON.")
    parser.add_argument("--per-type", type=int, default=250, help="Personas por tipo de prompt.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo Responses a utilizar.")
    parser.add_argument("--concurrency", type=int, default=8, help="Solicitudes simultáneas.")
    parser.add_argument(
        "--types",
        nargs="*",
        default=list(PROMPTS.keys()),
        choices=list(PROMPTS.keys()),
        help="Tipos de prompt a generar.",
    )

    args = parser.parse_args()
    asyncio.run(
        generate_all(
            output_root=args.output,
            per_prompt=args.per_type,
            prompts=args.types,
            model=args.model,
            concurrent_requests=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
