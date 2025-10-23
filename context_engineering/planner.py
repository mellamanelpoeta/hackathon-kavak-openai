"""
Planner agent responsible for selecting strategy and crafting proactive prompt seeds.
"""
from __future__ import annotations

from typing import Dict, Optional

from app.factories.agents_runner import AgentsRunner

from .conversation import StrategyPlan
from .prompts import PLANNER_SYSTEM_PROMPT, PROACTIVE_AGENT_TEMPLATE
from .strategies import get_strategy


class PlannerAgent:
    """LLM agent that generates StrategyPlan objects."""

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
        profile: Dict,
        cohort_summary: Optional[Dict] = None,
        history_notes: Optional[str] = None,
    ) -> StrategyPlan:
        """Generate a strategy plan for a specific customer profile."""
        user_prompt = self._build_prompt(profile, cohort_summary, history_notes)
        result = self.runner.run_json(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_content=user_prompt,
        )

        strategy_id = result.get("strategy_id", "Sin_Accion")
        tone = result.get("tone", "empático-directo")
        objectives = result.get("objectives", ["Restaurar confianza", "Clarificar próximo paso"])
        max_turns = int(result.get("max_turns", 3))
        end_triggers = result.get("end_triggers", ["END"])
        prompt_seed_extension = result.get("prompt_seed", "").strip()

        strategy_def = get_strategy(strategy_id)

        prompt_seed = (
            f"{PROACTIVE_AGENT_TEMPLATE.strip()}\n\n"
            f"Estrategia seleccionada: {strategy_def.nombre_estrategia}\n"
            f"Razonamiento: {strategy_def.razonamiento_estrategia}\n"
            f"Acción puntual sugerida: {strategy_def.accion_puntual}\n"
        )
        if prompt_seed_extension:
            prompt_seed += f"\nInstrucciones adicionales:\n{prompt_seed_extension}"

        return StrategyPlan(
            prompt_seed=prompt_seed,
            objectives=objectives,
            tone=tone,
            strategy_id=strategy_id,
            max_turns=max_turns,
            end_triggers=end_triggers,
        )

    def _build_prompt(
        self,
        profile: Dict,
        cohort_summary: Optional[Dict],
        history_notes: Optional[str],
    ) -> str:
        persona = profile.get("persona", {})
        purchase = profile.get("purchase", {})
        history = profile.get("history", {})
        risk = profile.get("risk_signals", {})
        cohort = profile.get("cohort", {})

        lines = [
            "## PERFIL DEL CLIENTE",
            f"ID: {profile.get('customer_id')}",
            f"Cohorte: {cohort}",
            f"Persona: {persona}",
            f"Compra: {purchase}",
            f"Historial: {history}",
            f"Riesgos: {risk}",
        ]

        if cohort_summary:
            lines.append("\n## RESUMEN COHORTE")
            lines.append(str(cohort_summary))

        if history_notes:
            lines.append("\n## HISTÓRICO DE APRENDIZAJES")
            lines.append(history_notes)

        lines.append("\nProporciona el JSON con la nueva estrategia siguiendo el formato requerido.")
        return "\n".join(lines)


__all__ = ["PlannerAgent"]
