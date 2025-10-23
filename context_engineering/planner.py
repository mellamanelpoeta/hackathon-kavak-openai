"""
Planner agent responsible for selecting strategy and crafting proactive prompt seeds.
Leverages persisted insights (strategy performance, prompt overrides) to bias decisions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.factories.agents_runner import AgentsRunner

from .conversation import StrategyPlan
from .profile_utils import infer_cohort_label
from .prompts import PLANNER_SYSTEM_PROMPT, PROACTIVE_AGENT_TEMPLATE
from .persistence import load_prompt_overrides, load_strategy_insights
from .strategies import STRATEGY_IDS, get_strategy

MAX_PLANNER_ATTEMPTS = 2


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
        invalid_strategies: List[str] = []
        prompt_overrides = load_prompt_overrides()
        strategy_insights = load_strategy_insights()
        cohort_label = infer_cohort_label(profile)
        initial_context = profile.get("_initial_customer_message")

        for attempt in range(MAX_PLANNER_ATTEMPTS):
            user_prompt = self._build_prompt(
                profile=profile,
                cohort_summary=cohort_summary,
                history_notes=history_notes,
                cohort_label=cohort_label,
                initial_context=initial_context,
                prompt_overrides=prompt_overrides,
                strategy_insights=strategy_insights,
                invalid_strategies=invalid_strategies,
            )
            result = self.runner.run_json(
                system_prompt=PLANNER_SYSTEM_PROMPT,
                user_content=user_prompt,
            )

            strategy_id = (result.get("strategy_id") or "").strip()
            if strategy_id in STRATEGY_IDS:
                tone = result.get("tone", "empático-directo")
                objectives = result.get(
                    "objectives", ["Restaurar confianza", "Clarificar próximo paso"]
                )
                max_turns = int(result.get("max_turns", 3))
                end_triggers = result.get("end_triggers", ["END"])
                prompt_seed_extension = result.get("prompt_seed", "").strip()
                break

            invalid_strategies.append(strategy_id or "(vacío)")
        else:
            raise ValueError(
                f"Estrategia no reconocida tras {MAX_PLANNER_ATTEMPTS} intentos: {invalid_strategies}"
            )

        strategy_def = get_strategy(strategy_id)

        prompt_seed = (
            f"{PROACTIVE_AGENT_TEMPLATE.strip()}\n\n"
            f"Estrategia seleccionada: {strategy_def.nombre_estrategia}\n"
            f"Razonamiento: {strategy_def.razonamiento_estrategia}\n"
            f"Acción puntual sugerida: {strategy_def.accion_puntual}\n"
        )
        if prompt_seed_extension:
            prompt_seed += f"\nInstrucciones adicionales:\n{prompt_seed_extension}"

        prompt_seed = self._apply_initiative_overrides(
            prompt_seed,
            cohort_label=cohort_label,
            initial_context=initial_context,
            prompt_overrides=prompt_overrides,
            strategy_insights=strategy_insights,
        )

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
        *,
        cohort_label: str,
        initial_context: Optional[str],
        prompt_overrides: Dict[str, Any],
        strategy_insights: Dict[str, Any],
        invalid_strategies: Optional[List[str]] = None,
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
            f"Cohorte derivada: {cohort_label}",
            "\n## ESTRATEGIAS DISPONIBLES",
            ", ".join(STRATEGY_IDS),
        ]

        if cohort_summary:
            lines.append("\n## RESUMEN COHORTE")
            lines.append(str(cohort_summary))

        if history_notes:
            lines.append("\n## HISTÓRICO DE APRENDIZAJES")
            lines.append(history_notes)

        if initial_context:
            lines.append("\n## EXPECTATIVA RECIENTE DEL CLIENTE")
            lines.append(initial_context)

        overall_insight = strategy_insights.get("overall", {})
        overall_strategy = overall_insight.get("strategy")
        if overall_strategy:
            metrics = overall_insight.get("metrics", {})
            lines.append(
                f"\n## INSIGHT GLOBAL\n"
                f"- Estrategia destacada: {overall_strategy} "
                f"(Δ LTV {metrics.get('ltv_gain_avg', 0):.2f}, reward {metrics.get('reward_avg', 0):.3f})."
            )

        cohort_insight = strategy_insights.get("best_by_cohort", {}).get(cohort_label)
        if cohort_insight:
            metrics = cohort_insight.get("metrics", {})
            lines.append(
                f"\n## INSIGHT COHORTE\n"
                f"- La cohorte {cohort_label} ha respondido mejor a {cohort_insight['strategy']} "
                f"(Δ LTV {metrics.get('ltv_gain_avg', 0):.2f}, reward {metrics.get('reward_avg', 0):.3f})."
            )

        planner_notes = self._collect_overrides(prompt_overrides.get("planner", {}), cohort_label)
        if planner_notes:
            lines.append("\n## APRENDIZAJES RECIENTES PARA EL PLANNER")
            lines.extend(f"- {note}" for note in planner_notes)

        if invalid_strategies:
            lines.append(
                "\nEl intento previo devolvió estrategias inválidas; "
                "selecciona exactamente uno de los identificadores anteriores."
            )
            lines.append(f"Estrategias inválidas previas: {invalid_strategies}")

        lines.append("\nProporciona el JSON con la nueva estrategia siguiendo el formato requerido.")
        return "\n".join(lines)

    @staticmethod
    def _collect_overrides(section: Dict[str, Any], cohort_label: str) -> List[str]:
        notes: List[str] = []
        global_notes = section.get("global", [])
        cohort_notes = section.get("cohorts", {}).get(cohort_label, [])
        for item in global_notes + cohort_notes:
            if item and item not in notes:
                notes.append(item)
        return notes

    def _apply_initiative_overrides(
        self,
        prompt_seed: str,
        *,
        cohort_label: str,
        initial_context: Optional[str],
        prompt_overrides: Dict[str, Any],
        strategy_insights: Dict[str, Any],
    ) -> str:
        notes = self._collect_overrides(prompt_overrides.get("initiative", {}), cohort_label)

        cohort_insight = strategy_insights.get("best_by_cohort", {}).get(cohort_label)
        if cohort_insight:
            metrics = cohort_insight.get("metrics", {})
            highlight = (
                f"Histórico: la estrategia '{cohort_insight['strategy']}' logró Δ LTV "
                f"{metrics.get('ltv_gain_avg', 0):.2f} y reward {metrics.get('reward_avg', 0):.3f} en esta cohorte."
            )
            if highlight not in notes:
                notes.append(highlight)

        if initial_context:
            context_note = (
                f"El cliente expresó recientemente: {initial_context}. Usa esta referencia sin citarla literalmente."
            )
            if context_note not in notes:
                notes.append(context_note)

        if notes:
            prompt_seed += "\n\nNotas recientes para el agente:\n" + "\n".join(f"- {note}" for note in notes)

        return prompt_seed


__all__ = ["PlannerAgent"]
