"""
Example runner showing how to generate customer agents, execute conversations,
score outcomes, and compute expected LTV deltas.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from app.factories.judge import Judge

from .agents_factory import CustomerAgentFactory
from .conversation import ProactiveConversationAgent, StrategyPlan
from .ltv import evaluate_conversation
from .planner import PlannerAgent
from .profile_utils import profile_to_context
from .strategies import get_strategy


def load_profiles(path: Path) -> List[dict]:
    """Load all JSON profiles from a directory."""
    profiles: List[dict] = []
    for json_file in sorted(path.glob("*.json")):
        with json_file.open("r", encoding="utf-8") as f:
            profiles.append(json.load(f))
    return profiles


def run_iteration(
    profiles: Iterable[dict],
    *,
    proactive_prompt: Optional[str] = None,
    objectives: Optional[List[str]] = None,
    strategy_id: Optional[str] = None,
    tone: str = "empático-directo",
    max_turns: int = 3,
    end_triggers: Optional[List[str]] = None,
    judge_model: str = "gpt-4.1-mini",
    run_number: int = 1,
    strategy_attempt_id: int = 1,
    message_attempt_id: int = 1,
    planner: Optional[PlannerAgent] = None,
    cohort_summary: Optional[Dict] = None,
    history_notes: Optional[str] = None,
    api_key: str | None = None,
) -> List[dict]:
    """
    High-level helper to:
      1. Build customer agents from profiles
      2. Run a conversation per profile using the provided strategy
      3. Evaluate results with Judge + LTV scoring
    Returns list of summaries per customer.
    """
    factory = CustomerAgentFactory(api_key=api_key)
    orchestrator = ProactiveConversationAgent(api_key=api_key)
    judge = Judge(api_key=api_key, model=judge_model)
    planner_agent = planner or PlannerAgent(api_key=api_key)

    summaries: List[dict] = []

    for profile in profiles:
        plan = _resolve_plan(
            profile=profile,
            planner=planner_agent,
            proactive_prompt=proactive_prompt,
            objectives=objectives,
            strategy_id=strategy_id,
            tone=tone,
            max_turns=max_turns,
            end_triggers=end_triggers,
            cohort_summary=cohort_summary,
            history_notes=history_notes,
        )

        strategy_def = get_strategy(plan.strategy_id)

        customer_agent = factory.create_agent(profile)
        result = orchestrator.run_conversation(
            customer_agent=customer_agent,
            plan=plan,
            profile=profile,
        )

        final_agent_message = next(
            (turn.content for turn in reversed(result.turns) if turn.role == "agent"),
            "",
        )

        ctx = profile_to_context(profile)
        score = judge.run(ctx, final_agent_message)

        metrics = evaluate_conversation(
            profile=profile,
            strategy=strategy_def,
            score=score,
        )

        nps_original = profile.get("history", {}).get("past_nps")
        if nps_original is not None:
            try:
                nps_original = int(nps_original)
            except (TypeError, ValueError):
                nps_original = None

        ltv_original = metrics["ltv_apriori"]
        ltv_final = metrics["ltv_expected"]
        ganancia_ltv = ltv_final - ltv_original

        resultado = (
            "cierre_positivo"
            if (score.NPS_expected >= 8 and score.EngagementProb >= 0.6)
            else "cierre_no_positivo"
        )

        cohort = profile.get("cohort", {})
        ctx_issue_bucket = ctx.issue_bucket

        record = {
            "client_id": result.customer_id,
            "nps_og": nps_original,
            "vocal": bool(profile.get("cohort", {}).get("vocal", False)),
            "satisfecho": bool(profile.get("cohort", {}).get("satisfied", True)),
            "cohort_label": _format_cohort_label(cohort),
            "run_number": run_number,
            "estrategia_intentada": strategy_attempt_id,
            "mensaje_intentado": message_attempt_id,
            "NPS_final": int(round(score.NPS_expected)),
            "LTV_og": float(ltv_original),
            "LTV_final": float(ltv_final),
            "engagement": float(score.EngagementProb),
            "resultado": resultado,
            "ganancia_LTV": float(ganancia_ltv),
            "costo_estrategia": float(strategy_def.costo),
            "reward": float(metrics["reward"]),
            "strategy_name": strategy_def.nombre_estrategia,
            "strategy_rationale": strategy_def.razonamiento_estrategia,
            "issue_bucket": ctx_issue_bucket,
            "mini_story": ctx.mini_story,
            "channel_pref": ctx.channel_pref,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        summaries.append(record)

        print("=" * 60)
        print(
            f"Customer: {result.customer_id} | Strategy: {result.strategy_id} "
            f"| Outcome: {result.outcome}"
        )
        for turn in result.turns:
            speaker = "Agente" if turn.role == "agent" else "Cliente"
            print(f"[{speaker}] {turn.content}")
        print(
            f"→ NPS esperado: {score.NPS_expected:.2f} | Engagement: {score.EngagementProb:.2%} "
            f"| Churn: {score.ChurnProb:.2%}"
        )
        print(
            f"→ LTV esperado: ${metrics['ltv_expected']:.2f} "
            f"(Δ: {metrics['ltv_expected'] - metrics['ltv_apriori']:.2f}) | Reward: {metrics['reward']:.3f}"
        )
        print(f"→ Strategy rationale: {strategy_def.razonamiento_estrategia}")
        print(f"→ Judge rationale: {score.rationale}")
        print(f"→ Resultado heurístico: {record['resultado']} | Ganancia LTV: {ganancia_ltv:.2f}")
        print()

    return summaries


if __name__ == "__main__":
    # Example usage: python -m context_engineering.example_runner profiles/
    import argparse

    parser = argparse.ArgumentParser(description="Run proactive outreach iteration")
    parser.add_argument("profiles_dir", type=Path, help="Directory with customer JSON profiles")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Optional file with proactive prompt seed (skip to use Planner)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="escalar",
        help="Strategy identifier",
    )
    parser.add_argument(
        "--tone",
        type=str,
        default="empático-directo",
        help="Desired tone for proactive agent",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=3,
        help="Maximum conversation turns",
    )
    parser.add_argument(
        "--run-number",
        type=int,
        default=1,
        help="Current learning iteration number",
    )
    parser.add_argument(
        "--strategy-attempt",
        type=int,
        default=1,
        help="Identifier for the strategy attempt",
    )
    parser.add_argument(
        "--message-attempt",
        type=int,
        default=1,
        help="Identifier for the message attempt/template variant",
    )
    args = parser.parse_args()

    prompt_seed = None
    if args.prompt_file:
        with args.prompt_file.open("r", encoding="utf-8") as f:
            prompt_seed = f.read().strip()
    planner = PlannerAgent()
    profiles = load_profiles(args.profiles_dir)

    objectives = [
        "Reducir churn percibido",
        "Establecer siguiente acción concreta",
        "Transmitir empatía sin prometer imposibles",
    ]

    run_iteration(
        profiles,
        proactive_prompt=prompt_seed,
        objectives=objectives if prompt_seed else None,
        strategy_id=args.strategy if prompt_seed else None,
        tone=args.tone,
        max_turns=args.max_turns,
        judge_model="gpt-4.1-mini",
        run_number=args.run_number,
        strategy_attempt_id=args.strategy_attempt,
        message_attempt_id=args.message_attempt,
        planner=planner,
    )


def _resolve_plan(
    *,
    profile: Dict,
    planner: PlannerAgent,
    proactive_prompt: Optional[str],
    objectives: Optional[List[str]],
    strategy_id: Optional[str],
    tone: str,
    max_turns: int,
    end_triggers: Optional[List[str]],
    cohort_summary: Optional[Dict],
    history_notes: Optional[str],
) -> StrategyPlan:
    if proactive_prompt is not None and strategy_id is not None:
        return StrategyPlan(
            prompt_seed=proactive_prompt,
            objectives=objectives or ["Mantener comunicación", "Solicitar respuesta"],
            tone=tone,
            strategy_id=strategy_id,
            max_turns=max_turns,
            end_triggers=end_triggers or ["END", "[END]", "<<END>>"],
        )

    plan = planner.run(
        profile=profile,
        cohort_summary=cohort_summary,
        history_notes=history_notes,
    )

    if end_triggers:
        plan.end_triggers = end_triggers
    plan.max_turns = max_turns
    return plan


def _format_cohort_label(cohort: Dict) -> str:
    vocal = "vocal" if cohort.get("vocal", False) else "no_vocal"
    satisf = "satisfecho" if cohort.get("satisfied", True) else "insatisfecho"
    return f"{vocal}_{satisf}"
