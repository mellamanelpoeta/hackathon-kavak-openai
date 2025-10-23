"""
Example runner showing how to generate customer agents, execute conversations,
score outcomes, and compute expected LTV deltas.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.HumanSimulacra.schemas import Persona
from app.factories.judge import Judge

from .agents_factory import CustomerAgentFactory
from .conversation import ProactiveConversationAgent, StrategyPlan
from .ltv import evaluate_conversation
from .planner import PlannerAgent
from .profile_utils import persona_to_profile, profile_to_context
from .strategies import get_strategy


def load_profiles(path: Path) -> List[dict]:
    """Load all JSON profiles from a directory (supports personas_output structure)."""
    profiles: List[dict] = []
    json_paths = sorted(path.glob("**/*.json"))
    for json_file in json_paths:
        with json_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if _is_persona_payload(payload):
            persona_profile = persona_to_profile(payload, customer_id=json_file.stem)
            profiles.append(persona_profile)
            continue

        persona_payload = payload.get("human_simulacra")
        if persona_payload:
            persona_obj = Persona(**persona_payload)
            payload["human_simulacra"] = persona_obj.model_dump(mode="python")

        profiles.append(payload)
    return profiles
def _is_persona_payload(payload: dict) -> bool:
    required = {"nombre", "historia_revelada", "historia_oculta", "prompt_conversacional"}
    return required.issubset(payload.keys())


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
    concurrency: int = 10,
    planner_model: str = "gpt-4.1",
    proactive_model: str = "gpt-4.1",
    customer_model: str = "gpt-4.1-mini",
    verbose: bool = True,
    include_logs: bool = True,
) -> List[dict]:
    """
    High-level helper to:
      1. Build customer agents from profiles
      2. Run a conversation per profile using the provided strategy
      3. Evaluate results with Judge + LTV scoring
    Returns list of summaries per customer.
    """
    indexed_profiles = list(enumerate(list(profiles)))
    if not indexed_profiles:
        return []

    config: Dict[str, Optional[object]] = {
        "proactive_prompt": proactive_prompt,
        "objectives": objectives,
        "strategy_id": strategy_id,
        "tone": tone,
        "max_turns": max_turns,
        "end_triggers": end_triggers,
        "run_number": run_number,
        "strategy_attempt_id": strategy_attempt_id,
        "message_attempt_id": message_attempt_id,
        "cohort_summary": cohort_summary,
        "history_notes": history_notes,
        "planner_model": planner_model,
        "proactive_model": proactive_model,
        "customer_model": customer_model,
        "judge_model": judge_model,
        "api_key": api_key,
    }

    records: Dict[int, dict] = {}
    logs: Dict[int, List[str]] = {} if include_logs else {}

    if concurrency <= 1:
        config["planner_instance"] = planner or PlannerAgent(api_key=api_key, model=planner_model)
        config["factory_instance"] = CustomerAgentFactory()
        config["orchestrator_instance"] = ProactiveConversationAgent(
            api_key=api_key,
            proactive_model=proactive_model,
            customer_model=customer_model,
        )
        config["judge_instance"] = Judge(api_key=api_key, model=judge_model)

        for idx, profile in indexed_profiles:
            idx_out, record, log_lines = _process_profile(idx, profile, config, reuse_agents=True)
            if include_logs:
                logs[idx_out] = log_lines
            if record:
                records[idx_out] = record
    else:
        config["planner_instance"] = None
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_process_profile, idx, profile, config, False): idx
                for idx, profile in indexed_profiles
            }
            for future in as_completed(futures):
                idx_out, record, log_lines = future.result()
                if include_logs:
                    logs[idx_out] = log_lines
                if record:
                    records[idx_out] = record

    summaries: List[dict] = []
    for idx, _ in indexed_profiles:
        if verbose and include_logs:
            for line in logs.get(idx, []):
                print(line)
        if idx in records:
            summaries.append(records[idx])

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
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of conversations to run in parallel",
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
        concurrency=args.concurrency,
        api_key=os.getenv("OPENAI_API_KEY"),
        include_logs=not args.quiet,
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


def _process_profile(
    idx: int,
    profile: Dict,
    config: Dict[str, Optional[object]],
    reuse_agents: bool,
) -> Tuple[int, Optional[dict], List[str]]:
    log_lines: List[str] = []
    try:
        planner: PlannerAgent
        if reuse_agents:
            planner = config["planner_instance"]  # type: ignore[assignment]
        else:
            planner = PlannerAgent(
                api_key=config.get("api_key"),
                model=config.get("planner_model", "gpt-4.1"),  # type: ignore[arg-type]
            )

        plan = _resolve_plan(
            profile=profile,
            planner=planner,
            proactive_prompt=config.get("proactive_prompt"),
            objectives=config.get("objectives"),  # type: ignore[arg-type]
            strategy_id=config.get("strategy_id"),  # type: ignore[arg-type]
            tone=config.get("tone", "empático-directo"),  # type: ignore[arg-type]
            max_turns=config.get("max_turns", 3),  # type: ignore[arg-type]
            end_triggers=config.get("end_triggers"),  # type: ignore[arg-type]
            cohort_summary=config.get("cohort_summary"),  # type: ignore[arg-type]
            history_notes=config.get("history_notes"),  # type: ignore[arg-type]
        )

        if reuse_agents:
            factory: CustomerAgentFactory = config["factory_instance"]  # type: ignore[assignment]
            orchestrator: ProactiveConversationAgent = config["orchestrator_instance"]  # type: ignore[assignment]
            judge: Judge = config["judge_instance"]  # type: ignore[assignment]
        else:
            factory = CustomerAgentFactory()
            orchestrator = ProactiveConversationAgent(
                api_key=config.get("api_key"),
                proactive_model=config.get("proactive_model", "gpt-4.1"),  # type: ignore[arg-type]
                customer_model=config.get("customer_model", "gpt-4.1-mini"),  # type: ignore[arg-type]
            )
            judge = Judge(
                api_key=config.get("api_key"),
                model=config.get("judge_model", "gpt-4.1-mini"),  # type: ignore[arg-type]
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

        if result.nps_score is not None:
            score = score.model_copy(update={"NPS_expected": float(result.nps_score)})

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

        transcript_records = [
            {
                "role": turn.role,
                "content": turn.content,
                "metadata": turn.metadata,
            }
            for turn in result.turns
        ]
        if result.initial_customer_message:
            transcript_records.insert(
                0,
                {
                    "role": "context",
                    "content": result.initial_customer_message,
                    "metadata": {"type": "initial_expectation"},
                },
            )

        record = {
            "client_id": result.customer_id,
            "nps_og": nps_original,
            "vocal": bool(cohort.get("vocal", False)),
            "satisfecho": bool(cohort.get("satisfied", True)),
            "cohort_label": _format_cohort_label(cohort),
            "run_number": config.get("run_number", 1),
            "estrategia_intentada": config.get("strategy_attempt_id", 1),
            "mensaje_intentado": config.get("message_attempt_id", 1),
            "NPS_final": int(round(score.NPS_expected)),
            "NPS_comment": result.nps_comment,
            "initial_customer_message": result.initial_customer_message,
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
            "nps_score_reported": result.nps_score,
            "transcript": transcript_records,
        }

        log_lines.append("=" * 60)
        log_lines.append(
            f"Customer: {result.customer_id} | Strategy: {result.strategy_id} | Outcome: {result.outcome}"
        )
        if result.initial_customer_message:
            log_lines.append(f"[Contexto] {result.initial_customer_message}")

        for turn in result.turns:
            speaker = "Agente" if turn.role == "agent" else "Cliente"
            log_lines.append(f"[{speaker}] {turn.content}")
        nps_line = (
            f"→ NPS auto-reportado: {result.nps_score:.2f}"
            if result.nps_score is not None
            else "→ NPS auto-reportado: N/D"
        )
        log_lines.append(
            f"{nps_line} | Engagement: {score.EngagementProb:.2%} | Churn: {score.ChurnProb:.2%}"
        )
        log_lines.append(
            f"→ LTV esperado: ${metrics['ltv_expected']:.2f} "
            f"(Δ: {metrics['ltv_expected'] - metrics['ltv_apriori']:.2f}) | Reward: {metrics['reward']:.3f}"
        )
        log_lines.append(f"→ Strategy rationale: {strategy_def.razonamiento_estrategia}")
        log_lines.append(f"→ Judge rationale: {score.rationale}")
        log_lines.append(f"→ Resultado heurístico: {record['resultado']} | Ganancia LTV: {ganancia_ltv:.2f}")
        log_lines.append("")

        return idx, record, log_lines
    except Exception as exc:  # keep batch running despite failures
        log_lines.append("=" * 60)
        log_lines.append(f"[ERROR] Cliente {profile.get('customer_id', idx)} -> {exc}")
        log_lines.append("")
        return idx, None, log_lines
