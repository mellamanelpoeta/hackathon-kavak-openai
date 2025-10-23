"""
Conversation orchestrator between proactive agent and generated customer agents.

This module handles 1:1 interactions:
- Receives a `CustomerAgent` (persona-specific system prompt)
- Uses a strategy prompt to drive the outreach agent
- Alternates turns (proactive -> customer) up to `max_turns`
- Returns full transcript and per-turn metadata
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Tuple

from app.factories.agents_runner import AgentsRunner

from .agents_factory import CustomerAgent


@dataclass
class StrategyPlan:
    """Structured plan produced by a planner agent."""

    prompt_seed: str
    objectives: List[str]
    tone: str
    strategy_id: str
    max_turns: int = 3
    end_triggers: List[str] = field(default_factory=lambda: ["END", "[END]", "<<END>>"])


@dataclass
class ConversationTurn:
    role: str
    content: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class ConversationResult:
    customer_id: str
    turns: List[ConversationTurn]
    objectives: List[str]
    strategy_id: str
    outcome: str
    ended_by: Optional[str]
    nps_score: Optional[float] = None
    nps_comment: Optional[str] = None
    initial_customer_message: Optional[str] = None


class ProactiveConversationAgent:
    """
    Orchestrates the proactive agent interacting with customer agents.
    The proactive agent is powered by an LLM prompt (strategy plan).
    """

    def __init__(
        self,
        *,
        proactive_model: str = "gpt-4.1",
        customer_model: str = "gpt-4.1-mini",
        api_key: Optional[str] = None,
    ):
        self.proactive_runner = AgentsRunner(
            api_key=api_key,
            model=proactive_model,
            temperature=0.3,
            max_output_tokens=400,
        )
        self.customer_runner = AgentsRunner(
            api_key=api_key,
            model=customer_model,
            temperature=0.4,
            max_output_tokens=200,
        )

    def run_conversation(
        self,
        *,
        customer_agent: CustomerAgent,
        plan: StrategyPlan,
        profile: Dict,
    ) -> ConversationResult:
        """
        Execute a multi-turn conversation between proactive agent and customer agent.

        Args:
            customer_agent: persona-specific configuration
            plan: strategy containing prompt seed and objectives
            profile: raw customer profile (for context conditioning)

        Returns:
            ConversationResult with alternating turns
        """
        turns: List[ConversationTurn] = []
        initial_context = customer_agent.profile.get("_initial_customer_message")
        conversation_context = self._build_context(profile, plan, initial_context)
        final_outcome = "max_turns_reached"
        final_ended_by: Optional[str] = None

        for step in range(plan.max_turns):
            # Generate proactive message
            downstream_prompt = self._compose_proactive_prompt(
                plan,
                turns,
                conversation_context,
            )
            agent_message = self.proactive_runner.run_text(
                system_prompt=plan.prompt_seed,
                user_content=downstream_prompt,
            )
            agent_turn = ConversationTurn(role="agent", content=agent_message)
            turns.append(agent_turn)

            ended, outcome = self._check_outcome(agent_message, plan.end_triggers, ended_by="agent")
            if ended:
                agent_turn.metadata["outcome"] = outcome
                final_outcome = outcome
                final_ended_by = "agent"
                break

            # Customer response (unless final turn)
            if step == plan.max_turns - 1:
                break

            customer_reply = self._generate_customer_reply(
                customer_agent,
                turns,
            )
            customer_turn = ConversationTurn(role="customer", content=customer_reply)
            turns.append(customer_turn)

            ended, outcome = self._check_outcome(customer_reply, plan.end_triggers, ended_by="customer")
            if ended:
                customer_turn.metadata["outcome"] = outcome
                final_outcome = outcome
                final_ended_by = "customer"
                break

        if not self._has_nps_response(turns):
            self._ensure_nps_exchange(customer_agent, turns)

        nps_score, nps_comment = self._extract_nps(turns)

        return ConversationResult(
            customer_id=customer_agent.customer_id,
            turns=turns,
            objectives=plan.objectives,
            strategy_id=plan.strategy_id,
            outcome=final_outcome,
            ended_by=final_ended_by,
            nps_score=nps_score,
            nps_comment=nps_comment,
            initial_customer_message=initial_context,
        )

    def _build_context(
        self,
        profile: Dict,
        plan: StrategyPlan,
        initial_context: Optional[str],
    ) -> str:
        """Create context digest for the proactive agent."""
        persona = profile.get("persona", {})
        purchase = profile.get("purchase", {})
        history = profile.get("history", {})
        risk = profile.get("risk_signals", {})
        cohort = profile.get("cohort")

        lines = [
            f"Cohorte: {cohort}",
            f"Persona: {persona}",
            f"Compra: {purchase}",
            f"Historial: {history}",
            f"Riesgos: {risk}",
            f"Estrategia seleccionada: {plan.strategy_id} con tono {plan.tone}",
            f"Objetivos: {', '.join(plan.objectives)}",
        ]
        if initial_context:
            lines.append(f"Expectativa reciente del cliente: {initial_context}")
        return "\n".join(lines)

    def _compose_proactive_prompt(
        self,
        plan: StrategyPlan,
        turns: List[ConversationTurn],
        context_digest: str,
    ) -> str:
        """Compose user content for proactive agent including history."""
        history_text = self._format_history(turns)
        extra_instruction = ""
        if not self._has_nps_request(turns):
            extra_instruction = (
                "\n\nRecuerda pedir explícitamente la calificación NPS del cliente (0-10) antes de cerrar."
            )

        return (
            f"Contexto del cliente:\n{context_digest}\n\n"
            f"Conversación hasta ahora:\n{history_text}\n\n"
            "Redacta la siguiente intervención cumpliendo los objetivos y manteniendo el tono indicado. "
            "No repitas ni cites literalmente el contexto; usa la información de forma natural y breve."
            f"{extra_instruction}"
        )

    def _generate_customer_reply(
        self,
        customer_agent: CustomerAgent,
        turns: List[ConversationTurn],
        *,
        force_nps: bool = False,
    ) -> str:
        """Generate customer reply using customer agent system prompt."""
        history_text = self._format_history(turns)
        persona = customer_agent.profile.get("human_simulacra", {})
        persona_name = persona.get("nombre", "Cliente")
        prompt_lines = [
            f"Eres {persona_name}. Responde al último mensaje como cliente de Kavak.",
            "Habla en primera persona, expresa emociones auténticas y necesidades reales.",
            "No ofrezcas soluciones ni comprometas acciones en nombre de Kavak.",
        ]

        if force_nps or self._last_agent_requested_nps(turns):
            prompt_lines.append(
                "El agente pidió tu calificación NPS. Responde con la forma exacta 'NPS: <número 0-10>' seguida de un comentario breve."
            )

        prompt_lines.append("")
        prompt_lines.append(f"Historial:\n{history_text}")
        prompt = "\n".join(prompt_lines)
        return self.customer_runner.run_text(
            system_prompt=customer_agent.system_prompt,
            user_content=prompt,
        )

    def _check_outcome(
        self,
        message: str,
        triggers: List[str],
        *,
        ended_by: str,
    ) -> tuple[bool, str]:
        """
        Determine if message signals the end of conversation.
        Returns (True, outcome) when any trigger is present.
        """
        normalized = message.strip().upper()
        for trigger in triggers:
            if trigger.upper() in normalized:
                outcome = f"{ended_by}_signaled_end:{trigger}"
                return True, outcome
        return False, ""

    def _has_nps_request(self, turns: List[ConversationTurn]) -> bool:
        return any(
            turn.role == "agent" and re.search(r"\bNPS\b", turn.content, re.IGNORECASE)
            for turn in turns
        )

    def _last_agent_requested_nps(self, turns: List[ConversationTurn]) -> bool:
        for turn in reversed(turns):
            if turn.role == "agent":
                return bool(re.search(r"\bNPS\b", turn.content, re.IGNORECASE))
            if turn.role == "customer":
                break
        return False

    def _has_nps_response(self, turns: List[ConversationTurn]) -> bool:
        for turn in turns:
            if turn.role != "customer":
                continue
            if re.search(r"\bNPS\s*[:=]\s*(\d{1,2})", turn.content, re.IGNORECASE):
                return True
        return False

    def _ensure_nps_exchange(
        self,
        customer_agent: CustomerAgent,
        turns: List[ConversationTurn],
    ) -> None:
        agent_message = (
            "Antes de cerrar, ¿podrías compartirme tu calificación NPS de 0 a 10 para esta experiencia"
            " y un breve comentario?"
        )
        agent_turn = ConversationTurn(
            role="agent",
            content=agent_message,
            metadata={"auto": "nps_request"},
        )
        turns.append(agent_turn)

        customer_reply = self._generate_customer_reply(
            customer_agent,
            turns,
            force_nps=True,
        )
        customer_turn = ConversationTurn(
            role="customer",
            content=customer_reply,
            metadata={"auto": "nps_response"},
        )
        turns.append(customer_turn)

    def _extract_nps(self, turns: List[ConversationTurn]) -> Tuple[Optional[float], Optional[str]]:
        for turn in reversed(turns):
            if turn.role != "customer":
                continue
            match = re.search(r"\bNPS\s*[:=]\s*(\d{1,2})", turn.content, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                value = max(0.0, min(10.0, value))
                comment = turn.content[match.end():].strip()
                return value, comment if comment else None
        return None, None

    def _format_history(self, turns: List[ConversationTurn]) -> str:
        """Format conversation history as plain text."""
        lines = []
        for idx, turn in enumerate(turns, start=1):
            speaker = "Agente" if turn.role == "agent" else "Cliente"
            lines.append(f"{idx}. {speaker}: {turn.content}")
        if not lines:
            return "(Sin mensajes previos)"
        return "\n".join(lines)


__all__ = [
    "ProactiveConversationAgent",
    "StrategyPlan",
    "ConversationResult",
    "ConversationTurn",
]
