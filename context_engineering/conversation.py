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
from typing import Dict, List, Optional

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
        conversation_context = self._build_context(profile, plan)

        # Optional opening from customer if available
        if customer_agent.starting_message:
            turns.append(
                ConversationTurn(
                    role="customer",
                    content=customer_agent.starting_message,
                    metadata={"initial": True},
                )
            )

        for step in range(plan.max_turns):
            # Generate proactive message
            downstream_prompt = self._compose_proactive_prompt(plan, profile, turns, conversation_context)
            agent_message = self.proactive_runner.run_text(
                system_prompt=plan.prompt_seed,
                user_content=downstream_prompt,
            )
            agent_turn = ConversationTurn(role="agent", content=agent_message)
            turns.append(agent_turn)

            ended, outcome = self._check_outcome(agent_message, plan.end_triggers, ended_by="agent")
            if ended:
                agent_turn.metadata["outcome"] = outcome
                return ConversationResult(
                    customer_id=customer_agent.customer_id,
                    turns=turns,
                    objectives=plan.objectives,
                    strategy_id=plan.strategy_id,
                    outcome=outcome,
                    ended_by="agent",
                )

            # Customer response (unless final turn)
            if step == plan.max_turns - 1:
                break

            customer_reply = self._generate_customer_reply(customer_agent, turns)
            customer_turn = ConversationTurn(role="customer", content=customer_reply)
            turns.append(customer_turn)

            ended, outcome = self._check_outcome(customer_reply, plan.end_triggers, ended_by="customer")
            if ended:
                customer_turn.metadata["outcome"] = outcome
                return ConversationResult(
                    customer_id=customer_agent.customer_id,
                    turns=turns,
                    objectives=plan.objectives,
                    strategy_id=plan.strategy_id,
                    outcome=outcome,
                    ended_by="customer",
                )

        return ConversationResult(
            customer_id=customer_agent.customer_id,
            turns=turns,
            objectives=plan.objectives,
            strategy_id=plan.strategy_id,
            outcome="max_turns_reached",
            ended_by=None,
        )

    def _build_context(self, profile: Dict, plan: StrategyPlan) -> str:
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
        return "\n".join(lines)

    def _compose_proactive_prompt(
        self,
        plan: StrategyPlan,
        profile: Dict,
        turns: List[ConversationTurn],
        context_digest: str,
    ) -> str:
        """Compose user content for proactive agent including history."""
        history_text = self._format_history(turns)
        return (
            f"Contexto del cliente:\n{context_digest}\n\n"
            f"Conversación hasta ahora:\n{history_text}\n\n"
            "Redacta la siguiente intervención cumpliendo los objetivos y manteniendo el tono indicado."
        )

    def _generate_customer_reply(
        self,
        customer_agent: CustomerAgent,
        turns: List[ConversationTurn],
    ) -> str:
        """Generate customer reply using customer agent system prompt."""
        history_text = self._format_history(turns)
        prompt = (
            "Responde al último mensaje recibido manteniendo tu personalidad y emociones.\n\n"
            f"Historial:\n{history_text}"
        )
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

    def _format_history(self, turns: List[ConversationTurn]) -> str:
        """Format conversation history as plain text."""
        if not turns:
            return "(Sin mensajes previos)"

        lines = []
        for idx, turn in enumerate(turns, start=1):
            speaker = "Agente" if turn.role == "agent" else "Cliente"
            lines.append(f"{idx}. {speaker}: {turn.content}")
        return "\n".join(lines)


__all__ = [
    "ProactiveConversationAgent",
    "StrategyPlan",
    "ConversationResult",
    "ConversationTurn",
]
