"""
Utility functions to compute expected LTV deltas after a conversation.

Formula:
    LTV = LTV_apriori + (0.7 * NPS_expected + 0.3 * EngagementProb) - Cost_estrategia
"""
from __future__ import annotations

from typing import Dict, Union

try:  # Optional import; avoid circular deps in type checking
    from app.models import Score  # type: ignore
except Exception:  # pragma: no cover
    Score = None  # type: ignore


def compute_expected_ltv(
    *,
    ltv_apriori: float,
    nps_expected: float,
    engagement_prob: float,
    strategy_cost: float,
) -> float:
    """Compute expected LTV after applying a strategy to a customer."""
    strategy_value = 0.7 * float(nps_expected) + 0.3 * float(engagement_prob)
    return float(ltv_apriori) + strategy_value - float(strategy_cost)


def _score_to_dict(score: Union[Dict, "Score"]) -> Dict:
    if hasattr(score, "model_dump"):
        return score.model_dump()
    return score  # type: ignore[arg-type]


def evaluate_conversation(
    *,
    profile: Dict,
    strategy,
    score: Union[Dict, "Score"],
) -> Dict:
    """
    Combine judge score with strategy metadata to produce final KPIs.

    Args:
        profile: customer profile dict
        strategy: StrategyDefinition (with atributos nombre_estrategia, razonamiento, costo, etc.)
        score: Judge output (Score pydantic model or dict)

    Returns:
        dict with NPS, engagement, churn, ltv_expected, reward proxy
    """
    score_dict = _score_to_dict(score)

    nps = float(score_dict.get("NPS_expected", 0.0))
    engagement = float(score_dict.get("EngagementProb", 0.0))
    churn = float(score_dict.get("ChurnProb", 0.0))

    risk = profile.get("risk_signals", {}) or {}
    ltv_apriori = float(
        risk.get("ltv_apriori", getattr(strategy, "ltv_base", 15000.0))
    )

    expected_ltv = compute_expected_ltv(
        ltv_apriori=ltv_apriori,
        nps_expected=nps,
        engagement_prob=engagement,
        strategy_cost=getattr(strategy, "costo", getattr(strategy, "cost", 0.0)),
    )

    reward_proxy = max(
        0.0,
        min(1.0, (0.6 * nps + 3.0 * engagement - 3.0 * churn) / 10.0),
    )

    return {
        "customer_id": profile.get("customer_id"),
        "strategy_id": getattr(strategy, "nombre_estrategia", "unknown"),
        "ltv_expected": expected_ltv,
        "ltv_apriori": ltv_apriori,
        "nps_expected": nps,
        "engagement": engagement,
        "churn": churn,
        "reward": reward_proxy,
        "strategy_cost": getattr(strategy, "costo", getattr(strategy, "cost", 0.0)),
        "strategy_description": getattr(strategy, "razonamiento_estrategia", ""),
        "accion_puntual": getattr(strategy, "accion_puntual", ""),
    }


__all__ = ["compute_expected_ltv", "evaluate_conversation"]
