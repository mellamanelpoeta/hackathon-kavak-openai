"""
Helpers to translate profile JSON structures into shared models.
"""
from __future__ import annotations

from typing import Dict, Optional

from app.models import Context


ISSUE_KEYWORDS = {
    "finanzas": ["finanzas", "pago", "crédito", "tarifa", "factura"],
    "mecanica": ["motor", "mecánico", "freno", "taller", "garantía", "ruido"],
    "logistica": ["entrega", "logística", "documentación", "papeles", "envío"],
    "atencion": ["servicio", "respuesta", "llamada", "atención", "soporte"],
}


def _infer_issue_bucket(profile: Dict) -> str:
    history = profile.get("history", {})
    tickets = history.get("tickets", [])
    messages = history.get("messages", [])
    text_sources = []

    for ticket in tickets:
        text_sources.append(ticket.get("issue", ""))
        text_sources.append(ticket.get("notes", ""))

    for msg in messages:
        if msg.get("role") == "customer":
            text_sources.append(msg.get("content", ""))

    combined = " ".join(text_sources).lower()
    for bucket, keywords in ISSUE_KEYWORDS.items():
        if any(keyword in combined for keyword in keywords):
            return bucket

    return "atencion"


def _extract_first_message(profile: Dict) -> Optional[str]:
    messages = profile.get("history", {}).get("messages", [])
    for msg in messages:
        if msg.get("role") == "customer":
            content = msg.get("content")
            if content:
                return content
    return None


def profile_to_context(profile: Dict) -> Context:
    persona = profile.get("persona", {})
    purchase = profile.get("purchase", {})
    history = profile.get("history", {})
    risk = profile.get("risk_signals", {})
    cohort = profile.get("cohort", {})

    segment = risk.get("value_segment", "VE")
    if segment not in {"VF", "VE", "NVF", "NVE"}:
        segment = "VE"

    is_vocal = bool(cohort.get("vocal", False))
    satisfied = bool(cohort.get("satisfied", True))
    last_purchase_days = int(purchase.get("last_purchase_days", 30))
    price = float(purchase.get("price", 200000))
    issues_flag = 0 if satisfied else 1

    past_nps = history.get("past_nps")
    if past_nps is None:
        past_nps = 9 if satisfied else 4

    first_message = _extract_first_message(profile)
    mini_story = persona.get("bio") or ""

    churn_risk = float(risk.get("churn_est", 0.2))
    issue_bucket = _infer_issue_bucket(profile)
    channel_pref = purchase.get("channel", "whatsapp") or "whatsapp"

    return Context(
        customer_id=profile.get("customer_id", "unknown"),
        segment=segment,
        is_vocal=is_vocal,
        last_purchase_days=last_purchase_days,
        price=price,
        issues_flag=issues_flag,
        past_NPS=int(past_nps),
        first_message=first_message,
        channel_pref=channel_pref,
        churn_risk_est=churn_risk,
        mini_story=mini_story,
        issue_bucket=issue_bucket,
    )


__all__ = ["profile_to_context"]
