"""
Helpers to translate profile JSON structures into shared models.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.HumanSimulacra.schemas import Persona

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
    persona = profile.get("persona")
    human = profile.get("human_simulacra")
    if human and not persona:
        persona = {
            "name": human.get("nombre"),
            "age": human.get("edad"),
            "location": human.get("ciudad"),
            "bio": human.get("historia_revelada"),
        }

    if persona is None:
        persona = {}

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
    if not mini_story and human:
        mini_story = human.get("historia_revelada", "")

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


def persona_to_profile(persona_payload: Dict[str, Any], *, customer_id: Optional[str] = None) -> Dict:
    """
    Convert a persona JSON (as generated in personas_output) into the profile
    structure expected by the context-engineering pipeline.
    """
    persona = Persona.model_validate(persona_payload)

    segment_map = {
        True: {True: "VF", False: "VE"},
        False: {True: "NVF", False: "NVE"},
    }
    satisfied_known = persona.satisfaccion == "Satisfecho" if persona.es_vocal else True
    segment = segment_map[persona.es_vocal][satisfied_known]

    cohort = {
        "vocal": persona.es_vocal,
        "satisfied": satisfied_known,
    }

    persona_section = {
        "name": persona.nombre,
        "age": persona.edad,
        "location": persona.ciudad,
        "bio": persona.historia_revelada,
    }

    price_anchor = float(persona.ltv)
    purchase = {
        "vehicle": persona.relacion_kavak,
        "price": max(10000.0, round(price_anchor, 2)),
        "last_purchase_days": 30 if persona.es_vocal else 90,
        "channel": "whatsapp",
    }

    past_nps = None
    tickets: List[Dict[str, Any]] = []
    for record in persona.historial_vocalidad:
        if record.nps is not None:
            past_nps = record.nps
        tickets.append(
            {
                "issue": record.resumen,
                "sentiment": (record.nps - 5) / 5 if record.nps is not None else 0.0,
                "last_touch_days": 7 if persona.es_vocal else 45,
            }
        )

    if past_nps is None:
        past_nps = 9 if satisfied_known else 4

    history_messages: List[Dict[str, Any]] = []
    if persona.es_vocal and persona.problema:
        history_messages.append({"role": "customer", "content": persona.problema})
    if persona.es_vocal and persona.expectativa_solucion:
        history_messages.append({"role": "customer", "content": persona.expectativa_solucion})

    history = {
        "past_nps": past_nps,
        "tickets": tickets,
        "messages": history_messages,
    }

    risk_signals = {
        "churn_est": 0.65 if not satisfied_known else 0.25,
        "value_segment": segment,
        "ltv_apriori": float(persona.ltv),
    }

    profile = {
        "customer_id": customer_id or persona_payload.get("customer_id") or persona.nombre,
        "cohort": cohort,
        "persona": persona_section,
        "purchase": purchase,
        "history": history,
        "risk_signals": risk_signals,
        "human_simulacra": persona.model_dump(mode="python"),
    }

    profile["cohort_label"] = infer_cohort_label(profile)

    return profile


def infer_cohort_label(profile: Dict) -> str:
    cohort = profile.get("cohort", {}) or {}
    vocal = "vocal" if cohort.get("vocal", False) else "no_vocal"
    satisfied = cohort.get("satisfied", True)
    satisf_label = "satisfecho" if satisfied else "insatisfecho"
    return f"{vocal}_{satisf_label}"


__all__ = ["profile_to_context", "persona_to_profile", "infer_cohort_label"]
