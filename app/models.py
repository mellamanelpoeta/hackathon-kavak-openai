"""
Data models and schemas for the Kavak customer service demo.
Defines Pydantic models for type safety and validation.
"""
from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field


class Customer(BaseModel):
    """Customer data model matching the CSV schema."""
    customer_id: str
    segment: Literal["VF", "VE", "NVF", "NVE"]
    is_vocal: bool
    last_purchase_days: int
    price: float
    issues_flag: int = Field(ge=0, le=1)
    past_NPS: int = Field(ge=0, le=10)
    first_message: Optional[str] = None
    channel_pref: Literal["whatsapp", "email", "phone", "sms"] = "whatsapp"
    churn_risk_est: float = Field(ge=0, le=1)
    issue_bucket: Optional[Literal["mecanica", "finanzas", "logistica", "atencion"]] = None
    mini_story: Optional[str] = None


class Context(BaseModel):
    """Context passed to agents for message generation."""
    customer_id: str
    segment: str
    is_vocal: bool
    last_purchase_days: int
    price: float
    issues_flag: int
    past_NPS: int
    channel_pref: str
    churn_risk_est: float
    mini_story: str
    first_message: Optional[str] = None
    issue_bucket: str


class Score(BaseModel):
    """Judge evaluation score with strict validation."""
    NPS_expected: float = Field(ge=0, le=10)
    EngagementProb: float = Field(ge=0, le=1)
    ChurnProb: float = Field(ge=0, le=1)
    AspectSentiment: Dict[str, float] = Field(
        default_factory=lambda: {
            "finanzas": 0.0,
            "mecanica": 0.0,
            "logistica": 0.0,
            "atencion": 0.0
        }
    )
    rationale: str = Field(max_length=280)


class Template(BaseModel):
    """Template definition for message generation."""
    id: str
    name: str
    slots: list[str]
    guardrails: list[str]
    template_text: str


class InteractionLog(BaseModel):
    """Log entry for each customer interaction."""
    customer_id: str
    segment: str
    issue_bucket: str
    arm: str  # template_id
    message: str
    score: Score
    reward: float
    iteration: int = 0
    interaction_type: Literal["vocal", "outreach"] = "vocal"
