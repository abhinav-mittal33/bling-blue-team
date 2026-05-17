from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ─── Request Schemas ────────────────────────────────────────────────────────

class TransactionScoreRequest(BaseModel):
    transaction_id: str = Field(..., min_length=1, max_length=36, pattern=r"^[a-zA-Z0-9_-]+$")
    account_id: str = Field(..., min_length=1, max_length=20)
    payee_account_id: Optional[str] = Field(None, max_length=20)
    amount: Decimal = Field(..., gt=0, lt=100_000_000)
    channel: Literal["UPI", "IMPS", "RTGS", "NEFT"]
    timestamp: datetime
    payee_vpa: Optional[str] = Field(None, max_length=100)
    payee_vpa_created_at: Optional[datetime] = None
    device_id: Optional[str] = Field(None, max_length=100)
    ip_address: Optional[str] = None
    geo_city: Optional[str] = Field(None, max_length=50)
    geo_state: Optional[str] = Field(None, max_length=50)
    merchant_terminal_id: Optional[str] = Field(None, max_length=50)

    @field_validator("timestamp")
    @classmethod
    def timestamp_not_future(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        if v > datetime.now(timezone.utc) + timedelta(minutes=5):
            raise ValueError("Transaction timestamp cannot be more than 5 minutes in the future")
        return v

    @field_validator("amount")
    @classmethod
    def amount_precision(cls, v: Decimal) -> Decimal:
        if v != round(v, 2):
            raise ValueError("Amount cannot have more than 2 decimal places")
        return v


class InvestigatorFeedbackRequest(BaseModel):
    alert_id: str = Field(..., min_length=1, max_length=36)
    investigator_id: str = Field(..., min_length=1, max_length=100)
    confirmed_fraud: bool
    fraud_type: Optional[str] = Field(None, max_length=100)
    str_submitted: bool = False
    notes: Optional[str] = Field(None, max_length=5000)


# ─── Response Schemas ────────────────────────────────────────────────────────

class ScoreResponse(BaseModel):
    transaction_id: str
    score: float
    action: Literal["PASS", "LOG", "REVIEW", "HIGH_RISK"]
    gate_fired: Optional[str] = None
    alert_id: Optional[str] = None
    processing_ms: int


class SHAPFeature(BaseModel):
    feature: str
    contribution: float


class SHAPExplanation(BaseModel):
    top_features: list[SHAPFeature]


class GraphNode(BaseModel):
    id: str
    account_type: Optional[str] = None
    is_ghost: bool = False


class GraphEdge(BaseModel):
    source: str
    target: str
    amount: float
    channel: str
    timestamp: str


class GraphEvidence(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    layout: str = "sankey"


class LayerBreakdown(BaseModel):
    tier1_flags: list[str]
    tier2_gate: Optional[str] = None
    tier3_score: Optional[float] = None


class IndianContextApplied(BaseModel):
    adjustments: dict[str, float] = {}
    final_adjustment_factor: float = 1.0


class STRDraft(BaseModel):
    reporting_entity: str = "Union Bank of India"
    report_type: str = "STR"
    transaction_date: Optional[str] = None
    transaction_amount: Optional[float] = None
    transaction_currency: str = "INR"
    transaction_channel: Optional[str] = None
    suspicious_activity_type: Optional[str] = None
    narrative: Optional[str] = None
    subject_account: Optional[str] = None
    detection_method: Optional[str] = None
    submission_status: str = "DRAFT"


class AlertResponse(BaseModel):
    alert_id: str
    transaction_id: str
    score: float
    gate: Optional[str] = None
    action: str
    status: str
    trail_status: str
    total_amount_in_trail: Optional[float] = None
    account_count: Optional[int] = None
    hop_count: Optional[int] = None
    trail_start: Optional[str] = None
    trail_end: Optional[str] = None
    ghost_nodes: list[str] = []
    graph_evidence: Optional[GraphEvidence] = None
    shap_explanation: Optional[SHAPExplanation] = None
    layer_breakdown: Optional[LayerBreakdown] = None
    indian_context_applied: Optional[IndianContextApplied] = None
    str_draft: Optional[STRDraft] = None
    created_at: str


class FeedbackResponse(BaseModel):
    alert_id: str
    model_updated: bool
    blockchain_sealed: bool
    red_team_notified: bool


# ─── Wrapper ──────────────────────────────────────────────────────────────────

class APIResponse(BaseModel):
    data: Optional[object] = None
    error: Optional[dict] = None
    meta: dict = {}
