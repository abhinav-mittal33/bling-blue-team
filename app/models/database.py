from datetime import datetime, timezone
from decimal import Decimal
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Numeric, Text,
    TIMESTAMP, BigInteger, Index, ForeignKey, JSON
)
from sqlalchemy.dialects.postgresql import JSONB, INET
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Account(Base):
    __tablename__ = "accounts"

    id = Column(String(20), primary_key=True)
    account_type = Column(String(20), nullable=False)  # SAVINGS, CURRENT, JAN_DHAN, INTERNAL, TREASURY, NOSTRO, VOSTRO
    kyc_occupation = Column(String(100))
    kyc_declared_income_monthly = Column(Numeric(15, 2))
    kyc_age = Column(Integer)
    kyc_home_state = Column(String(50))
    kyc_completeness_score = Column(Float, default=0.0)
    account_age_days = Column(Integer)
    is_merchant = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False)


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(String(36), primary_key=True)
    account_id = Column(String(20), ForeignKey("accounts.id"), nullable=False)
    payee_account_id = Column(String(20))
    amount = Column(Numeric(15, 2), nullable=False)
    channel = Column(String(10), nullable=False)  # UPI, IMPS, RTGS, NEFT
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
    payee_vpa = Column(String(100))
    payee_vpa_created_at = Column(TIMESTAMP(timezone=True))
    merchant_terminal_id = Column(String(50))
    merchant_mcc = Column(String(10))
    device_id = Column(String(100))
    ip_address = Column(INET)
    geo_city = Column(String(50))
    geo_state = Column(String(50))
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_transactions_account_timestamp", "account_id", "timestamp"),
        Index("idx_transactions_payee", "payee_account_id"),
        Index("idx_transactions_timestamp", "timestamp"),
    )


class FraudScore(Base):
    __tablename__ = "fraud_scores"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    transaction_id = Column(String(36), ForeignKey("transactions.id"), nullable=False)
    score = Column(Float, nullable=False)
    gate_fired = Column(String(50))
    action = Column(String(20), nullable=False)  # PASS, LOG, REVIEW, HIGH_RISK
    tier1_flags = Column(JSONB)
    tier2_gate = Column(String(50))
    tier3_score = Column(Float)
    feature_vector = Column(JSONB)
    shap_values = Column(JSONB)
    indian_context_applied = Column(JSONB)
    model_version = Column(String(20))
    processing_ms = Column(Integer)
    scored_at = Column(TIMESTAMP(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(String(36), primary_key=True)
    transaction_id = Column(String(36), ForeignKey("transactions.id"), nullable=False)
    score = Column(Float, nullable=False)
    gate = Column(String(50))
    action = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False, default="OPEN")  # OPEN, UNDER_REVIEW, CONFIRMED_FRAUD, FALSE_POSITIVE
    trail_status = Column(String(20), nullable=False, default="PENDING")  # PENDING, PROCESSING, COMPLETE, FAILED
    evidence_package = Column(JSONB)
    fraud_type = Column(String(100))
    investigator_id = Column(String(100))
    investigator_decision = Column(Boolean)  # NULL=pending, True=fraud, False=FP
    investigator_notes = Column(Text)
    blockchain_sealed = Column(Boolean, default=False)
    red_team_notified = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class FeedbackLog(Base):
    __tablename__ = "feedback_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    alert_id = Column(String(36), ForeignKey("alerts.id"), nullable=False)
    transaction_id = Column(String(36), nullable=False)
    label = Column(Integer, nullable=False)  # 1=fraud, 0=false positive
    investigator_id = Column(String(100))
    model_version_at_feedback = Column(String(20))
    features_at_scoring = Column(JSONB)
    feedback_timestamp = Column(TIMESTAMP(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class GraphFeaturesCache(Base):
    __tablename__ = "graph_features_cache"

    account_id = Column(String(20), ForeignKey("accounts.id"), primary_key=True)
    degree_centrality = Column(Float)
    betweenness_centrality = Column(Float)
    clustering_coefficient = Column(Float)
    pagerank_fraud_seeded = Column(Float)
    community_id = Column(Integer)
    community_fraud_ratio = Column(Float)
    shortest_path_to_fraud = Column(Integer)
    cycle_membership = Column(Boolean, default=False)
    sink_score = Column(Float)
    bipartite_score = Column(Float)
    fan_out_ratio = Column(Float)
    computed_at = Column(TIMESTAMP(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class ModelAudit(Base):
    """
    IMMUTABLE — INSERT ONLY. DB-level rules block UPDATE and DELETE.
    RBI PMLA Section 12 compliance — tamper-proof audit trail.
    """
    __tablename__ = "model_audit"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    event_type = Column(String(50), nullable=False)  # SCORE, FEEDBACK, RETRAIN, THRESHOLD_CHANGE
    transaction_id = Column(String(36))
    model_version = Column(String(20), nullable=False)
    event_data = Column(JSONB, nullable=False)
    event_timestamp = Column(TIMESTAMP(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
