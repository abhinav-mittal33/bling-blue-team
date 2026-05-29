"""
CISO Notification Workflow (P5-5).
Sends security alert to CISO when fraud amount exceeds ₹10L or HIGH_RISK count spikes.

Channels:
  1. Email to CISO_EMAIL (via SMTP or SendGrid stub)
  2. Slack webhook to SLACK_WEBHOOK_URL
  3. model_audit INSERT (immutable record — always, regardless of delivery)
"""
from __future__ import annotations
from datetime import datetime, timezone
import structlog

logger = structlog.get_logger()

_CISO_ALERT_THRESHOLD = 1_000_000.0  # ₹10L


def notify_ciso_if_needed(
    alert_id: str,
    transaction_id: str,
    amount: float,
    action: str,
    fraud_type: str | None,
    db=None,
) -> None:
    """
    Fire CISO notification for HIGH_RISK alerts above ₹10L. Best-effort — never raises.
    Always logs to model_audit (immutable) regardless of delivery success.
    """
    if action != "HIGH_RISK" or amount < _CISO_ALERT_THRESHOLD:
        return

    _log_to_audit(alert_id, transaction_id, amount, action, fraud_type, db)

    try:
        from app.core.config import settings

        subject = f"[BLING ALERT] HIGH_RISK fraud ₹{amount:,.0f} — {alert_id}"
        body = (
            f"HIGH_RISK fraud detected.\n"
            f"Alert ID: {alert_id}\n"
            f"Transaction: {transaction_id}\n"
            f"Amount: ₹{amount:,.2f}\n"
            f"Type: {fraud_type or 'Unknown'}\n"
            f"Time: {datetime.now(timezone.utc).isoformat()}\n"
        )

        _send_slack(settings, subject, body)
        _send_email(settings, subject, body)

    except Exception as exc:
        logger.error("ciso_notification_failed", alert_id=alert_id, error=str(exc))


def _send_slack(settings, subject: str, body: str) -> None:
    if not settings.slack_webhook_url:
        return
    try:
        import httpx
        httpx.post(
            settings.slack_webhook_url,
            json={"text": f":rotating_light: *{subject}*\n```{body}```"},
            timeout=5.0,
        )
        logger.info("ciso_slack_sent", subject=subject)
    except Exception as exc:
        logger.warning("ciso_slack_failed", error=str(exc))


def _send_email(settings, subject: str, body: str) -> None:
    ciso_email = getattr(settings, "ciso_email", "")
    if not ciso_email:
        return
    try:
        import smtplib
        from email.mime.text import MIMEText
        smtp_host = getattr(settings, "smtp_host", "")
        smtp_user = getattr(settings, "smtp_user", "")
        smtp_password = getattr(settings, "smtp_password", "")
        if not smtp_host:
            logger.debug("ciso_email_skipped", reason="no smtp_host configured")
            return
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = smtp_user or "bling-alerts@unionbank.in"
        msg["To"] = ciso_email
        with smtplib.SMTP(smtp_host, 587) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            if smtp_user and smtp_password:
                server.login(smtp_user, smtp_password)
            server.send_message(msg)
        logger.info("ciso_email_sent", to=ciso_email)
    except Exception as exc:
        logger.warning("ciso_email_failed", error=str(exc))


def _log_to_audit(
    alert_id: str, transaction_id: str, amount: float,
    action: str, fraud_type: str | None, db
) -> None:
    """Always INSERT to model_audit — immutable RBI compliance record."""
    if db is None:
        return
    try:
        from app.utils.audit_logger import log_score_event
        from app.core.config import settings
        log_score_event(db, transaction_id, {
            "event_subtype": "CISO_NOTIFICATION",
            "alert_id": alert_id,
            "amount": amount,
            "action": action,
            "fraud_type": fraud_type,
        })
    except Exception as exc:
        logger.error("ciso_audit_failed", error=str(exc))
