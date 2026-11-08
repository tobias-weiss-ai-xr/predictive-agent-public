"""Email and webhook notification system for predictive-agent.

Sends alerts via SMTP email and HTTP webhook when remediation actions are taken
or when risk thresholds are exceeded.

Email: uses smtplib with STARTTLS to smtp.example.com:25 (HRZ relay, no
  auth — port 587 submission would require SMTP credentials the operator
  does not have)
  - Default recipient/From: alerts@example.com
    (From must be a resolvable uni address: the relay rejects other
    domains with 550 5.1.8 "Sender address rejected: Domain not found")
  - Configurable via ALERT_EMAIL_TO, ALERT_EMAIL_FROM, SMTP_HOST, SMTP_PORT,
    SMTP_USER, SMTP_PASSWORD

Webhook: uses urllib.request to POST JSON to WEBHOOK_URL
  - Configurable via WEBHOOK_URL
  - No-op if WEBHOOK_URL is empty

Alertmanager-compatible payloads for incident-bridge:
  build_alertmanager_payload() / WebhookNotifier.send_alertmanager() /
  NotificationManager.notify_alertmanager() emit incidents in the standard
  Alertmanager v4 webhook schema (an ``alerts`` list whose entries carry
  ``labels``, ``annotations``, and ``startsAt``) so they can be consumed by
  incident-bridge or any Alertmanager-compatible receiver. Unreachable
  endpoints never raise — they log a warning and return False.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from predictive_agent.config import (
    ALERT_EMAIL_FROM,
    ALERT_EMAIL_TO,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USE_TLS,
    SMTP_USER,
    WEBHOOK_TIMEOUT,
    WEBHOOK_URL,
)

# Note: these are imported at module load time. create_notifier_from_config()
# reads os.environ directly for testability.

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _rfc3339_now() -> str:
    """Return current UTC time as an RFC 3339 string with Z suffix.

    Alertmanager webhook timestamps use RFC 3339 (e.g. ``2026-08-29T10:00:00Z``)
    rather than the ``+00:00`` ISO 8601 form returned by _utc_now().
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _severity_for_risk(risk_score: float) -> str:
    """Map a risk score (0-1) to an Alertmanager label severity.

    Thresholds: >= 0.9 critical, >= 0.7 warning, otherwise info.
    """
    if risk_score >= 0.9:
        return "critical"
    if risk_score >= 0.7:
        return "warning"
    return "info"


def build_alertmanager_payload(
    alert_type: str,
    pod_name: str,
    risk_score: float,
    action_taken: str = "",
    details: str = "",
    status: str = "firing",
    starts_at: Optional[str] = None,
    severity: Optional[str] = None,
    receiver: str = "incident-bridge",
    external_url: str = "",
    extra_labels: Optional[dict] = None,
) -> dict:
    """Build an Alertmanager v4-compatible webhook payload for incident-bridge.

    The payload follows the Alertmanager webhook JSON schema so that any
    Alertmanager-compatible receiver (notably incident-bridge) can ingest it:
    a top-level ``alerts`` list whose entries carry ``labels``, ``annotations``,
    and ``startsAt``.

    Alert labels are derived from real incident data only: the alert name
    (alert_type), the namespace/pod parsed from pod_name (the ``ns/name``
    convention used throughout the agent), an action tag when a remediation
    action was taken, and a severity mapped from risk_score.

    Args:
        alert_type: Alert name (e.g. "high_risk", "remediation", "prediction").
        pod_name: Pod key, ``namespace/name`` (or bare name).
        risk_score: Risk score (0-100) from the predictor.
        action_taken: Remediation action taken, if any.
        details: Free-text or structured incident details.
        status: Alertmanager status: "firing" or "resolved".
        starts_at: Optional RFC 3339 start time; defaults to now.
        severity: Override the risk-derived severity label.
        receiver: Alertmanager receiver name (default "incident-bridge").
        external_url: Generator external URL (e.g. agent /status endpoint).
        extra_labels: Additional string labels to merge into the alert.

    Returns:
        A plain dict serializable with json.dumps.
    """
    namespace, _, name = pod_name.partition("/")
    if not name:
        name = namespace
        namespace = ""

    labels = {
        "alertname": alert_type,
        "severity": severity or _severity_for_risk(risk_score),
    }
    if namespace:
        labels["namespace"] = namespace
    labels["pod"] = name
    if action_taken:
        labels["action"] = action_taken
    if extra_labels:
        for k, v in extra_labels.items():
            labels[k] = str(v)

    annotations = {
        "summary": f"{alert_type}: {pod_name}",
        "description": details or f"Risk score {risk_score:.1f}",
        "risk_score": f"{risk_score:.1f}",
    }
    if action_taken:
        annotations["action"] = action_taken

    start = starts_at or _rfc3339_now()
    # Alertmanager uses the zero time for firing alerts (no end time yet).
    end = "0001-01-01T00:00:00Z" if status == "firing" else start

    group_key = f'{{}}:{{alertname="{alert_type}", pod="{name}"}}'

    return {
        "version": "4",
        "groupKey": group_key,
        "truncatedAlerts": 0,
        "status": status,
        "receiver": receiver,
        "groupLabels": {},
        "commonLabels": dict(labels),
        "commonAnnotations": dict(annotations),
        "externalURL": external_url,
        "alerts": [
            {
                "status": status,
                "labels": labels,
                "annotations": annotations,
                "startsAt": start,
                "endsAt": end,
                "generatorURL": external_url,
            }
        ],
    }


class EmailNotifier:
    """Send email alerts via SMTP with STARTTLS."""

    def __init__(
        self,
        host: str = SMTP_HOST,
        port: int = SMTP_PORT,
        user: str = SMTP_USER,
        password: str = SMTP_PASSWORD,
        from_addr: str = ALERT_EMAIL_FROM,
        to_addrs: Optional[list] = None,
        use_tls: bool = SMTP_USE_TLS,
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs or [ALERT_EMAIL_TO]
        self.use_tls = use_tls

    def send(self, subject: str, body: str) -> bool:
        """Send an email alert.

        Returns True on success, False on failure (never raises).
        """
        msg = MIMEText(body)
        msg["Subject"] = f"[SCS-Predictive-Agent] {subject}"
        msg["From"] = self.from_addr
        msg["To"] = ", ".join(self.to_addrs)

        try:
            with smtplib.SMTP(self.host, self.port, timeout=10) as server:
                if self.use_tls:
                    server.starttls(context=ssl.create_default_context())
                if self.user and self.password:
                    server.login(self.user, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            logger.info("Email alert sent: %s to %s", subject, self.to_addrs)
            return True
        except smtplib.SMTPException as e:
            logger.error("SMTP error sending email: %s", e)
            return False
        except Exception as e:
            logger.error("Error sending email: %s", e)
            return False


class WebhookNotifier:
    """Send JSON webhook notifications via HTTP POST."""

    def __init__(self, url: str = WEBHOOK_URL, timeout: int = WEBHOOK_TIMEOUT):
        self.url = url
        self.timeout = timeout

    def send(self, payload: dict) -> bool:
        """Send a webhook notification.

        Returns True on success (HTTP 200), False on failure.
        If url is empty, returns True (no-op, not an error).
        """
        if not self.url:
            return True

        try:
            data = json.dumps(payload).encode("utf-8")
            req = Request(
                self.url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=self.timeout) as response:
                return response.status == 200
        except URLError as e:
            logger.error("Webhook error: %s", e)
            return False
        except Exception as e:
            logger.error("Error sending webhook: %s", e)
            return False

    def send_alertmanager(
        self,
        alert_type: str,
        pod_name: str,
        risk_score: float,
        action_taken: str = "",
        details: str = "",
        **kwargs,
    ) -> bool:
        """Build an Alertmanager-compatible payload and POST it to the webhook.

        Same contract as send() — returns True on success (HTTP 200), False on
        failure, True on empty URL (no-op). Additional keyword arguments are
        forwarded to build_alertmanager_payload() (e.g. ``status``, ``starts_at``).
        """
        payload = build_alertmanager_payload(
            alert_type=alert_type,
            pod_name=pod_name,
            risk_score=risk_score,
            action_taken=action_taken,
            details=details,
            **kwargs,
        )
        return self.send(payload)


class NotificationManager:
    """Manage email and webhook notifications for remediation alerts."""

    def __init__(
        self,
        email_notifier: Optional[EmailNotifier] = None,
        webhook_notifier: Optional[WebhookNotifier] = None,
    ):
        self.email = email_notifier or EmailNotifier()
        self.webhook = webhook_notifier or WebhookNotifier()
        self.history: list[dict] = []

    def notify(
        self,
        alert_type: str,
        pod_name: str,
        risk_score: float,
        action_taken: str,
        details: str = "",
    ) -> tuple[bool, bool]:
        """Send notification via both email and webhook.

        Args:
            alert_type: Type of alert (e.g., "high_risk", "remediation", "prediction")
            pod_name: Name of the pod that triggered the alert
            risk_score: Risk score (0.0-1.0)
            action_taken: Remediation action taken (e.g., "pod_restart", "node_cordon")
            details: Additional details (JSON string or free text)

        Returns:
            (email_sent, webhook_sent) tuple of booleans
        """
        timestamp = _utc_now()

        # Email
        subject = f"{alert_type}: {pod_name}"
        body = (
            f"Predictive-Agent Alert\n"
            f"=======================\n\n"
            f"Timestamp: {timestamp}\n"
            f"Alert Type: {alert_type}\n"
            f"Pod: {pod_name}\n"
            f"Risk Score: {risk_score:.1f}\n"
            f"Action Taken: {action_taken}\n"
            f"Details: {details}\n"
        )
        email_sent = self.email.send(subject, body)

        # Webhook
        payload = {
            "alert_type": alert_type,
            "pod_name": pod_name,
            "risk_score": risk_score,
            "action_taken": action_taken,
            "details": details,
            "timestamp": timestamp,
        }
        webhook_sent = self.webhook.send(payload)

        # Record in history
        self.history.append(
            {
                "timestamp": timestamp,
                "alert_type": alert_type,
                "pod_name": pod_name,
                "risk_score": risk_score,
                "action_taken": action_taken,
                "email_sent": email_sent,
                "webhook_sent": webhook_sent,
            }
        )

        # Keep last 100 notifications
        if len(self.history) > 100:
            self.history = self.history[-100:]

        return email_sent, webhook_sent

    def notify_alertmanager(
        self,
        alert_type: str,
        pod_name: str,
        risk_score: float,
        action_taken: str = "",
        details: str = "",
        status: str = "firing",
        **kwargs,
    ) -> bool:
        """Emit an incident to incident-bridge in Alertmanager webhook format.

        Incident bridges are HTTP receivers, so this sends only via the
        webhook channel (no email). The payload is built with
        build_alertmanager_payload() and the outcome is recorded in history
        with ``format: "alertmanager"``. Returns True when the webhook
        accepted the payload (or none is configured); never raises.
        """
        timestamp = _utc_now()
        payload = build_alertmanager_payload(
            alert_type=alert_type,
            pod_name=pod_name,
            risk_score=risk_score,
            action_taken=action_taken,
            details=details,
            status=status,
            **kwargs,
        )
        webhook_sent = self.webhook.send(payload)

        self.history.append(
            {
                "timestamp": timestamp,
                "alert_type": alert_type,
                "pod_name": pod_name,
                "risk_score": risk_score,
                "action_taken": action_taken,
                "status": status,
                "format": "alertmanager",
                "email_sent": False,
                "webhook_sent": webhook_sent,
            }
        )

        # Keep last 100 notifications
        if len(self.history) > 100:
            self.history = self.history[-100:]

        return webhook_sent

    def get_history(self, limit: int = 20) -> list[dict]:
        """Return recent notification history."""
        return self.history[-limit:]


def create_notifier_from_config() -> NotificationManager:
    """Create a NotificationManager configured from environment variables.

    Reads os.environ at call time (not import time) for testability.
    """
    # Parse comma-separated email recipients
    alert_to = os.environ.get("ALERT_EMAIL_TO", "alerts@example.com")
    to_addrs = [addr.strip() for addr in alert_to.split(",") if addr.strip()]

    email = EmailNotifier(
        host=os.environ.get("SMTP_HOST", "smtp.example.com"),
        port=int(os.environ.get("SMTP_PORT", "25")),
        user=os.environ.get("SMTP_USER", ""),
        password=os.environ.get("SMTP_PASSWORD", ""),
        from_addr=os.environ.get("ALERT_EMAIL_FROM", "alerts@example.com"),
        to_addrs=to_addrs,
        use_tls=os.environ.get("SMTP_USE_TLS", "true").lower() in ("true", "1", "yes"),
    )

    webhook = WebhookNotifier(
        url=os.environ.get("WEBHOOK_URL", ""),
        timeout=int(os.environ.get("WEBHOOK_TIMEOUT", "10")),
    )

    return NotificationManager(email_notifier=email, webhook_notifier=webhook)
