"""Tests for the email and webhook notifier (REM-6)."""

import json
import os
from unittest.mock import MagicMock, patch

from predictive_agent.notifier import (
    EmailNotifier,
    NotificationManager,
    WebhookNotifier,
    build_alertmanager_payload,
    create_notifier_from_config,
)


class TestEmailNotifier:
    """Test EmailNotifier with mocked SMTP."""

    def _make_notifier(self, **kwargs):
        defaults = dict(
            host="192.0.2.1", port=587, user="", password="",
            from_addr="predictive-agent@example.local",
            to_addrs=["alerts@example.org"], use_tls=True,
        )
        defaults.update(kwargs)
        return EmailNotifier(**defaults)

    @patch("predictive_agent.notifier.smtplib.SMTP")
    def test_send_basic(self, mock_smtp):
        """Test basic email sending with STARTTLS."""
        server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = server
        notifier = self._make_notifier()
        result = notifier.send("Test Alert", "Body text")
        assert result is True
        mock_smtp.assert_called_once_with("192.0.2.1", 587, timeout=10)
        server.starttls.assert_called_once()
        # No login when user/password empty
        server.login.assert_not_called()
        server.sendmail.assert_called_once()

    @patch("predictive_agent.notifier.smtplib.SMTP")
    def test_send_with_auth(self, mock_smtp):
        """Test email sending with SMTP auth."""
        server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = server
        notifier = self._make_notifier(user="user@example.org", password="secret")
        result = notifier.send("Test", "Body")
        assert result is True
        server.starttls.assert_called_once()
        server.login.assert_called_once_with("user@example.org", "secret")

    @patch("predictive_agent.notifier.smtplib.SMTP")
    def test_send_no_tls(self, mock_smtp):
        """Test email sending without TLS."""
        server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = server
        notifier = self._make_notifier(use_tls=False)
        result = notifier.send("Test", "Body")
        assert result is True
        server.starttls.assert_not_called()

    @patch("predictive_agent.notifier.smtplib.SMTP")
    def test_send_subject_format(self, mock_smtp):
        """Test that subject is prefixed with [SCS-Predictive-Agent]."""
        server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = server
        notifier = self._make_notifier()
        notifier.send("High Risk", "Body")
        # Check sendmail was called with proper subject
        sendmail_args = server.sendmail.call_args
        msg_str = sendmail_args[0][2]
        assert "[SCS-Predictive-Agent] High Risk" in msg_str

    @patch("predictive_agent.notifier.smtplib.SMTP")
    def test_send_to_multiple_recipients(self, mock_smtp):
        """Test sending to multiple recipients."""
        server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = server
        notifier = self._make_notifier(
            to_addrs=["alerts@example.org", "admin@scs.example"]
        )
        result = notifier.send("Test", "Body")
        assert result is True
        sendmail_args = server.sendmail.call_args
        recipients = sendmail_args[0][1]
        assert "alerts@example.org" in recipients
        assert "admin@scs.example" in recipients

    @patch("predictive_agent.notifier.smtplib.SMTP")
    def test_send_smtp_exception_returns_false(self, mock_smtp):
        """Test that SMTPException returns False, not raises."""
        import smtplib
        mock_smtp.side_effect = smtplib.SMTPException("Connection refused")
        notifier = self._make_notifier()
        result = notifier.send("Test", "Body")
        assert result is False

    @patch("predictive_agent.notifier.smtplib.SMTP")
    def test_send_timeout_returns_false(self, mock_smtp):
        """Test that timeout returns False."""
        mock_smtp.side_effect = TimeoutError("timed out")
        notifier = self._make_notifier()
        result = notifier.send("Test", "Body")
        assert result is False

    @patch("predictive_agent.notifier.smtplib.SMTP")
    def test_send_generic_exception_returns_false(self, mock_smtp):
        """Test that any exception returns False."""
        mock_smtp.side_effect = Exception("unexpected error")
        notifier = self._make_notifier()
        result = notifier.send("Test", "Body")
        assert result is False

    @patch("predictive_agent.notifier.smtplib.SMTP")
    def test_default_recipient_is_configured(self, mock_smtp):
        """Default recipient is alerts@example.com (HRZ relay)."""
        notifier = EmailNotifier()
        assert "alerts@example.com" in notifier.to_addrs
        assert notifier.host == "smtp.example.com"
        assert notifier.port == 25

    @patch("predictive_agent.notifier.smtplib.SMTP")
    def test_email_body_contains_alert_details(self, mock_smtp):
        """Test that email body contains timestamp, pod name, risk score, action."""
        server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = server
        notifier = self._make_notifier()
        notifier.send("High Risk", "Pod: api-server-abc\nRisk: 85.5\nAction: pod_restart")
        sendmail_args = server.sendmail.call_args
        msg_str = sendmail_args[0][2]
        assert "High Risk" in msg_str
        assert "api-server-abc" in msg_str
        assert "85.5" in msg_str
        assert "pod_restart" in msg_str


class TestWebhookNotifier:
    """Test WebhookNotifier with mocked urllib."""

    @patch("predictive_agent.notifier.urlopen")
    def test_send_basic(self, mock_urlopen):
        """Test basic webhook send."""
        response = MagicMock()
        response.status = 200
        mock_urlopen.return_value.__enter__.return_value = response
        notifier = WebhookNotifier(url="http://example.com/webhook")
        result = notifier.send({"alert": "test", "risk": 85})
        assert result is True
        mock_urlopen.assert_called_once()

    @patch("predictive_agent.notifier.urlopen")
    def test_send_json_payload(self, mock_urlopen):
        """Test that webhook sends JSON payload."""
        response = MagicMock()
        response.status = 200
        mock_urlopen.return_value.__enter__.return_value = response
        notifier = WebhookNotifier(url="http://example.com/webhook")
        payload = {"alert_type": "high_risk", "pod": "api-server", "risk_score": 90.5}
        notifier.send(payload)
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.method == "POST"
        assert req.headers.get("Content-type") == "application/json"
        data = json.loads(req.data.decode("utf-8"))
        assert data["alert_type"] == "high_risk"
        assert data["pod"] == "api-server"
        assert data["risk_score"] == 90.5

    def test_send_empty_url_returns_true(self):
        """Test that empty URL returns True (no-op)."""
        notifier = WebhookNotifier(url="")
        result = notifier.send({"alert": "test"})
        assert result is True

    @patch("predictive_agent.notifier.urlopen")
    def test_send_url_error_returns_false(self, mock_urlopen):
        """Test that URLError returns False."""
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection refused")
        notifier = WebhookNotifier(url="http://example.com/webhook")
        result = notifier.send({"alert": "test"})
        assert result is False

    @patch("predictive_agent.notifier.urlopen")
    def test_send_non_200_returns_false(self, mock_urlopen):
        """Test that non-200 status returns False."""
        response = MagicMock()
        response.status = 500
        mock_urlopen.return_value.__enter__.return_value = response
        notifier = WebhookNotifier(url="http://example.com/webhook")
        result = notifier.send({"alert": "test"})
        assert result is False

    @patch("predictive_agent.notifier.urlopen")
    def test_send_timeout(self, mock_urlopen):
        """Test that timeout is passed to urlopen."""
        response = MagicMock()
        response.status = 200
        mock_urlopen.return_value.__enter__.return_value = response
        notifier = WebhookNotifier(url="http://example.com/webhook", timeout=30)
        notifier.send({"alert": "test"})
        call_kwargs = mock_urlopen.call_args[1]
        assert call_kwargs["timeout"] == 30


class TestNotificationManager:
    """Test NotificationManager."""

    def _make_manager(self):
        email = MagicMock(spec=EmailNotifier)
        email.send.return_value = True
        webhook = MagicMock(spec=WebhookNotifier)
        webhook.send.return_value = True
        return NotificationManager(email_notifier=email, webhook_notifier=webhook), email, webhook

    def test_notify_calls_both(self):
        """Test that notify() calls both email and webhook."""
        manager, email, webhook = self._make_manager()
        email_sent, webhook_sent = manager.notify(
            "high_risk", "api-server-abc", 85.5, "pod_restart", "Pod in CrashLoopBackOff"
        )
        assert email_sent is True
        assert webhook_sent is True
        email.send.assert_called_once()
        webhook.send.assert_called_once()

    def test_notify_email_subject(self):
        """Test that email subject contains alert type and pod name."""
        manager, email, webhook = self._make_manager()
        manager.notify("remediation", "pod-1", 90.0, "node_cordon")
        call_args = email.send.call_args
        subject = call_args[0][0]
        assert "remediation" in subject
        assert "pod-1" in subject

    def test_notify_email_body_contains_details(self):
        """Test that email body contains all alert details."""
        manager, email, webhook = self._make_manager()
        manager.notify("high_risk", "api-server", 85.5, "pod_restart", "CrashLoopBackOff")
        call_args = email.send.call_args
        body = call_args[0][1]
        assert "api-server" in body
        assert "85.5" in body
        assert "pod_restart" in body
        assert "CrashLoopBackOff" in body

    def test_notify_webhook_payload(self):
        """Test that webhook payload contains all fields."""
        manager, email, webhook = self._make_manager()
        manager.notify("high_risk", "api-server", 85.5, "pod_restart", "details")
        call_args = webhook.send.call_args
        payload = call_args[0][0]
        assert payload["alert_type"] == "high_risk"
        assert payload["pod_name"] == "api-server"
        assert payload["risk_score"] == 85.5
        assert payload["action_taken"] == "pod_restart"
        assert payload["details"] == "details"
        assert "timestamp" in payload

    def test_notify_history(self):
        """Test that notifications are recorded in history."""
        manager, email, webhook = self._make_manager()
        for i in range(5):
            manager.notify("alert", f"pod-{i}", 80.0, "action")
        history = manager.get_history()
        assert len(history) == 5
        assert history[0]["pod_name"] == "pod-0"
        assert history[4]["pod_name"] == "pod-4"

    def test_notify_history_limit(self):
        """Test that history is capped at 100 entries."""
        manager, email, webhook = self._make_manager()
        for i in range(105):
            manager.notify("alert", f"pod-{i}", 80.0, "action")
        history = manager.get_history(limit=200)
        assert len(history) == 100

    def test_notify_email_failure(self):
        """Test that email failure is recorded."""
        email = MagicMock(spec=EmailNotifier)
        email.send.return_value = False
        webhook = MagicMock(spec=WebhookNotifier)
        webhook.send.return_value = True
        manager = NotificationManager(email_notifier=email, webhook_notifier=webhook)
        email_sent, webhook_sent = manager.notify("alert", "pod-1", 80.0, "action")
        assert email_sent is False
        assert webhook_sent is True
        assert manager.history[-1]["email_sent"] is False
        assert manager.history[-1]["webhook_sent"] is True


class TestCreateNotifierFromConfig:
    """Test create_notifier_from_config()."""

    def test_defaults(self):
        with patch.dict(os.environ, {
            "ALERT_EMAIL_TO": "alerts@example.org",
            "ALERT_EMAIL_FROM": "predictive-agent@example.local",
            "SMTP_HOST": "192.0.2.1",
            "SMTP_PORT": "587",
            "SMTP_USER": "",
            "SMTP_PASSWORD": "",
            "SMTP_USE_TLS": "true",
            "WEBHOOK_URL": "",
            "WEBHOOK_TIMEOUT": "10",
        }):
            manager = create_notifier_from_config()
            assert manager.email.host == "192.0.2.1"
            assert manager.email.port == 587
            assert "alerts@example.org" in manager.email.to_addrs
            assert manager.email.use_tls is True
            assert manager.webhook.url == ""

    def test_multiple_recipients(self):
        with patch.dict(os.environ, {
            "ALERT_EMAIL_TO": "alerts@example.org,admin@scs.example",
            "SMTP_HOST": "192.0.2.1",
            "SMTP_PORT": "587",
            "SMTP_USE_TLS": "true",
            "WEBHOOK_URL": "",
        }):
            manager = create_notifier_from_config()
            assert len(manager.email.to_addrs) == 2
            assert "alerts@example.org" in manager.email.to_addrs
            assert "admin@scs.example" in manager.email.to_addrs

    def test_webhook_url(self):
        with patch.dict(os.environ, {
            "ALERT_EMAIL_TO": "test@example.com",
            "SMTP_HOST": "smtp.example.com",
            "SMTP_PORT": "587",
            "WEBHOOK_URL": "http://webhook.example.com/alert",
            "WEBHOOK_TIMEOUT": "30",
        }):
            manager = create_notifier_from_config()
            assert manager.webhook.url == "http://webhook.example.com/alert"
            assert manager.webhook.timeout == 30


class TestBuildAlertmanagerPayload:
    """Test Alertmanager v4-compatible payload construction."""

    def test_top_level_structure(self):
        """Test the Alertmanager v4 envelope structure."""
        payload = build_alertmanager_payload("high_risk", "default/api-server", 85.5)
        assert payload["version"] == "4"
        assert payload["status"] == "firing"
        assert payload["receiver"] == "incident-bridge"
        assert payload["truncatedAlerts"] == 0
        assert isinstance(payload["alerts"], list)
        assert len(payload["alerts"]) == 1
        assert "groupLabels" in payload
        assert "commonLabels" in payload
        assert "commonAnnotations" in payload

    def test_alert_entry_fields(self):
        """Test each alert entry has labels, annotations, startsAt, endsAt."""
        payload = build_alertmanager_payload(
            "remediation", "opendesk/pod-1", 92.0, action_taken="pod_restart"
        )
        alert = payload["alerts"][0]
        assert alert["status"] == "firing"
        assert isinstance(alert["labels"], dict)
        assert isinstance(alert["annotations"], dict)
        assert alert["startsAt"].endswith("Z")
        assert "endsAt" in alert
        assert "generatorURL" in alert

    def test_labels_derived_from_pod_key(self):
        """Test namespace/pod labels are parsed from the ns/name pod key."""
        payload = build_alertmanager_payload("high_risk", "default/api-server-abc", 85.5)
        labels = payload["alerts"][0]["labels"]
        assert labels["alertname"] == "high_risk"
        assert labels["namespace"] == "default"
        assert labels["pod"] == "api-server-abc"
        assert "severity" in labels

    def test_bare_pod_name_has_no_namespace(self):
        """Test a pod name without a namespace slash sets pod only."""
        payload = build_alertmanager_payload("high_risk", "api-server", 50.0)
        labels = payload["alerts"][0]["labels"]
        assert labels["pod"] == "api-server"
        assert "namespace" not in labels

    def test_severity_mapping(self):
        """Test severity is derived from the real risk score."""
        assert build_alertmanager_payload("a", "x/y", 95.0)["alerts"][0]["labels"]["severity"] == "critical"
        assert build_alertmanager_payload("a", "x/y", 80.0)["alerts"][0]["labels"]["severity"] == "warning"
        assert build_alertmanager_payload("a", "x/y", 45.0)["alerts"][0]["labels"]["severity"] == "info"

    def test_severity_override(self):
        """Test an explicit severity wins over the risk-derived value."""
        payload = build_alertmanager_payload("a", "x/y", 50.0, severity="critical")
        assert payload["alerts"][0]["labels"]["severity"] == "critical"

    def test_action_label_and_annotation(self):
        """Test the remediation action appears in labels and annotations."""
        payload = build_alertmanager_payload(
            "remediation", "opendesk/pod-1", 90.0, action_taken="node_cordon"
        )
        alert = payload["alerts"][0]
        assert alert["labels"]["action"] == "node_cordon"
        assert alert["annotations"]["action"] == "node_cordon"

    def test_annotations(self):
        """Test summary, description and risk_score annotations."""
        payload = build_alertmanager_payload(
            "high_risk", "default/pod-1", 85.5, details="CrashLoopBackOff"
        )
        annotations = payload["alerts"][0]["annotations"]
        assert annotations["summary"] == "high_risk: default/pod-1"
        assert annotations["description"] == "CrashLoopBackOff"
        assert annotations["risk_score"] == "85.5"

    def test_firing_ends_at_is_zero_time(self):
        """Test firing alerts carry the Alertmanager zero end time."""
        payload = build_alertmanager_payload("high_risk", "default/pod-1", 80.0)
        assert payload["alerts"][0]["endsAt"] == "0001-01-01T00:00:00Z"

    def test_resolved_ends_at_matches_start(self):
        """Test resolved alerts get an explicit end time."""
        starts_at = "2026-08-29T10:00:00Z"
        payload = build_alertmanager_payload(
            "high_risk", "default/pod-1", 80.0, status="resolved", starts_at=starts_at
        )
        assert payload["status"] == "resolved"
        assert payload["alerts"][0]["status"] == "resolved"
        assert payload["alerts"][0]["startsAt"] == starts_at
        assert payload["alerts"][0]["endsAt"] == starts_at

    def test_custom_receiver_and_generator(self):
        """Test custom receiver and external URL are honored."""
        payload = build_alertmanager_payload(
            "high_risk",
            "default/pod-1",
            80.0,
            receiver="custom",
            external_url="http://agent:8080/status",
        )
        assert payload["receiver"] == "custom"
        assert payload["externalURL"] == "http://agent:8080/status"
        assert payload["alerts"][0]["generatorURL"] == "http://agent:8080/status"

    def test_extra_labels_merged(self):
        """Test extra labels are merged onto the alert."""
        payload = build_alertmanager_payload(
            "high_risk", "default/pod-1", 80.0, extra_labels={"node": "edge-node"}
        )
        assert payload["alerts"][0]["labels"]["node"] == "edge-node"

    def test_json_serializable(self):
        """Test the payload can be serialized to JSON."""
        payload = build_alertmanager_payload("high_risk", "default/pod-1", 80.0)
        dumped = json.dumps(payload)
        assert "startsAt" in dumped
        assert "alerts" in dumped


class TestWebhookAlertmanagerSend:
    """Test WebhookNotifier.send_alertmanager with mocked urllib."""

    def _webhook(self, url="http://example.com/incidents"):
        return WebhookNotifier(url=url)

    @patch("predictive_agent.notifier.urlopen")
    def test_sends_alertmanager_payload(self, mock_urlopen):
        """Test send_alertmanager POSTs an Alertmanager-compatible payload."""
        response = MagicMock()
        response.status = 200
        mock_urlopen.return_value.__enter__.return_value = response
        ok = self._webhook().send_alertmanager(
            "high_risk", "default/pod-1", 85.5, action_taken="pod_restart"
        )
        assert ok is True
        req = mock_urlopen.call_args[0][0]
        assert req.method == "POST"
        data = json.loads(req.data.decode("utf-8"))
        assert data["receiver"] == "incident-bridge"
        assert data["alerts"][0]["labels"]["pod"] == "pod-1"
        assert "startsAt" in data["alerts"][0]

    def test_empty_url_returns_true(self):
        """Test empty URL is a no-op, not an error."""
        notifier = WebhookNotifier(url="")
        assert notifier.send_alertmanager("high_risk", "default/pod-1", 80.0) is True

    @patch("predictive_agent.notifier.urlopen")
    def test_endpoint_unreachable_returns_false(self, mock_urlopen):
        """Test an unreachable incident-bridge returns False, does not raise."""
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection refused")
        ok = self._webhook().send_alertmanager("high_risk", "default/pod-1", 80.0)
        assert ok is False

    @patch("predictive_agent.notifier.urlopen")
    def test_non_200_returns_false(self, mock_urlopen):
        """Test a non-200 response returns False."""
        response = MagicMock()
        response.status = 500
        mock_urlopen.return_value.__enter__.return_value = response
        ok = self._webhook().send_alertmanager("high_risk", "default/pod-1", 80.0)
        assert ok is False


class TestNotificationManagerAlertmanager:
    """Test NotificationManager.notify_alertmanager."""

    def _make_manager(self):
        email = MagicMock(spec=EmailNotifier)
        email.send.return_value = True
        webhook = MagicMock(spec=WebhookNotifier)
        webhook.send.return_value = True
        return NotificationManager(email_notifier=email, webhook_notifier=webhook), email, webhook

    def test_sends_webhook_payload(self):
        """Test notify_alertmanager sends only via webhook with the right shape."""
        manager, email, webhook = self._make_manager()
        ok = manager.notify_alertmanager(
            "high_risk", "default/pod-1", 85.5, action_taken="pod_restart"
        )
        assert ok is True
        webhook.send.assert_called_once()
        payload = webhook.send.call_args[0][0]
        assert payload["receiver"] == "incident-bridge"
        assert payload["alerts"][0]["labels"]["alertname"] == "high_risk"
        email.send.assert_not_called()

    def test_history_recorded(self):
        """Test the alertmanager emission is recorded in history."""
        manager, email, webhook = self._make_manager()
        manager.notify_alertmanager(
            "remediation", "opendesk/pod-1", 90.0, action_taken="node_cordon"
        )
        entry = manager.get_history()[-1]
        assert entry["format"] == "alertmanager"
        assert entry["pod_name"] == "opendesk/pod-1"
        assert entry["status"] == "firing"
        assert entry["webhook_sent"] is True

    def test_failure_recorded_and_returns_false(self):
        """Test a failed webhook delivery is recorded and returns False."""
        manager, email, webhook = self._make_manager()
        webhook.send.return_value = False
        ok = manager.notify_alertmanager("high_risk", "default/pod-1", 80.0)
        assert ok is False
        entry = manager.get_history()[-1]
        assert entry["webhook_sent"] is False

    def test_resolved_status_forwarded(self):
        """Test the firing/resolved status is forwarded into the payload."""
        manager, email, webhook = self._make_manager()
        manager.notify_alertmanager("high_risk", "default/pod-1", 80.0, status="resolved")
        payload = webhook.send.call_args[0][0]
        assert payload["status"] == "resolved"
        assert payload["alerts"][0]["status"] == "resolved"
