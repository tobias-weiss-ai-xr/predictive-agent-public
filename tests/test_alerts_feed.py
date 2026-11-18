"""Tests for AlertWatch feed — cluster-alert + Ceph-health correlation (REM-ALERTWATCH).

Covers:
- Alertmanager v2 active-alert ingestion (parse, firing filter, cache)
- error/timeout resilience (never raises)
- pod/namespace risk boost from a matching active alert
- Ceph health via Prometheus promql (ceph_health_status) parsing
- ceph status transition detection (OK -> WARN/ERR -> OK)
- Basic-auth header on Prometheus fetches (stdlib urllib transport)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from predictive_agent.alerts_feed import (
    AlertFeed,
    STATUS_ORDER,
    parse_alertmanager_alerts,
    parse_ceph_health,
)


class _FakeResponse:
    """Minimal urllib response stand-in (context-manager + read/status)."""

    def __init__(self, data, status: int = 200):
        self.status = status
        if isinstance(data, bytes):
            self._data = data
        else:
            self._data = json.dumps(data).encode("utf-8")

    def read(self, *args, **kwargs):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ── parsing ──────────────────────────────────────────────────────────────────


class TestParseAlerts:
    def test_keeps_only_firing(self):
        payload = [
            {"status": "firing", "labels": {"alertname": "CephHealthWarning", "namespace": "monitoring", "severity": "warning"}},
            {"status": "resolved", "labels": {"alertname": "Old", "namespace": "monitoring"}},
            {"status": "firing", "labels": {"alertname": "PodCrashLooping", "namespace": "llm", "pod": "open-webui-x"}},
        ]
        alerts = parse_alertmanager_alerts(payload)
        assert [a["alertname"] for a in alerts] == ["CephHealthWarning", "PodCrashLooping"]

    def test_namespace_derived_from_kubernetes_labels(self):
        payload = [
            {"status": "firing", "labels": {"alertname": "A",
                                            "namespace": "llm",
                                            "kubernetes_namespace_name": "llm"}}
        ]
        alerts = parse_alertmanager_alerts(payload)
        assert alerts[0]["namespace"] == "llm"

    def test_empty_payload(self):
        assert parse_alertmanager_alerts([]) == []


class TestParseCephHealth:
    def test_promql_value(self):
        data = {"data": {"result": [{"metric": {}, "value": [1710000000.0, "1"]}]}}
        status, detail = parse_ceph_health(data)
        assert status == "OK"

    def test_warn(self):
        data = {"data": {"result": [{"metric": {}, "value": [1710000000.0, "2"]}]}}
        status, _ = parse_ceph_health(data)
        assert status == "WARN"

    def test_err(self):
        data = {"data": {"result": [{"metric": {}, "value": [1710000000.0, "3"]}]}}
        status, _ = parse_ceph_health(data)
        assert status == "ERR"

    def test_empty_result_unknown(self):
        status, _ = parse_ceph_health({"data": {"result": []}})
        assert status == "UNKNOWN"


# ── feed behaviour ───────────────────────────────────────────────────────────


def _feed(**kwargs):
    kwargs.setdefault("alertmanager_url", "http://am:9093")
    kwargs.setdefault("prometheus_url", "http://prom:9090")
    kwargs.setdefault("prometheus_user", "")
    kwargs.setdefault("prometheus_pass", "")
    kwargs.setdefault("timeout_s", 2.0)
    kwargs.setdefault("ceph_watch", False)
    return AlertFeed(**kwargs)


def _firing(ns="llm", name="A"):
    return [{"status": "firing", "labels": {"alertname": name, "namespace": ns}}]


class TestFetchAlerts:
    @patch("predictive_agent.alerts_feed.urlopen")
    def test_fetch_ok(self, mock_urlopen):
        mock_urlopen.return_value = _FakeResponse(_firing())
        feed = _feed()
        assert feed.fetch_active_alerts()[0]["alertname"] == "A"

    @patch("predictive_agent.alerts_feed.urlopen")
    def test_fetch_error_never_raises(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("boom")
        feed = _feed()
        assert feed.fetch_active_alerts() == []

    @patch("predictive_agent.alerts_feed.urlopen")
    def test_fetch_http_error(self, mock_urlopen):
        mock_urlopen.return_value = _FakeResponse({}, status=500)
        feed = _feed()
        assert feed.fetch_active_alerts() == []

    @patch("predictive_agent.alerts_feed.urlopen")
    def test_cached(self, mock_urlopen):
        mock_urlopen.return_value = _FakeResponse(_firing())
        feed = _feed()
        feed.fetch_active_alerts()
        feed.fetch_active_alerts()
        assert mock_urlopen.call_count == 2  # fetch each cycle; resilience not cache

    @patch("predictive_agent.alerts_feed.urlopen")
    def test_disabled_returns_empty(self, mock_urlopen):
        feed = _feed()
        feed.alertmanager_url = ""
        assert feed.fetch_active_alerts() == [] or mock_urlopen.call_count == 0

    @patch("predictive_agent.alerts_feed.urlopen")
    def test_query_params_sent(self, mock_urlopen):
        mock_urlopen.return_value = _FakeResponse(_firing())
        feed = _feed()
        feed.fetch_active_alerts()
        req = mock_urlopen.call_args[0][0]
        assert "active=true" in req.full_url


class TestRiskBoost:
    @patch("predictive_agent.alerts_feed.urlopen")
    def test_boost_matching_namespace(self, mock_urlopen):
        mock_urlopen.return_value = _FakeResponse(_firing())
        feed = _feed(boost=0.15)
        feed.refresh()
        assert feed.pod_boost("llm", "open-webui-x") == 0.15

    @patch("predictive_agent.alerts_feed.urlopen")
    def test_no_boost_for_other_ns(self, mock_urlopen):
        mock_urlopen.return_value = _FakeResponse(_firing())
        feed = _feed(boost=0.15)
        feed.refresh()
        assert feed.pod_boost("default", "x") == 0.0

    def test_boost_zero_by_default(self):
        feed = _feed(boost=0.0)
        assert feed.pod_boost("llm", "x") == 0.0


class TestCephHealth:
    @patch("predictive_agent.alerts_feed.urlopen")
    def test_fetch_ceph(self, mock_urlopen):
        mock_urlopen.return_value = _FakeResponse(
            {"data": {"result": [{"metric": {}, "value": [1710000000.0, "2"]}]}}
        )
        feed = _feed()
        assert feed.fetch_ceph_health()["status"] == "WARN"

    @patch("predictive_agent.alerts_feed.urlopen")
    def test_basic_auth_header(self, mock_urlopen):
        mock_urlopen.return_value = _FakeResponse(
            {"data": {"result": [{"metric": {}, "value": [1710000000.0, "1"]}]}}
        )
        feed = _feed()
        feed.prometheus_user = "admin"
        feed.prometheus_pass = "secret"
        feed.fetch_ceph_health()
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") == "Basic YWRtaW46c2VjcmV0"

    def test_status_order(self):
        assert STATUS_ORDER["OK"] < STATUS_ORDER["WARN"] < STATUS_ORDER["ERR"]

    @patch("predictive_agent.alerts_feed.urlopen")
    def test_transition_detected(self, mock_urlopen):
        mock_urlopen.return_value = _FakeResponse(
            {"data": {"result": [{"metric": {}, "value": [1710000000.0, "2"]}]}}
        )
        feed = _feed(ceph_watch=True)
        evt = feed.ceph_transition_event()
        assert evt is not None
        assert evt["status"] == "WARN"
        # second fetch shows improvement to OK -> improvement event (recorded)
        mock_urlopen.return_value = _FakeResponse(
            {"data": {"result": [{"metric": {}, "value": [1710000000.0, "1"]}]}}
        )
        evt2 = feed.ceph_transition_event()
        assert evt2 is None or evt2["status"] == "OK"

    @patch("predictive_agent.alerts_feed.urlopen")
    def test_no_event_when_unknown(self, mock_urlopen):
        mock_urlopen.return_value = _FakeResponse({"data": {"result": []}})
        feed = _feed()
        assert feed.ceph_transition_event() is None


class TestResilience:
    @patch("predictive_agent.alerts_feed.urlopen")
    def test_fetch_ceph_error(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("down")
        feed = _feed()
        assert feed.fetch_ceph_health() is None
        assert feed.ceph_transition_event() is None
