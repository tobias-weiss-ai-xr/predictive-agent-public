"""AlertWatch feed — cluster-alert and storage-health correlation.

A lightweight, optional, never-raising watch layer that connects the agent's
per-pod risk signals to cluster-level facts it cannot see from pod metrics:

  * Alertmanager active alerts (``ALERTMANAGER_URL``) — a firing alert for a
    watched namespace nudges that namespace's pod risk scores up
    (``ALERT_FEED_RISK_BOOST``), so crash-loop / eviction alerts and the
    agent's own signals agree.
  * Storage health via Prometheus (``PROMETHEUS_URL``): parses
    ``ceph_health_status`` and exposes it on ``GET /ceph-health``; a
    WARN/ERR transition (from OK) raises an alerting event that the
    reconcile loop can drain to the notifier.

Nothing here ever raises; every fetch is bounded by a timeout, cached, and
returns neutral values ([]) on failure so the reconcile loop is never broken
by a dead Alertmanager/Prometheus.
"""

from __future__ import annotations

import json
import logging
import time
from base64 import b64encode
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# ceph_health_status prometheus metric: 1=OK, 2=WARN, 3=ERR
STATUS_ORDER = {"UNKNOWN": -1, "OK": 0, "WARN": 1, "ERR": 2}
CEPH_STATUS_BY_VALUE = {"1": "OK", "2": "WARN", "3": "ERR"}


def parse_alertmanager_alerts(payload) -> list:
    """Extract firing alerts with a usable namespace from Alertmanager v2 JSON."""
    alerts = []
    for a in payload or []:
        if a.get("status") != "firing":
            continue
        labels = a.get("labels") or {}
        ns = (
            labels.get("kubernetes_namespace_name")
            or labels.get("namespace")
            or ""
        )
        alerts.append(
            {
                "alertname": labels.get("alertname", "unknown"),
                "namespace": ns,
                "severity": labels.get("severity", ""),
                "pod": labels.get("pod") or labels.get("exported_pod") or "",
                "annotations": a.get("annotations") or {},
            }
        )
    return alerts


def parse_ceph_health(payload):
    """Parse a Prometheus promql result for ceph_health_status.

    Returns ``(status, detail)``; status in OK/WARN/ERR/UNKNOWN.
    """
    try:
        results = payload["data"]["result"]
    except (TypeError, KeyError):
        return "UNKNOWN", ""
    if not results:
        return "UNKNOWN", ""
    value = results[0].get("value")
    if not value or len(value) < 2:
        return "UNKNOWN", ""
    status = CEPH_STATUS_BY_VALUE.get(str(value[1]), "UNKNOWN")
    return status, str(value[1])


class AlertFeed:
    """Watch-layer feed over Alertmanager + Prometheus (+ Ceph health)."""

    def __init__(
        self,
        alertmanager_url: str = "",
        prometheus_url: str = "",
        prometheus_user: str = "",
        prometheus_pass: str = "",
        timeout_s: float = 10.0,
        boost: float = 0.0,
        ceph_watch: bool = False,
    ):
        self.alertmanager_url = alertmanager_url or ""
        self.prometheus_url = prometheus_url or ""
        self.prometheus_user = prometheus_user or ""
        self.prometheus_pass = prometheus_pass or ""
        self.timeout_s = float(timeout_s)
        self.boost = float(boost)
        self.ceph_watch = bool(ceph_watch)
        self._active_namespaces: set = set()
        self._active_alerts: list = []
        self._last_ceph_status: str = "UNKNOWN"
        self.last_alert_fetch: float = 0.0
        self.last_ceph_fetch: float = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.alertmanager_url)

    def _get_json(self, url: str, params=None, auth=None):
        """GET + JSON parse using stdlib only (no requests in the nix image)."""
        if params:
            sep = "&" if "?" in url else "?"
            url = url + sep + urlencode(params)
        request = Request(url)
        if auth and auth[0]:
            token = b64encode(f"{auth[0]}:{auth[1] or ''}".encode()).decode("ascii")
            request.add_header("Authorization", f"Basic {token}")
        try:
            with urlopen(request, timeout=self.timeout_s) as resp:
                if getattr(resp, "status", 200) != 200:
                    logger.warning("AlertWatch: %s -> HTTP %s", url, resp.status)
                    return None
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.debug("AlertWatch: %s fetch failed: %s", url, e)
            return None

    def fetch_active_alerts(self) -> list:
        """Fetch and parse Alertmanager active alerts; never raises."""
        if not self.alertmanager_url:
            return []
        payload = self._get_json(
            self.alertmanager_url.rstrip("/") + "/api/v2/alerts",
            params={"active": "true"},
        )
        alerts = parse_alertmanager_alerts(payload)
        self._active_alerts = alerts
        self._active_namespaces = {
            a["namespace"] for a in alerts if a["namespace"]
        }
        self.last_alert_fetch = time.time()
        return alerts

    def refresh(self) -> None:
        """Refresh alert state in one go (called by the reconcile loop).

        Ceph health is refreshed separately via ``ceph_transition_event`` so
        the status (and its change detection) is not double-fetched.
        """
        self.fetch_active_alerts()

    def pod_boost(self, namespace: str, pod_name: str = "") -> float:
        """Risk boost (0.0 or configured value) for a namespace with a firing alert."""
        if self.boost <= 0:
            return 0.0
        if not self._active_namespaces:
            return 0.0
        if namespace in self._active_namespaces:
            return self.boost
        return 0.0

    # ── Ceph health ─────────────────────────────────────────────────────────
    def fetch_ceph_health(self):
        """Parse ceph_health_status from Prometheus; None on failure/disabled."""
        if not self.prometheus_url:
            return None
        auth = None
        if self.prometheus_user:
            auth = (self.prometheus_user, self.prometheus_pass)
        payload = self._get_json(
            self.prometheus_url.rstrip("/") + "/api/v1/query",
            params={"query": "ceph_health_status"},
            auth=auth,
        )
        if payload is None:
            return None
        status, detail = parse_ceph_health(payload)
        self.last_ceph_fetch = time.time()
        return {"status": status, "detail": detail, "healthy": status == "OK"}

    def ceph_transition_event(self):
        """Return a dict describing a meaningful Ceph health transition.

        Emits an event when the cluster moves into WARN/ERR (from OK/UNKNOWN)
        or recovers back to OK from WARN/ERR. None when unchanged or unknown.
        Also records the new status for the next transition call.
        """
        if not self.ceph_watch or not self.prometheus_url:
            return None
        health = self.fetch_ceph_health()
        if health is None:
            return None
        status = health["status"]
        if status == "UNKNOWN":
            return None
        prev = self._last_ceph_status
        self._last_ceph_status = status
        new_order = STATUS_ORDER.get(status, -1)
        prev_order = STATUS_ORDER.get(prev, -1)
        if status in ("WARN", "ERR") and (prev in ("OK", "UNKNOWN") or new_order > prev_order):
            level = "error" if status == "ERR" else "warning"
            return {
                "kind": "ceph_health",
                "level": level,
                "status": status,
                "detail": health["detail"],
                "previous": prev,
            }
        if status == "OK" and prev in ("WARN", "ERR"):
            return {
                "kind": "ceph_health",
                "level": "info",
                "status": "OK",
                "detail": "recovered",
                "previous": prev,
            }
        return None
