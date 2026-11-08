"""Second batch of edge-case tests for predictive_agent.server HTTP handling.

Complements ``tests/test_server_edges.py``: this module drills into the
*remaining* uncovered branches of ``predictive_agent/server.py``:

1. ``POST /remediate`` — missing remediation manager (valid JSON body), success
   path returning executed results to the caller.
2. ``POST /plan`` — missing manager, malformed JSON, unknown pod, invalid state,
   bad-horizon fallback, plan-only vs ``execute=true``, and ``Content-Length``
   handling (empty / zero-length bodies).
3. The remaining ``do_POST`` routes: ``/reanalyze``, ``/cost-model``,
   ``/coordinate``, ``/review`` — each with its missing-manager, validation
   (400), and success/error branches.
4. The remaining ``do_GET`` endpoints' error/degraded branches:
   ``/ready`` (degraded/down collector -> 503), ``/state``, ``/history``,
   ``/cache``, ``/remediate``, ``/remediation-log``, ``/plan`` (all
   strategies), ``/world-model``, ``/notifications``, ``/backtest``,
   ``/benchmark``, ``/shadow``, ``/calibration``, ``/decision-metrics``,
   ``/cost-model``, ``/policy/health``, ``/policy/stats``.

The handlers are driven entirely in-process (no sockets/threads), exactly like
``tests/test_server_edges.py``. The module-level ``_context`` is stubbed per
test and restored automatically.
"""

import io
import json
from types import SimpleNamespace
from typing import Dict, Optional, Tuple
from unittest import mock

import pytest

import predictive_agent.server as server
from predictive_agent.collector import (
    COLLECTOR_STATE_DEGRADED,
    COLLECTOR_STATE_DOWN,
    COLLECTOR_STATE_HEALTHY,
)
from predictive_agent.server import RequestHandler, _ServerState


class _NoCloseBytesIO(io.BytesIO):
    """BytesIO whose ``close`` is a no-op so the response stays readable."""

    def close(self):
        pass


class _FakeSocket:
    """Minimal stand-in for a connected socket (mirrors test_server_edges.py)."""

    def __init__(self, raw_request):
        self.rfile = io.BytesIO(raw_request)
        self.wfile = _NoCloseBytesIO()

    def makefile(self, mode, *args, **kwargs):
        return self.rfile if mode.startswith("r") else self.wfile

    def sendall(self, data):
        self.wfile.write(data)

    def shutdown(self, how):
        pass

    def close(self):
        pass


class _TestRequestHandler(RequestHandler):
    wbufsize = 65536


def _dispatch(method: str, path: str, body: bytes = b"", headers: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    """Run one full HTTP request through RequestHandler; return (status, body_str)."""
    lines = [f"{method} {path} HTTP/1.0"]
    headers = dict(headers or {})
    if body:
        headers.setdefault("Content-Length", str(len(body)))
    for name, value in headers.items():
        lines.append(f"{name}: {value}")
    raw = ("\r\n".join(lines) + "\r\n\r\n").encode() + body

    sock = _FakeSocket(raw)
    _TestRequestHandler(sock, ("127.0.0.1", 0), mock.Mock())

    response = sock.wfile.getvalue()
    head, _, payload = response.partition(b"\r\n\r\n")
    status_line = head.split(b"\r\n", 1)[0].decode("utf-8", "replace")
    status = int(status_line.split(" ", 2)[1])
    return status, payload.decode("utf-8", "replace")


def _json(result: Tuple[int, str]) -> Tuple[int, dict]:
    """Return (status, parsed_json) for a ``_dispatch`` result."""
    status, body = result
    return status, json.loads(body)


@pytest.fixture
def ctx():
    """Stub the module-level server context and restore it after the test."""
    fresh = _ServerState()
    with mock.patch.object(server, "_context", fresh):
        yield fresh


def _ns_pod(state: str = "CRITICAL", namespace: str = "ns") -> SimpleNamespace:
    """A minimal stand-in PodTracker with the attributes handlers read."""
    return SimpleNamespace(
        namespace=namespace,
        pod_key=f"{namespace}/pod",
        node_name="node-1",
        state=state,
    )


def _result(action: str = "pod_restart"):
    """A minimal RemediationResult-shaped object the handlers serialize."""
    return SimpleNamespace(
        action=action,
        target="ns/pod",
        success=True,
        dry_run=True,
        message="ok",
        timestamp="2026-01-01T00:00:00Z",
        command="kubectl rollout restart",
    )


# ── POST /remediate ─────────────────────────────────────────────────────────
def test_remediate_post_missing_manager_valid_json_returns_503(ctx):
    # Even a *well-formed* body must get a structured 503 when no remediation
    # manager is initialized (the malformed-JSON variant is in test_server_edges).
    status, body = _json(_dispatch(
        "POST", "/remediate",
        body=json.dumps({"pod_name": "ns/pod", "risk_score": 80.0}).encode(),
        headers={"Content-Type": "application/json"},
    ))
    assert status == 503
    assert body.get("status") == "error"
    assert "not initialized" in body.get("error", "").lower()


def test_remediate_post_success_returns_executed_results(ctx):
    # Happy path: pod resolves through the state model and the manager's
    # evaluation results are returned to the caller untouched.
    ctx.remediation_manager = mock.MagicMock()
    ctx.remediation_manager.evaluate.return_value = [_result("pod_restart")]
    ctx.state_model = SimpleNamespace(pods={"ns/pod": _ns_pod()})
    status, body = _json(_dispatch(
        "POST", "/remediate",
        body=json.dumps({"pod_name": "ns/pod", "risk_score": 85.0}).encode(),
        headers={"Content-Type": "application/json"},
    ))
    assert status == 200
    assert body.get("status") == "ok"
    assert body.get("pod_name") == "ns/pod"
    assert body.get("risk_score") == 85.0
    assert body["results"][0]["action"] == "pod_restart"
    ctx.remediation_manager.evaluate.assert_called_once()


# ── POST /plan ──────────────────────────────────────────────────────────────
def test_plan_post_missing_manager_returns_503(ctx):
    status, body = _json(_dispatch(
        "POST", "/plan", body=json.dumps({"state": "CRITICAL"}).encode(),
    ))
    assert status == 503
    assert "not initialized" in body.get("error", "").lower()


def test_plan_post_malformed_json_returns_400(ctx):
    ctx.remediation_manager = mock.MagicMock()
    status, body = _json(_dispatch(
        "POST", "/plan",
        body=b'{"state": "CRITICAL",',
        headers={"Content-Type": "application/json"},
    ))
    assert status == 400
    assert "Invalid JSON" in body.get("error", "")


def test_plan_post_unknown_pod_returns_404(ctx):
    ctx.remediation_manager = mock.MagicMock()
    ctx.state_model = SimpleNamespace(pods={})
    status, body = _json(_dispatch(
        "POST", "/plan",
        body=json.dumps({"pod_name": "ns/ghost"}).encode(),
    ))
    assert status == 404
    assert "not found" in body.get("error", "").lower()


def test_plan_post_invalid_state_returns_400(ctx):
    ctx.remediation_manager = mock.MagicMock()
    status, body = _json(_dispatch(
        "POST", "/plan", body=json.dumps({"state": "BOGUS"}).encode(),
    ))
    assert status == 400
    assert "invalid" in body.get("error", "").lower()


def test_plan_post_bad_horizon_falls_back_to_default(ctx):
    # A non-integer horizon must not crash the handler: it falls back to the
    # default horizon and still produces a plan.
    ctx.remediation_manager = mock.MagicMock()
    ctx.remediation_manager.plan_remediation.return_value = {"steps": []}
    ctx.state_model = SimpleNamespace(pods={"ns/pod": _ns_pod()})
    status, body = _json(_dispatch(
        "POST", "/plan",
        body=json.dumps({"pod_name": "ns/pod", "horizon": "abc"}).encode(),
    ))
    assert status == 200
    assert body.get("status") == "ok"
    ctx.remediation_manager.plan_remediation.assert_called_once_with("CRITICAL", 3)


def test_plan_post_plan_only_does_not_execute(ctx):
    # Default execute=false must produce a plan with no execution and never
    # call plan_and_execute.
    ctx.remediation_manager = mock.MagicMock()
    ctx.remediation_manager.plan_remediation.return_value = {"steps": []}
    ctx.state_model = SimpleNamespace(pods={"ns/pod": _ns_pod()})
    status, body = _json(_dispatch(
        "POST", "/plan",
        body=json.dumps({"pod_name": "ns/pod", "state": "CRITICAL"}).encode(),
    ))
    assert status == 200
    assert body.get("execution") is None
    ctx.remediation_manager.plan_and_execute.assert_not_called()


def test_plan_post_execute_true_runs_plan(ctx):
    # execute=true on a known pod runs the plan through the manager and returns
    # the execution transcript.
    ctx.remediation_manager = mock.MagicMock()
    ctx.remediation_manager.plan_remediation.return_value = {"steps": []}
    ctx.remediation_manager.plan_and_execute.return_value = [_result("pod_restart")]
    ctx.state_model = SimpleNamespace(pods={"ns/pod": _ns_pod()})
    status, body = _json(_dispatch(
        "POST", "/plan",
        body=json.dumps({
            "pod_name": "ns/pod", "risk_score": 0.9, "execute": True
        }).encode(),
    ))
    assert status == 200
    assert body.get("status") == "ok"
    assert body["execution"][0]["action"] == "pod_restart"
    ctx.remediation_manager.plan_and_execute.assert_called_once()


def test_plan_post_execute_requires_pod_name(ctx):
    # execute=true without a resolvable pod is rejected with 400 (the planner
    # would otherwise have no tracker to act on).
    ctx.remediation_manager = mock.MagicMock()
    status, body = _json(_dispatch(
        "POST", "/plan",
        body=json.dumps({"state": "CRITICAL", "execute": True}).encode(),
    ))
    assert status == 400
    assert "requires pod_name" in body.get("error", "")


# ── Content-Length handling ─────────────────────────────────────────────────
def test_plan_post_zero_content_length_treated_as_empty_body(ctx):
    # Explicit Content-Length: 0 means "no body" — the handler must not block
    # on rfile nor crash; it proceeds with an empty payload and validates it.
    ctx.remediation_manager = mock.MagicMock()
    status, body = _json(_dispatch(
        "POST", "/plan", body=b"", headers={"Content-Length": "0"},
    ))
    assert status == 400  # empty payload -> no pod_name / no state
    assert isinstance(body, dict)
    assert "error" in body


# ── POST /reanalyze ─────────────────────────────────────────────────────────
def test_reanalyze_no_callback_returns_ok(ctx):
    status, body = _json(_dispatch("POST", "/reanalyze"))
    assert status == 200
    assert body.get("status") == "ok"


def test_reanalyze_callback_result_reported(ctx):
    ctx.reconcile_callback = mock.Mock(return_value={"reconciled": 3})
    status, body = _json(_dispatch("POST", "/reanalyze"))
    assert status == 200
    assert body["result"] == {"reconciled": 3}


def test_reanalyze_callback_error_returns_503(ctx):
    # A failing reconcile callback is surfaced as a structured 503, never a 500.
    def boom():
        raise RuntimeError("reconcile exploded")

    ctx.reconcile_callback = boom
    status, body = _json(_dispatch("POST", "/reanalyze"))
    assert status == 503
    assert "reconcile exploded" in body.get("error", "")


# ── POST /cost-model ────────────────────────────────────────────────────────
def test_cost_model_post_malformed_json_returns_400(ctx):
    ctx.remediation_manager = mock.MagicMock()
    status, body = _json(_dispatch(
        "POST", "/cost-model",
        body=b'{"incidents": [',
        headers={"Content-Type": "application/json"},
    ))
    assert status == 400
    assert "Invalid JSON" in body.get("error", "")


def test_cost_model_post_missing_manager_returns_503(ctx):
    status, body = _json(_dispatch(
        "POST", "/cost-model", body=json.dumps({"incidents": []}).encode(),
    ))
    assert status == 503
    assert "not initialized" in body.get("error", "").lower()


def test_cost_model_post_incidents_required_returns_400(ctx):
    ctx.remediation_manager = mock.MagicMock()
    status, body = _json(_dispatch(
        "POST", "/cost-model", body=json.dumps({"slo_cost_per_minute": 5.0}).encode(),
    ))
    assert status == 400
    assert "incidents" in body.get("error", "")


def test_cost_model_post_calibration_error_returns_400(ctx):
    ctx.remediation_manager = mock.MagicMock()
    ctx.remediation_manager.calibrate_from_incidents.side_effect = ValueError("bad data")
    status, body = _json(_dispatch(
        "POST", "/cost-model", body=json.dumps({"incidents": [{"action": "x"}]}).encode(),
    ))
    assert status == 400
    assert "bad data" in body.get("error", "")


def test_cost_model_post_success_returns_costs(ctx):
    ctx.remediation_manager = mock.MagicMock()
    ctx.remediation_manager.calibrate_from_incidents.return_value = {
        "failure_cost": 500.0,
        "action_costs": {"pod_restart": 10.0},
    }
    status, body = _json(_dispatch(
        "POST", "/cost-model",
        body=json.dumps({"incidents": [{"action": "pod_restart", "downtime_minutes": 12}]}).encode(),
    ))
    assert status == 200
    assert body.get("status") == "ok"
    assert body.get("source") == "calibrated"
    assert body["failure_cost"] == 500.0
    assert body["action_costs"]["pod_restart"] == 10.0


# ── POST /coordinate ────────────────────────────────────────────────────────
def test_coordinate_post_missing_manager_returns_503(ctx):
    status, body = _json(_dispatch(
        "POST", "/coordinate", body=json.dumps({"pods": {"a": "CRITICAL"}}).encode(),
    ))
    assert status == 503
    assert "not initialized" in body.get("error", "").lower()


def test_coordinate_post_malformed_json_returns_400(ctx):
    ctx.remediation_manager = mock.MagicMock()
    status, body = _json(_dispatch(
        "POST", "/coordinate",
        body=b'{"pods": {' ,
        headers={"Content-Type": "application/json"},
    ))
    assert status == 400
    assert "Invalid JSON" in body.get("error", "")


def test_coordinate_post_pods_required_returns_400(ctx):
    ctx.remediation_manager = mock.MagicMock()
    status, body = _json(_dispatch(
        "POST", "/coordinate", body=json.dumps({"horizon": 3}).encode(),
    ))
    assert status == 400
    assert "pods" in body.get("error", "")


def test_coordinate_post_success_returns_joint_plan(ctx):
    ctx.remediation_manager = mock.MagicMock()
    coordinator_cls = mock.MagicMock()
    coordinator_cls.return_value.coordinate.return_value = {"waves": [["pod_restart"]]}
    with mock.patch("predictive_agent.coordinator.MultiAgentCoordinator", coordinator_cls):
        status, body = _json(_dispatch(
            "POST", "/coordinate",
            body=json.dumps({"pods": {"ns/a": {"state": "CRITICAL", "node": "n1"}}}).encode(),
        ))
    assert status == 200
    assert body.get("status") == "ok"
    assert body["waves"] == [["pod_restart"]]


def test_coordinate_post_coordinator_error_returns_400(ctx):
    ctx.remediation_manager = mock.MagicMock()
    coordinator_cls = mock.MagicMock()
    coordinator_cls.return_value.coordinate.side_effect = RuntimeError("conflict")
    with mock.patch("predictive_agent.coordinator.MultiAgentCoordinator", coordinator_cls):
        status, body = _json(_dispatch(
            "POST", "/coordinate",
            body=json.dumps({"pods": {"ns/a": "CRITICAL"}}).encode(),
        ))
    assert status == 400
    assert "conflict" in body.get("error", "")


# ── POST /review ────────────────────────────────────────────────────────────
def test_review_post_missing_manager_returns_503(ctx):
    status, body = _json(_dispatch(
        "POST", "/review", body=json.dumps({"id": "rev-1"}).encode(),
    ))
    assert status == 503
    assert "not initialized" in body.get("error", "").lower()


def test_review_post_malformed_json_returns_400(ctx):
    ctx.remediation_manager = mock.MagicMock()
    status, body = _json(_dispatch(
        "POST", "/review",
        body=b'{"signature": "ns/pod/act",',
        headers={"Content-Type": "application/json"},
    ))
    assert status == 400
    assert "Invalid JSON" in body.get("error", "")


def test_review_post_non_dict_body_returns_400(ctx):
    ctx.remediation_manager = mock.MagicMock()
    status, body = _json(_dispatch(
        "POST", "/review", body=b'["not", "an", "object"]',
        headers={"Content-Type": "application/json"},
    ))
    assert status == 400
    assert "JSON object" in body.get("error", "")


def test_review_post_missing_identifiers_returns_400(ctx):
    ctx.remediation_manager = mock.MagicMock()
    status, body = _json(_dispatch(
        "POST", "/review", body=json.dumps({"irrelevant": 1}).encode(),
    ))
    assert status == 400
    assert "signature" in body.get("error", "") or "id" in body.get("error", "")


def test_review_post_approval_success_returns_200(ctx):
    ctx.remediation_manager = mock.MagicMock()
    ctx.remediation_manager.approve_review.return_value = {"approved": True}
    ctx.remediation_manager.list_reviews.return_value = []
    status, body = _json(_dispatch(
        "POST", "/review",
        body=json.dumps({"signature": "ns/pod/pod_restart"}).encode(),
    ))
    assert status == 200
    assert body.get("status") == "ok"
    assert body.get("approved") is True
    assert body["pending"] == []
    ctx.remediation_manager.approve_review.assert_called_once_with(
        signature="ns/pod/pod_restart", review_id=""
    )


# ── GET /ready (collector degraded/down branches) ───────────────────────────
@pytest.mark.parametrize("collector_state", [COLLECTOR_STATE_DEGRADED, COLLECTOR_STATE_DOWN])
def test_ready_degraded_or_down_returns_503(ctx, collector_state):
    with mock.patch.object(server, "get_collector_state", return_value=collector_state):
        status, body = _json(_dispatch("GET", "/ready"))
    assert status == 503
    assert body.get("status") == "not_ready"
    assert body.get("collector_state") == collector_state


def test_ready_healthy_returns_200(ctx):
    with mock.patch.object(server, "get_collector_state", return_value=COLLECTOR_STATE_HEALTHY):
        status, body = _json(_dispatch("GET", "/ready"))
    assert status == 200
    assert body.get("status") == "ready"


# ── GET /state & /history & /cache ──────────────────────────────────────────
def test_state_with_model_returns_tracked_data(ctx):
    sm = mock.MagicMock()
    sm.pods = {"ns/pod": _ns_pod()}
    sm.to_dict.return_value = {"pods": {"ns/pod": {"state": "CRITICAL"}}}
    ctx.state_model = sm
    status, body = _json(_dispatch("GET", "/state"))
    assert status == 200
    assert body["total_pods_tracked"] == 1
    assert body["pods"]["ns/pod"]["state"] == "CRITICAL"


def test_history_nonempty_is_returned(ctx):
    ctx.history = [{"ts": "2026-01-01", "note": "spike"}]
    status, body = _dispatch("GET", "/history")
    assert status == 200
    assert json.loads(body) == [{"ts": "2026-01-01", "note": "spike"}]


def test_cache_none_returns_empty(ctx):
    ctx.cache = None
    status, body = _json(_dispatch("GET", "/cache"))
    assert status == 200
    assert body == {"cache": {}, "total": 0}


def test_cache_dict_returns_items(ctx):
    ctx.cache = {"k1": "v1"}
    status, body = _json(_dispatch("GET", "/cache"))
    assert status == 200
    assert body["cache"] == {"k1": "v1"}
    assert body["total"] == 1


def test_cache_unmappable_returns_empty(ctx):
    # A cache object whose dict() conversion blows up must degrade to empty.
    class BadCache:
        def __iter__(self):
            raise RuntimeError("not iterable")

    ctx.cache = BadCache()
    status, body = _json(_dispatch("GET", "/cache"))
    assert status == 200
    assert body == {"cache": {}, "total": 0}


# ── GET /remediate & /remediation-log ───────────────────────────────────────
def test_remediate_get_with_manager_returns_audit(ctx):
    ctx.remediation_manager = mock.MagicMock()
    ctx.remediation_manager.dry_run = True
    ctx.remediation_manager.risk_threshold = 0.65
    ctx.remediation_manager.get_stats.return_value = {"registered_actions": 3}
    ctx.remediation_manager.get_audit_trail.return_value = [{"action": "pod_restart"}]
    ctx.remediation_manager.safety_policy = SimpleNamespace(
        max_per_minute=2, max_per_hour=10, cooldown_seconds=60,
        protected_namespaces={"kube-system"},
    )
    status, body = _json(_dispatch("GET", "/remediate"))
    assert status == 200
    assert body.get("status") == "dry_run"
    assert body["registered_actions"] == 3
    assert body["safety_policy"]["max_per_minute"] == 2
    assert body["safety_policy"]["protected_namespaces"] == ["kube-system"]


def test_review_get_with_manager_returns_pending(ctx):
    ctx.remediation_manager = mock.MagicMock()
    ctx.remediation_manager.list_reviews.return_value = [{"id": "rev-1"}]
    ctx.remediation_manager.review_tier = "manual"
    ctx.remediation_manager.review_ttl_s = 300
    ctx.remediation_manager.reviews_deferred = 2
    status, body = _json(_dispatch("GET", "/review"))
    assert status == 200
    assert body.get("enabled") is True
    assert body.get("review_tier") == "manual"
    assert body.get("reviews_deferred") == 2
    assert body["pending"] == [{"id": "rev-1"}]


def test_remediation_log_get_without_manager_disabled(ctx):
    status, body = _json(_dispatch("GET", "/remediation-log"))
    assert status == 200
    assert body.get("status") == "disabled"
    assert body.get("remediation_log") == []
    assert body.get("total") == 0


def test_remediation_log_get_with_manager_returns_log(ctx):
    ctx.remediation_manager = mock.MagicMock()
    ctx.remediation_manager.dry_run = False
    ctx.remediation_manager.get_remediation_log.return_value = [{"action": "pod_restart", "outcome": "acted"}]
    status, body = _json(_dispatch("GET", "/remediation-log"))
    assert status == 200
    assert body.get("status") == "enabled"
    assert body["total"] == 1
    assert body["remediation_log"][0]["outcome"] == "acted"


# ── GET /plan strategies ────────────────────────────────────────────────────
def test_plan_get_without_manager_disabled(ctx):
    status, body = _json(_dispatch("GET", "/plan?state=CRITICAL"))
    assert status == 200
    assert body.get("status") == "disabled"


def test_plan_get_bad_horizon_falls_back_to_default(ctx):
    # Non-integer horizon on GET falls back to the default (3) instead of
    # crashing the handler.
    ctx.remediation_manager = mock.MagicMock()
    ctx.remediation_manager.plan_remediation.return_value = {"steps": []}
    status, body = _json(_dispatch("GET", "/plan?state=CRITICAL&horizon=not-an-int"))
    assert status == 200
    assert body.get("status") == "ok"
    ctx.remediation_manager.plan_remediation.assert_called_once_with("CRITICAL", 3)


def test_plan_get_refinement_failure_does_not_break_planning(ctx):
    # A backtester whose refinement raises must never break planning: the
    # handler swallows refinement errors and still produces a plan.
    ctx.remediation_manager = mock.MagicMock()
    ctx.remediation_manager.plan_remediation.return_value = {"steps": []}
    ctx.backtester = mock.Mock()
    ctx.remediation_manager.refine_from_backtester.side_effect = RuntimeError("refine failed")
    status, body = _json(_dispatch("GET", "/plan?state=CRITICAL"))
    assert status == 200
    assert body.get("status") == "ok"
    ctx.remediation_manager.refine_from_backtester.assert_called_once_with(ctx.backtester)


def test_plan_get_invalid_state_returns_400(ctx):
    ctx.remediation_manager = mock.MagicMock()
    status, body = _json(_dispatch("GET", "/plan?state=NOPE"))
    assert status == 400
    assert "invalid state" in body.get("error", "")


def test_plan_get_strategy_policy(ctx):
    ctx.remediation_manager = mock.MagicMock()
    ctx.remediation_manager.decision_model = mock.Mock()
    ctx.remediation_manager.failure_cost = 100.0
    ctx.remediation_manager.action_costs = {"pod_restart": 5.0}
    policy_cls = mock.MagicMock()
    policy_cls.return_value.plan.return_value.to_dict.return_value = {"steps": [{"action": "pod_restart"}]}
    with mock.patch.object(server, "ValueIterationPolicy", policy_cls):
        status, body = _json(_dispatch("GET", "/plan?state=CRITICAL&horizon=3&strategy=policy"))
    assert status == 200
    assert body.get("strategy") == "policy"
    assert body["plan"]["steps"][0]["action"] == "pod_restart"
    policy_cls.assert_called_once()


def test_plan_get_strategy_q(ctx):
    ctx.remediation_manager = mock.MagicMock()
    ctx.remediation_manager.decision_model = mock.Mock()
    qpolicy_cls = mock.MagicMock()
    qpolicy_cls.return_value.plan.return_value = (["pod_restart", "none"], 0.05)
    with mock.patch("predictive_agent.rl.QLearningPolicy", qpolicy_cls):
        status, body = _json(_dispatch("GET", "/plan?state=CRITICAL&horizon=2&strategy=q"))
    assert status == 200
    assert body.get("strategy") == "q"
    assert body["plan"]["steps"][0]["action"] == "pod_restart"


def test_plan_get_default_strategy_planner(ctx):
    ctx.remediation_manager = mock.MagicMock()
    ctx.remediation_manager.plan_remediation.return_value = {"steps": []}
    status, body = _json(_dispatch("GET", "/plan?state=CRITICAL&horizon=5"))
    assert status == 200
    assert body.get("strategy") == "planner"
    assert body["plan"] == {"steps": []}
    ctx.remediation_manager.plan_remediation.assert_called_once_with("CRITICAL", 5)


# ── GET /world-model ────────────────────────────────────────────────────────
def test_world_model_invalid_state_returns_400(ctx):
    status, body = _json(_dispatch("GET", "/world-model?state=NOPE&action=pod_restart"))
    assert status == 400
    assert "invalid state" in body.get("error", "")


def test_world_model_invalid_action_returns_400(ctx):
    status, body = _json(_dispatch("GET", "/world-model?state=CRITICAL&action=teleport"))
    assert status == 400
    assert "invalid action" in body.get("error", "")


def test_world_model_bad_horizon_falls_back_to_default(ctx):
    model_cls = mock.MagicMock()
    model_cls.return_value.what_if.return_value = {"steps": []}
    with mock.patch.object(server, "ActionAwareTransitionModel", model_cls):
        status, body = _json(_dispatch(
            "GET", "/world-model?state=CRITICAL&action=pod_restart&horizon=abc"
        ))
    assert status == 200
    assert body.get("status") == "ok"
    model_cls.return_value.what_if.assert_called_once_with("CRITICAL", "pod_restart", 3)


def test_world_model_valid_returns_what_if(ctx):
    model_cls = mock.MagicMock()
    model_cls.return_value.what_if.return_value = {"steps": [{"step": 1}]}
    with mock.patch.object(server, "ActionAwareTransitionModel", model_cls):
        status, body = _json(_dispatch("GET", "/world-model?state=CRITICAL&action=pod_restart&horizon=2"))
    assert status == 200
    assert body.get("status") == "ok"
    assert body["steps"] == [{"step": 1}]
    model_cls.return_value.what_if.assert_called_once_with("CRITICAL", "pod_restart", 2)


# ── GET /notifications & /backtest ──────────────────────────────────────────
def test_notifications_with_notifier_returns_history(ctx):
    ctx.notifier = mock.MagicMock()
    ctx.notifier.get_history.return_value = [{"level": "warning", "msg": "risk"}]
    status, body = _json(_dispatch("GET", "/notifications"))
    assert status == 200
    assert body.get("total") == 1
    assert body["notifications"][0]["msg"] == "risk"
    ctx.notifier.get_history.assert_called_once_with(limit=50)


def test_backtest_without_backtester_disabled(ctx):
    status, body = _json(_dispatch("GET", "/backtest"))
    assert status == 200
    assert body.get("status") == "disabled"


def test_backtest_success_returns_report(ctx):
    bt = mock.MagicMock()
    bt.evaluate.return_value.to_dict.return_value = {"accuracy": 0.93, "samples": 120}
    ctx.backtester = bt
    status, body = _json(_dispatch("GET", "/backtest"))
    assert status == 200
    assert body["accuracy"] == 0.93
    assert body["samples"] == 120
    bt.evaluate.assert_called_once()


def test_backtest_error_returns_error_payload(ctx):
    bt = mock.MagicMock()
    bt.evaluate.side_effect = RuntimeError("no data")
    ctx.backtester = bt
    status, body = _json(_dispatch("GET", "/backtest"))
    assert status == 200
    assert body.get("status") == "error"
    assert "no data" in body.get("error", "")


# ── GET /benchmark ──────────────────────────────────────────────────────────
def test_benchmark_without_manager_disabled(ctx):
    status, body = _json(_dispatch("GET", "/benchmark"))
    assert status == 200
    assert body.get("status") == "disabled"


def test_benchmark_synthetic_success(ctx):
    ctx.remediation_manager = mock.MagicMock()
    report = SimpleNamespace(
        scenarios=[SimpleNamespace(to_dict=lambda _s=s: {"state": _s}) for s in ["CRITICAL"]],
        summary=lambda: {"acc": 1.0},
    )
    with mock.patch.object(server, "run_benchmark", return_value=report):
        status, body = _json(_dispatch("GET", "/benchmark"))
    assert status == 200
    assert body.get("source") == "synthetic"
    assert body["summary"]["acc"] == 1.0
    assert body["scenarios"] == [{"state": "CRITICAL"}]


def test_benchmark_error_returns_500(ctx):
    ctx.remediation_manager = mock.MagicMock()
    with mock.patch.object(server, "run_benchmark", side_effect=RuntimeError("boom")):
        status, body = _json(_dispatch("GET", "/benchmark"))
    assert status == 500
    assert body.get("status") == "error"
    assert "boom" in body.get("error", "")


# ── GET /shadow ─────────────────────────────────────────────────────────────
def test_shadow_without_harness_disabled(ctx):
    status, body = _json(_dispatch("GET", "/shadow"))
    assert status == 200
    assert body.get("enabled") is False


def test_shadow_with_harness_returns_summary(ctx):
    harness = SimpleNamespace(summary=lambda: {"agreed": 10, "total": 12})
    with mock.patch.object(server.main, "_shadow_harness", harness):
        status, body = _json(_dispatch("GET", "/shadow"))
    assert status == 200
    assert body.get("enabled") is True
    assert body["summary"]["agreed"] == 10


def test_shadow_harness_error_returns_500(ctx):
    class BadHarness:
        def summary(self):
            raise RuntimeError("harness broken")

    with mock.patch.object(server.main, "_shadow_harness", BadHarness()):
        status, body = _json(_dispatch("GET", "/shadow"))
    assert status == 500
    assert body.get("enabled") is True
    assert "harness broken" in body.get("error", "")


# ── GET /calibration with a monitor ─────────────────────────────────────────
def test_calibration_with_monitor_returns_summary(ctx):
    rm = mock.MagicMock()
    rm.degraded = True
    rm.calibration_monitor = SimpleNamespace(summary=lambda: {"brier": 0.04})
    ctx.remediation_manager = rm
    status, body = _json(_dispatch("GET", "/calibration"))
    assert status == 200
    assert body.get("enabled") is True
    assert body.get("degraded") is True
    assert body["summary"]["brier"] == 0.04


def test_calibration_monitor_error_returns_500(ctx):
    class BadMonitor:
        def summary(self):
            raise RuntimeError("monitor down")

    rm = mock.MagicMock()
    rm.degraded = False
    rm.calibration_monitor = BadMonitor()
    ctx.remediation_manager = rm
    status, body = _json(_dispatch("GET", "/calibration"))
    assert status == 500
    assert body.get("enabled") is True
    assert "monitor down" in body.get("error", "")


# ── GET /decision-metrics & /cost-model ─────────────────────────────────────
def test_decision_metrics_returns_values(ctx):
    with mock.patch.object(
        server, "collect_decision_metrics",
        return_value=[("reconcile_total", "counter", "h", 7)],
    ):
        status, body = _json(_dispatch("GET", "/decision-metrics"))
    assert status == 200
    assert body == {"reconcile_total": 7}


def test_cost_model_get_without_manager_disabled(ctx):
    status, body = _json(_dispatch("GET", "/cost-model"))
    assert status == 200
    assert body.get("status") == "disabled"


def test_cost_model_get_with_manager_returns_costs(ctx):
    ctx.remediation_manager = mock.MagicMock()
    ctx.remediation_manager.cost_source = "explicit"
    ctx.remediation_manager.failure_cost = 250.0
    ctx.remediation_manager.action_costs = {"pod_restart": 8.0}
    status, body = _json(_dispatch("GET", "/cost-model"))
    assert status == 200
    assert body.get("source") == "explicit"
    assert body["failure_cost"] == 250.0
    assert body["action_costs"]["pod_restart"] == 8.0


# ── GET /policy/health & /policy/stats ──────────────────────────────────────
def test_policy_health_without_engine_disabled(ctx):
    status, body = _json(_dispatch("GET", "/policy/health"))
    assert status == 200
    assert body.get("status") == "disabled"


def test_policy_health_engine_ok(ctx):
    ctx.policy_engine = SimpleNamespace(check_health=lambda: True, _enabled=True)
    status, body = _json(_dispatch("GET", "/policy/health"))
    assert status == 200
    assert body["opa_healthy"] is True
    assert body["enabled"] is True


def test_policy_health_engine_error_returns_error(ctx):
    class BadEngine:
        def check_health(self):
            raise RuntimeError("opa down")

    ctx.policy_engine = BadEngine()
    status, body = _json(_dispatch("GET", "/policy/health"))
    assert status == 200
    assert body.get("status") == "error"
    assert "opa down" in body.get("error", "")


def test_policy_stats_with_engine_returns_stats(ctx):
    ctx.policy_engine = SimpleNamespace(get_stats=lambda: {"decisions": 42})
    status, body = _json(_dispatch("GET", "/policy/stats"))
    assert status == 200
    assert body["decisions"] == 42


def test_policy_stats_without_engine_disabled(ctx):
    status, body = _json(_dispatch("GET", "/policy/stats"))
    assert status == 200
    assert body.get("status") == "disabled"


# ── Sanity: uncovered GET branches never 500 even with a manager absent ─────
def test_get_remaining_endpoints_never_500_with_empty_context(ctx):
    # Every remaining read-only endpoint is safe even with a totally empty
    # context (no manager, no model, no engine): they must answer 2xx/4xx,
    # never raise into a 500.
    for path in (
        "/benchmark", "/review", "/shadow", "/calibration", "/decision-metrics",
        "/cost-model", "/policy/health", "/policy/stats", "/remediation-log",
        "/plan", "/world-model", "/notifications", "/backtest",
    ):
        status, body = _dispatch("GET", path)
        assert status in (200, 400, 404, 503), f"{path} -> {status}"
        assert isinstance(json.loads(body), dict)
