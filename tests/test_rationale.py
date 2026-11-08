"""Tests for operator-facing natural-language rationale (#6).

The planner should explain *why* each action is (or isn't) chosen in
human-readable form, surfaced both via the /plan endpoint and in the operator
`plan` notification sent during reconcile.
"""

import json
import urllib.request

import pytest

from predictive_agent import config, main
from predictive_agent.pomdp import ActionAwareTransitionModel, MarkovChain, Planner
from predictive_agent.remediator import (
    RemediationAction,
    RemediationManager,
    RemediationResult,
)
from predictive_agent.server import _context, start_server
from predictive_agent.state_model import PodTracker, StateModel

ACTION_NAMES = [
    "pod_restart", "rollout_restart", "node_cordon",
    "scale", "right_size", "tune_resources",
]


class _FakeAction(RemediationAction):
    def __init__(self, name):
        self.name = name

    def should_execute(self, pod_state, prediction, risk_score):
        return True

    def execute(self, target, context):
        return RemediationResult(action=self.name, target=target, success=True,
                                  dry_run=context.dry_run, message="ok (fake)")


class _FakeResult:
    risk_score = 0.95
    ttf_minutes = 5.0
    confidence = 0.9
    markov_state = "CRITICAL"
    memory_trend = 0.0
    cpu_trend = 0.0
    memory_pct = 0.0
    cpu_pct = 0.0
    pod_key = "ns/pod"


class _FakePredictor:
    risk_threshold = 0.7
    _predictions = {}

    def predict(self, **kwargs):
        r = _FakeResult()
        r.pod_key = kwargs.get("pod_key", "ns/pod")
        return r


class _FakeNotifier:
    def __init__(self):
        self.calls = []

    def notify(self, **kwargs):
        self.calls.append(kwargs)

    def get_history(self, limit=50):
        return self.calls[-limit:]


def _build_model():
    model = ActionAwareTransitionModel(base_chain=MarkovChain())
    for _ in range(40):
        model.record_transition("CRITICAL", "RECOVERED", "pod_restart")
        model.record_transition("CRITICAL", "FAILED", "none")
        model.record_transition("STRESSED", "RECOVERED", "scale")
        model.record_transition("STRESSED", "FAILED", "none")
    return model


def _build_pod(mode):
    restart = 10 if mode == "critical" else 0
    return {
        "metadata": {"namespace": "ns", "name": "pod",
                     "creationTimestamp": __import__("datetime").datetime.now(
                         __import__("datetime").timezone.utc).isoformat(),
                     "ownerReferences": []},
        "spec": {"nodeName": "node1",
                 "containers": [{"name": "c", "resources": {"limits": {"memory": "1Gi"}}}]},
        "status": {"phase": "Running",
                   "containerStatuses": [{"ready": True, "restartCount": restart}]},
    }


def _make_run_cmd(holder):
    def fake_run_cmd(cmd, timeout=10):
        c = list(cmd)
        if c[:3] == ["kubectl", "top", "pods"]:
            top = "ns pod 960m 1000Mi" if holder["mode"] == "critical" else "ns pod 50m 200Mi"
            return (0, "NAMESPACE NAME CPU MEMORY\n" + top, "")
        if c[:3] == ["kubectl", "top", "nodes"]:
            return (0, "NAME CPU MEMORY CPU% MEM%\nnode1 500m 2000Mi 50% 50%", "")
        if c[:3] == ["kubectl", "get", "nodes"]:
            return (1, "", "")
        if c[:3] == ["kubectl", "get", "pods"]:
            return (0, json.dumps({"items": [_build_pod(holder["mode"])]}), "")
        if c[:2] == ["kubectl", "logs"]:
            return (1, "", "")
        if c[:2] == ["kubectl", "version"]:
            return (0, "Client Version v1.29.0", "")
        return (0, "", "")
    return fake_run_cmd


@pytest.fixture
def rationale_env(monkeypatch):
    holder = {"mode": "critical"}
    monkeypatch.setattr(main, "run_cmd", _make_run_cmd(holder))
    monkeypatch.setattr(main, "detect_runtime", lambda: "kubernetes")
    monkeypatch.setattr(config, "PLANNING_ENABLED", True)
    monkeypatch.setattr(config, "PLANNING_HORIZON", 3)
    monkeypatch.setattr(config, "WATCH_NAMESPACES", [])
    monkeypatch.setattr(config, "SKIP_NAMESPACES", [])
    sm = StateModel()
    sm.track_pod = lambda ns, n: sm.pods.setdefault(
        f"{ns}/{n}", PodTracker(ns, n, hysteresis=1))
    rem = RemediationManager(dry_run=False)
    for name in ACTION_NAMES:
        rem.register_action(_FakeAction(name))
    rem.decision_model = _build_model()
    notifier = _FakeNotifier()
    predictor = _FakePredictor()
    saved = {k: getattr(main, k, None) for k in (
        "_state_model", "_predictor", "_remediation_manager", "_notifier",
        "_llm_analyzer", "_backtester", "_kg_client", "_state_store", "_shadow_harness",
        "_last_plan_notify",
    )}
    main._state_model = sm
    main._predictor = predictor
    main._remediation_manager = rem
    main._notifier = notifier
    main._llm_analyzer = None
    main._backtester = None
    main._kg_client = None
    main._state_store = None
    main._shadow_harness = None
    main._last_plan_notify = {}
    yield rem, sm, notifier, holder
    for k, v in saved.items():
        setattr(main, k, v)


def test_plan_steps_carry_rationale():
    planner = Planner(model=_build_model())
    plan = planner.plan("CRITICAL", 3)
    for step in plan.steps:
        assert step.rationale, "every step must explain itself"
        # Numbers should be present (percentages / costs).
        assert "%" in step.rationale
        if step.action != "none":
            assert step.action in step.rationale


def test_plan_to_dict_includes_rationale():
    planner = Planner(model=_build_model())
    data = planner.plan("CRITICAL", 3).to_dict()
    for s in data["steps"]:
        assert "rationale" in s and s["rationale"]


def test_plan_endpoint_includes_rationale():
    rem = RemediationManager(dry_run=False)
    for name in ACTION_NAMES:
        rem.register_action(_FakeAction(name))
    rem.decision_model = _build_model()

    ports = (18200, 18201)
    srv = start_server(*ports)
    _context.remediation_manager = rem
    try:
        with urllib.request.urlopen(f"http://localhost:{ports[0]}/plan?state=CRITICAL&horizon=3") as resp:
            status = resp.status
            data = json.loads(resp.read().decode())
    finally:
        _context.remediation_manager = None
        srv.shutdown()
        srv.server_close()

    assert status == 200
    for s in data["plan"]["steps"]:
        assert s["rationale"]
        assert "%" in s["rationale"]


def test_reconcile_notification_includes_rationale(rationale_env):
    rem, sm, notifier, holder = rationale_env
    holder["mode"] = "critical"
    main.reconcile()
    plan_calls = [c for c in notifier.calls if c.get("alert_type") == "plan"]
    assert plan_calls, "a plan notification should have been sent"
    details = plan_calls[0]["details"]
    # The operator-facing notification must carry the NL rationale, not just actions.
    assert "apply" in details or "Step" in details
    assert "%" in details
