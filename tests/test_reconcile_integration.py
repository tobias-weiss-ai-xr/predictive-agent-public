"""Reconcile-level end-to-end test for the POMDP decision layer.

Drives the *real* ``predictive_agent.main.reconcile()`` with a patched
``run_cmd`` (synthetic Kubernetes output) so no cluster is required. This
exercises the production wiring that unit tests bypass:

  predict -> (at-risk) plan_remediation + notify -> plan_and_execute (executes)
  -> _record_pending  ->  next cycle: record_observations -> feed_transition
  -> decision model learns  ->  plan surfaced in server context.

It proves the closed learning loop actually closes through the reconcile path,
not just via direct method calls.
"""

import json
from datetime import datetime, timezone

import pytest

from predictive_agent import config, main
from predictive_agent.markov import MarkovChain
from predictive_agent.pomdp import ActionAwareTransitionModel
from predictive_agent.remediator import (
    RemediationAction,
    RemediationManager,
    RemediationResult,
)
from predictive_agent.server import _context as _server_context
from predictive_agent.state_model import PodTracker, StateModel

ACTION_NAMES = [
    "pod_restart",
    "rollout_restart",
    "node_cordon",
    "scale",
    "right_size",
    "tune_resources",
]
POD_KEY = "ns/pod"
FI = MarkovChain.STATE_IDX


class _FakeAction(RemediationAction):
    def __init__(self, name):
        self.name = name

    def should_execute(self, pod_state, prediction, risk_score):
        return True

    def execute(self, target, context):
        return RemediationResult(
            action=self.name,
            target=target,
            success=True,
            dry_run=context.dry_run,
            message="ok (fake, no kubectl)",
        )


class _FakeResult:
    risk_score = 0.95
    ttf_minutes = 5.0
    confidence = 0.9
    markov_state = "CRITICAL"
    memory_trend = 0.0
    cpu_trend = 0.0
    memory_pct = 0.0
    cpu_pct = 0.0
    pod_key = POD_KEY


class _FakePredictor:
    risk_threshold = 0.7
    _predictions = {}

    def predict(self, **kwargs):
        r = _FakeResult()
        r.pod_key = kwargs.get("pod_key", POD_KEY)
        return r


class _FakeNotifier:
    def __init__(self):
        self.calls = []

    def notify(self, **kwargs):
        self.calls.append(kwargs)

    def get_history(self, limit=50):
        return self.calls[-limit:]


def _build_pod(mode):
    """A pod JSON that classifies CRITICAL (high mem/cpu/restart) or not."""
    restart = 10 if mode == "critical" else 0
    return {
        "metadata": {
            "namespace": "ns",
            "name": "pod",
            "creationTimestamp": datetime.now(timezone.utc).isoformat(),
            "ownerReferences": [],
        },
        "spec": {
            "nodeName": "node1",
            "containers": [{"name": "c", "resources": {"limits": {"memory": "1Gi"}}}],
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [{"ready": True, "restartCount": restart}],
        },
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
            return (1, "", "")  # skip node conditions -> no node pressure
        if c[:3] == ["kubectl", "get", "pods"]:
            return (0, json.dumps({"items": [_build_pod(holder["mode"])]}), "")
        if c[:2] == ["kubectl", "logs"]:
            return (1, "", "")  # skip log analysis
        if c[:2] == ["kubectl", "version"]:
            return (0, "Client Version v1.29.0", "")
        return (0, "", "")

    return fake_run_cmd


@pytest.fixture
def reconcile_env(monkeypatch):
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
    model = ActionAwareTransitionModel(base_chain=MarkovChain())
    # Seed learning: pod_restart recovers CRITICAL; scale recovers STRESSED.
    for _ in range(40):
        model.record_transition("CRITICAL", "RECOVERED", "pod_restart")
        model.record_transition("CRITICAL", "FAILED", "none")
        model.record_transition("STRESSED", "RECOVERED", "scale")
        model.record_transition("STRESSED", "FAILED", "none")
    rem.decision_model = model
    notifier = _FakeNotifier()
    predictor = _FakePredictor()

    saved = {
        k: getattr(main, k, None)
        for k in (
            "_state_model",
            "_predictor",
            "_remediation_manager",
            "_notifier",
            "_llm_analyzer",
            "_backtester",
            "_kg_client",
            "_state_store",
        )
    }
    main._state_model = sm
    main._predictor = predictor
    main._remediation_manager = rem
    main._notifier = notifier
    main._llm_analyzer = None
    main._backtester = None
    main._kg_client = None
    main._state_store = None

    yield rem, sm, notifier, holder

    for k, v in saved.items():
        setattr(main, k, v)


def test_reconcile_closes_learning_loop(reconcile_env):
    rem, sm, notifier, holder = reconcile_env

    # ── Reconcile #1: a CRITICAL pod ──────────────────────────────────
    holder["mode"] = "critical"
    res1 = main.reconcile()
    assert res1["at_risk_count"] == 1, res1
    assert res1["pods_tracked"] == 1, res1

    # The planner chose an action, executed it, and surfaced the plan.
    plan = _server_context.remediation_plans.get(POD_KEY)
    assert plan is not None and any(s["action"] != "none" for s in plan["steps"])
    # Execution recorded pending feedback for next cycle's observation.
    assert POD_KEY in rem._pending_feedback

    # No CRITICAL -> {DEGRADED,HEALTHY,STRESSED} transition learned yet.
    row = rem.decision_model.action_counts["pod_restart"][FI["CRITICAL"]]
    assert row[FI["DEGRADED"]] == 0 and row[FI["HEALTHY"]] == 0 and row[FI["STRESSED"]] == 0

    # ── Reconcile #2: the pod recovers (low mem/cpu, no restarts) ─────
    holder["mode"] = "healthy"
    main.reconcile()
    observed = sm.pods[POD_KEY].state

    # The production learning loop paired the pending (CRITICAL, pod_restart)
    # with this cycle's observed state and taught the decision model.
    assert (
        rem.decision_model.action_counts["pod_restart"][FI["CRITICAL"]][FI[observed]] >= 1
    ), f"loop did not learn CRITICAL->pod_restart->{observed}"

    # The plan was surfaced again for the (now recovering) pod.
    assert _server_context.remediation_plans.get(POD_KEY) is not None
