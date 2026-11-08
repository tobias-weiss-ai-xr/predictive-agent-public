"""Tests for the shadow-mode validation harness (#1).

The harness runs the planner observe-only next to production: it logs planner
vs executed disagreements and predicted-vs-actual calibration without
executing anything or mutating the decision model.
"""

import urllib.request

from predictive_agent import main
from predictive_agent.markov import MarkovChain
from predictive_agent.pomdp import ActionAwareTransitionModel
from predictive_agent.remediator import (
    RemediationAction,
    RemediationManager,
    RemediationResult,
)
from predictive_agent.server import _context, start_server
from predictive_agent.shadow import ShadowModeHarness

ACTION_NAMES = [
    "pod_restart",
    "rollout_restart",
    "node_cordon",
    "scale",
    "right_size",
    "tune_resources",
]


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
            message="ok (fake)",
        )


def _build_manager():
    rem = RemediationManager(dry_run=False)
    for name in ACTION_NAMES:
        rem.register_action(_FakeAction(name))
    model = ActionAwareTransitionModel(base_chain=MarkovChain())
    for _ in range(40):
        model.record_transition("CRITICAL", "RECOVERED", "pod_restart")
        model.record_transition("CRITICAL", "FAILED", "none")
        model.record_transition("STRESSED", "RECOVERED", "scale")
        model.record_transition("STRESSED", "FAILED", "none")
    rem.decision_model = model
    return rem


def test_shadow_detects_planner_executed_disagreement():
    rem = _build_manager()
    harness = ShadowModeHarness(rem)

    # The planner would recommend pod_restart from CRITICAL; production ran scale.
    planned = harness.recommend("CRITICAL")
    assert "pod_restart" in planned

    harness.observe("ns/pod", "CRITICAL", ["scale"], risk_score=0.9)
    summary = harness.summary()
    assert summary["disagreements"] == 1
    assert summary["agreements"] == 0
    assert summary["recent"][0]["planned"] == planned
    assert summary["recent"][0]["executed"] == ["scale"]


def test_shadow_records_agreement():
    rem = _build_manager()
    harness = ShadowModeHarness(rem)
    planned = harness.recommend("CRITICAL")
    harness.observe("ns/pod", "CRITICAL", planned, risk_score=0.9)
    s = harness.summary()
    assert s["agreements"] == 1
    assert s["disagreements"] == 0


def test_shadow_recommend_does_not_execute():
    """The harness only plans; it must never act on the cluster."""
    rem = _build_manager()
    harness = ShadowModeHarness(rem)
    harness.observe("ns/pod", "CRITICAL", ["none"], risk_score=0.9)
    # No execution side effects from shadow observation.
    assert "ns/pod" not in rem._pending_feedback
    assert len(rem.audit_trail) == 0


def test_shadow_calibration_logs_over_under():
    rem = _build_manager()
    harness = ShadowModeHarness(rem)
    # Predicted high failure but pod actually fine -> over-prediction.
    harness.calibration("ns/a", 0.8, actual_failed=False)
    # Predicted low failure but pod actually failed -> under-prediction.
    harness.calibration("ns/b", 0.2, actual_failed=True)
    # Accurate.
    harness.calibration("ns/c", 0.9, actual_failed=True)

    s = harness.summary()
    assert s["over_predictions"] == 1
    assert s["under_predictions"] == 1
    assert len(s["calibrations_recent"]) == 3


def test_shadow_endpoint_serves_summary():
    rem = _build_manager()
    harness = ShadowModeHarness(rem)
    harness.observe("ns/pod", "CRITICAL", ["scale"], risk_score=0.9)

    ports = (18160, 18161)
    srv = start_server(*ports)
    _context.remediation_manager = rem
    saved_harness = getattr(main, "_shadow_harness", None)
    main._shadow_harness = harness
    try:
        with urllib.request.urlopen(f"http://localhost:{ports[0]}/shadow") as resp:
            status = resp.status
            data = __import__("json").loads(resp.read().decode())
    finally:
        main._shadow_harness = saved_harness
        _context.remediation_manager = None
        srv.shutdown()
        srv.server_close()

    assert status == 200
    assert data["enabled"] is True
    assert data["summary"]["disagreements"] == 1
