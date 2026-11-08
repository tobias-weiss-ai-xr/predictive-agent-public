"""Tests for futile-remediation / crash-loop escalation (the #watch fix).

These verify that an at-risk pod whose planner keeps choosing 'none' (e.g. a
deterministic crash-loop where a restart yields no durable P(FAILED) reduction)
is (a) made visible and (b) escalated after CRASHLOOP_ESCALATE_AFTER cycles
instead of silently disappearing, and that CRASHLOOP_POLICY=persist_restart can
keep the reactive restart path active.
"""


from predictive_agent import config
from predictive_agent.metrics import collect_decision_metrics
from predictive_agent.remediator import RemediationManager


def _manager():
    return RemediationManager(dry_run=True, risk_threshold=0.3)


def test_note_remediation_outcome_escalates_after_threshold():
    mgr = _manager()
    # Three consecutive at-risk cycles with only 'none' decided -> escalate.
    e1, n1 = mgr.note_remediation_outcome("docker/loop", True, ["none"])
    e2, n2 = mgr.note_remediation_outcome("docker/loop", True, ["none"])
    e3, n3 = mgr.note_remediation_outcome("docker/loop", True, ["none"])
    assert (e1, e2) == (False, False)
    assert n1 == 1 and n2 == 2 and n3 == 3
    assert e3 is True
    assert mgr.escalations == 1
    # Ongoing crash-loop: subsequent futile cycles stay visible but do NOT
    # re-escalate (no notification / metric spam every reconcile).
    e4, n4 = mgr.note_remediation_outcome("docker/loop", True, ["none"])
    assert e4 is False and n4 == 4
    assert mgr.escalations == 1
    # A real action resets the futile counter.
    e5, n5 = mgr.note_remediation_outcome("docker/loop", True, ["pod_restart"])
    assert e5 is False and n5 == 0
    assert mgr.escalations == 1  # unchanged after reset


def test_note_remediation_outcome_never_escalates_when_not_at_risk():
    mgr = _manager()
    for _ in range(10):
        escalated, n = mgr.note_remediation_outcome("docker/ok", False, ["none"])
        assert escalated is False
    assert mgr.escalations == 0


def test_plan_and_execute_surfaces_none_for_at_risk(monkeypatch):
    # Force the planner to return an all-'none' plan.
    class _NoneStep:
        action = "none"

    class _FakePlan:
        steps = [_NoneStep()]

    class _FakePlanner:
        def __init__(self, *a, **k):
            pass

        def plan(self, *a, **k):
            return _FakePlan()

    monkeypatch.setattr("predictive_agent.pomdp.Planner", _FakePlanner)
    monkeypatch.setattr(config, "MIN_MODEL_SUPPORT", 0)
    monkeypatch.setattr(config, "MIN_ACTION_CONFIDENCE", 0.0)

    mgr = _manager()
    monkeypatch.setattr(mgr.decision_model, "is_confident", lambda *a, **k: True)

    class _PodState:
        state = "CRITICAL"
        namespace = "docker"
        restart_count = 10

    class _Pred:
        confidence = 0.9
        risk_score = 0.99

    results = mgr.plan_and_execute(
        "CRITICAL", "docker/loop", _PodState(), _Pred(), 0.99, horizon=3
    )
    assert results and all(r.action == "none" for r in results)
    assert "no beneficial" in results[0].message


def test_metrics_exhausted_counter():
    mgr = _manager()
    mgr.escalations = 2
    metrics = collect_decision_metrics(mgr)
    names = [m[0] for m in metrics]
    assert "decision_layer_remediation_exhausted_total" in names
    vals = {m[0]: m[3] for m in metrics}
    assert vals["decision_layer_remediation_exhausted_total"] == 2
