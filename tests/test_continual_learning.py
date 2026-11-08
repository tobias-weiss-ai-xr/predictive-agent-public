"""Tests for continual-learning hardening (#10).

Covers exponential forgetting (decay), data-support / confidence gating of the
planner, and drift-aware reweighting (stale transitions forgotten on detected
calibration drift).
"""

import pytest

from predictive_agent import config, main
from predictive_agent.pomdp import ActionAwareTransitionModel, MarkovChain
from predictive_agent.remediator import (
    RemediationAction,
    RemediationManager,
    RemediationResult,
)
from predictive_agent.state_model import StateModel

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


class _Pod:
    state = "CRITICAL"
    name = "ns/pod"
    namespace = "ns"
    node_name = "n"
    pod_phase = "Running"
    cpu_trend = 0.0
    memory_trend = 0.0
    memory_pct = 0
    cpu_pct = 0
    memory_anomaly_score = 0
    cpu_anomaly_score = 0


class _Pred:
    risk_score = 0.9
    confidence = 0.9


def _build_model():
    model = ActionAwareTransitionModel(base_chain=MarkovChain())
    for _ in range(40):
        model.record_transition("CRITICAL", "RECOVERED", "pod_restart")
        model.record_transition("CRITICAL", "FAILED", "none")
    return model


def _build_manager():
    rem = RemediationManager(dry_run=False)
    for name in ACTION_NAMES:
        rem.register_action(_FakeAction(name))
    rem.decision_model = _build_model()
    return rem


def test_decay_reduces_support():
    model = _build_model()
    before = model.total_support()
    removed = model.decay(0.5)
    after = model.total_support()
    assert before == 80.0  # 2 actions x 40
    assert after < before
    assert removed > 0
    # Decaying again keeps shrinking.
    model.decay(0.5)
    assert model.total_support() < after


def test_decay_disabled_at_one():
    model = _build_model()
    before = model.total_support()
    assert model.decay(1.0) == 0.0
    assert model.total_support() == before


def test_state_support_and_confidence():
    model = _build_model()
    assert model.state_support("CRITICAL") == 80.0
    assert model.is_confident("CRITICAL", 80.0) is True
    assert model.is_confident("CRITICAL", 81.0) is False
    assert model.is_confident("CRITICAL", 0.0) is True  # gate disabled


def test_confidence_gate_falls_back_when_support_low(monkeypatch):
    from predictive_agent.pomdp import Planner

    rem = _build_manager()
    monkeypatch.setattr(config, "MIN_MODEL_SUPPORT", 1000.0)
    called = {"planner": False}
    monkeypatch.setattr(
        Planner, "plan",
        lambda self, *a, **k: (_ for _ in ()).throw(AssertionError("planner used")),
    )
    rem.plan_and_execute("CRITICAL", "ns/pod", _Pod(), _Pred(), 0.9, horizon=3)
    assert called["planner"] is False  # fell back to reactive, never planned


def test_confidence_gate_uses_planner_when_disabled(monkeypatch):
    from predictive_agent.pomdp import Planner

    rem = _build_manager()
    monkeypatch.setattr(config, "MIN_MODEL_SUPPORT", 0.0)
    monkeypatch.setattr(
        Planner, "plan",
        lambda self, *a, **k: (_ for _ in ()).throw(AssertionError("planner used")),
    )
    # With the gate disabled the planner path is taken -> the patched plan raises.
    with pytest.raises(AssertionError):
        rem.plan_and_execute("CRITICAL", "ns/pod", _Pod(), _Pred(), 0.9, horizon=3)


class _FakeMonitor:
    def drift_detected(self):
        return True

    def brier(self):
        return 0.5

    def over_rate(self):
        return 0.6

    def status(self):
        return "drift"


def test_drift_triggers_model_decay(monkeypatch):
    model = _build_model()
    rem = RemediationManager(dry_run=False)
    rem.decision_model = model
    rem.calibration_monitor = _FakeMonitor()
    sm = StateModel()

    saved = {
        k: getattr(main, k, None)
        for k in (
            "_state_model", "_predictor", "_remediation_manager", "_notifier",
            "_llm_analyzer", "_backtester", "_kg_client", "_state_store",
        )
    }
    main._state_model = sm
    main._predictor = None
    main._remediation_manager = rem
    main._notifier = None
    main._llm_analyzer = None
    main._backtester = None
    main._kg_client = None
    main._state_store = None
    monkeypatch.setattr(main, "run_cmd", lambda *a, **k: (0, '{"items": []}', ""))
    monkeypatch.setattr(main, "detect_runtime", lambda: "kubernetes")

    before = model.total_support()
    main.reconcile()
    after = model.total_support()

    for k, v in saved.items():
        setattr(main, k, v)

    assert after < before  # drift-aware reweighting decayed stale transitions
