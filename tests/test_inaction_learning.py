"""A1: natural-incident learning via implicit 'none' outcomes.

When a pod worsens (HEALTHY/DEGRADED -> CRITICAL/FAILED) while the agent
executed no action, record the observed "none" outcome so the decision
model, calibration monitor and RL policy learn from real incidents even in
an intervention-light steady state (both the learning and calibration
counters were previously stuck at 0 because they only fed on executed
actions). See suggestion A1.
"""

from unittest import mock

from predictive_agent import config
from predictive_agent.calibration import CalibrationMonitor
from tests.test_calibration_guardrails import _build_manager
from tests.test_reconcile_phases import MockPolicy, _build_ctx, _healthy_pod

# ── Manager-level: note_inaction_outcome ────────────────────────────────────


class TestNoteInactionOutcome:
    def test_records_decision_model_transition_for_none(self):
        rem = _build_manager()
        spy = mock.MagicMock()
        rem.decision_model = spy
        assert rem.note_inaction_outcome("ns/pod", "HEALTHY", "FAILED", 0.8) is True
        spy.record_transition.assert_called_once_with("HEALTHY", "FAILED", "none")

    def test_records_calibration_sample_actual_failed(self):
        rem = _build_manager()
        mon = CalibrationMonitor(window=50, min_samples=1)
        rem.calibration_monitor = mon
        assert rem.note_inaction_outcome("ns/pod", "DEGRADED", "CRITICAL", 0.6) is True
        assert mon.n() == 1
        predicted, actual = mon._pairs[0]
        assert abs(predicted - 0.6) < 1e-9
        assert actual is True

    def test_updates_rl_policy(self):
        rem = _build_manager()
        rl = mock.MagicMock()
        rem.rl_policy = rl
        rem.note_inaction_outcome("ns/pod", "HEALTHY", "FAILED", 0.9)
        rl.update.assert_called_once_with("HEALTHY", "none", "FAILED")

    def test_returns_false_when_states_equal(self):
        rem = _build_manager()
        spy = mock.MagicMock()
        rem.decision_model = spy
        mon = CalibrationMonitor(window=50, min_samples=1)
        rem.calibration_monitor = mon
        assert rem.note_inaction_outcome("ns/pod", "CRITICAL", "CRITICAL", 0.8) is False
        spy.record_transition.assert_not_called()
        assert mon.n() == 0

    def test_returns_false_when_to_state_not_critical_or_failed(self):
        # HEALTHY -> DEGRADED is a worsening but not a "missed failure";
        # HEALTHY -> RECOVERED is an improvement. Neither is learned.
        rem = _build_manager()
        spy = mock.MagicMock()
        rem.decision_model = spy
        assert rem.note_inaction_outcome("ns/pod", "HEALTHY", "DEGRADED", 0.8) is False
        assert rem.note_inaction_outcome("ns/pod", "DEGRADED", "HEALTHY", 0.2) is False
        spy.record_transition.assert_not_called()

    def test_returns_false_when_from_state_terminal(self):
        # CRITICAL -> FAILED is beyond the "inaction miss" scope: re-learning
        # the terminal sequence from already-failed states would flood the
        # model on persistently-failing pods.
        rem = _build_manager()
        spy = mock.MagicMock()
        rem.decision_model = spy
        assert rem.note_inaction_outcome("ns/pod", "CRITICAL", "FAILED", 0.9) is False
        spy.record_transition.assert_not_called()

    def test_disabled_by_config_gate(self, monkeypatch):
        rem = _build_manager()
        spy = mock.MagicMock()
        rem.decision_model = spy
        monkeypatch.setattr(config, "LEARN_FROM_INACTION", False)
        assert rem.note_inaction_outcome("ns/pod", "HEALTHY", "FAILED", 0.8) is False
        spy.record_transition.assert_not_called()

    def test_calibration_skipped_without_predicted_p(self):
        rem = _build_manager()
        mon = CalibrationMonitor(window=50, min_samples=1)
        rem.calibration_monitor = mon
        # Transition still recorded; only the calibration sample needs a
        # predicted P(FAILED).
        assert rem.note_inaction_outcome("ns/pod", "HEALTHY", "FAILED", None) is True
        assert mon.n() == 0

    def test_unknown_target_is_safe(self):
        rem = _build_manager()
        assert rem.note_inaction_outcome("ghost/pod", "HEALTHY", "FAILED", 0.9) is True


# ── Phases wiring: evaluate_pods calls note_inaction_outcome ────────────────


class TestEvaluateWiresInactionLearning:
    def test_worsened_unacted_pod_teaches_none_outcome(self, monkeypatch):
        from predictive_agent import reconcile_phases as phases
        from predictive_agent import state_model as smod

        # Commit state changes on the first agreeing cycle so the two-cycle
        # HEALTHY -> CRITICAL sequence is observed within this test.
        monkeypatch.setattr(smod, "_STATE_HYSTERESIS", 1)
        StateModel = smod.StateModel

        cm = _build_manager()
        # Never act: planning/execution always yields an empty result set, so
        # any calibration sample must come from the inaction path.
        cm.plan_and_execute = mock.MagicMock(return_value=[])
        cm.evaluate = mock.MagicMock(return_value=[])
        mon = CalibrationMonitor(window=50, min_samples=1)
        cm.calibration_monitor = mon
        pred = mock.MagicMock()
        pred.risk_score = 0.9
        pred.ttf_minutes = 30
        pred.confidence = 0.8

        ctx = _build_ctx(
            state_model=StateModel(),
            remediation_manager=cm,
            predictor=mock.MagicMock(risk_threshold=0.5),
            policy_engine=MockPolicy(),
            get_node_pressure=lambda conds, node: (True, False),
            get_pod_memory_limit=lambda p, cn="": 1000,
        )
        ctx.predictor.predict = mock.MagicMock(return_value=pred)

        # Cycle 1: fresh pod, only node pressure -> DEGRADED. The manager's
        # scope guard rejects DEGRADED as a "miss", so nothing is learned yet.
        phases.evaluate_pods(ctx, 1, "kubernetes", [_healthy_pod()], {}, {}, set())
        assert mon.n() == 0

        # Cycle 2: memory+cpu saturation (200% of limit) + pressure resolves
        # the pod DEGRADED -> CRITICAL with no action executed.
        critical_metrics = {"default/p1": {"memory_mib": 2000, "cpu_m": 2000}}
        phases.evaluate_pods(
            ctx, 2, "kubernetes", [_healthy_pod()], critical_metrics, {}, set()
        )
        assert mon.n() >= 1
        for predicted, actual in mon._pairs:
            assert actual is True
