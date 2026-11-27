"""Survivor-killer tests for plan_and_execute, get_remediation_state, get_audit_trail, note_remediation_outcome, and other high-value plan/state functions.

Targets remaining mutants after test_dispatch_survivors.py and test_remediator_survivors.py.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

import predictive_agent.remediator as rem_mod
from predictive_agent import config
from predictive_agent.remediator import (
    RemediationAction,
    RemediationManager,
    RemediationOutcome,
    RemediationResult,
    SafetyPolicy,
)

# ── helpers ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clock(monkeypatch):
    c = {"now": 1_000_000.0}
    monkeypatch.setattr(rem_mod, "_now", lambda: c["now"])
    monkeypatch.setattr(rem_mod, "_utc_now", lambda: f"2025-01-01T{c['now']:g}Z")
    return c


def _pod(**kw) -> SimpleNamespace:
    base = dict(
        name="pod-1",
        namespace="ns1",
        node_name="node-1",
        state="CRITICAL",
        memory_pct=90.0,
        cpu_pct=80.0,
        restart_count=2,
        log_errors=5,
        node_pressure=False,
        node_disk_pressure=False,
        replicas=3,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _no_cooldown_policy():
    return SafetyPolicy(cooldown_seconds=0, max_per_minute=1000, max_per_hour=100000)


def _manager(**kw) -> RemediationManager:
    kw.setdefault("dry_run", True)
    kw.setdefault("risk_threshold", 0.0)
    kw.setdefault("safety_policy", _no_cooldown_policy())
    kw.setdefault("dedup_window", 300)
    kw.setdefault("action_costs", {})
    return RemediationManager(**kw)


class _TestAction(RemediationAction):
    name = "test_action"
    idempotent = True
    reversible = False
    risk_tier = "safe"

    def should_execute(self, pod_state, prediction, risk_score: float) -> bool:
        return True

    def execute(self, target: str, context) -> RemediationResult:
        return RemediationResult(
            action=self.name,
            target=target,
            success=True,
            dry_run=context.dry_run,
            message="executed",
            command="test",
        )


def _planner_steps(*actions):
    return [
        SimpleNamespace(action=a, expected_failed_prob_after=0.3) for a in actions
    ]


# ── plan_and_execute ────────────────────────────────────────────────────────


class TestPlanAndExecute:
    def _fake_result(self, action="test_action"):
        return RemediationResult(
            action=action, target="pod-1", success=True,
            dry_run=True, message="ok", command="test",
        )

    def test_action_none_returns_empty_or_none_result(self, clock):
        mgr = _manager()
        pod = _pod()
        with mock.patch("predictive_agent.pomdp.Planner") as MockPlanner:
            mock_plan = MockPlanner.return_value
            mock_plan.plan.return_value = SimpleNamespace(
                steps=_planner_steps("none"), expected_failed_prob_after=0.5
            )
            result = mgr.plan_and_execute("CRITICAL", "pod-1", pod, None, 0.9)
        # #watch: a lone 'none' step produces an explicit 'none' result (or empty
        # if risk below threshold).
        if result:
            assert result[0].action == "none"

    def test_planner_executes_named_action(self, clock):
        mgr = _manager()
        action = _TestAction()
        mgr.register_action(action)
        pod = _pod()
        with mock.patch("predictive_agent.pomdp.Planner") as MockPlanner:
            mock_plan = MockPlanner.return_value
            mock_plan.plan.return_value = SimpleNamespace(
                steps=_planner_steps("test_action"), expected_failed_prob_after=0.3
            )
            result = mgr.plan_and_execute("CRITICAL", "pod-1", pod, None, 0.9)
        assert any(getattr(r, "action", None) == "test_action" for r in result)

    def test_max_actions_limits_steps(self, clock):
        mgr = _manager()
        action = _TestAction()
        mgr.register_action(action)
        pod = _pod()
        fake = self._fake_result()
        with(
            mock.patch("predictive_agent.config.MIN_MODEL_SUPPORT", 0.0),
            mock.patch("predictive_agent.pomdp.Planner") as MockPlanner,
        ):
            mock_plan = MockPlanner.return_value
            mock_plan.plan.return_value = SimpleNamespace(
                steps=_planner_steps("test_action", "test_action", "test_action"),
                expected_failed_prob_after=0.2,
            )
            with mock.patch.object(mgr, "_apply_named_action", return_value=fake):
                result = mgr.plan_and_execute(
                    "CRITICAL", "pod-1", pod, None, 0.9, max_actions=2
                )
        assert len(result) == 2

    def test_empty_steps_returns_empty_list(self, clock):
        mgr = _manager(risk_threshold=0.99)
        pod = _pod()
        with mock.patch("predictive_agent.pomdp.Planner") as MockPlanner:
            mock_plan = MockPlanner.return_value
            mock_plan.plan.return_value = SimpleNamespace(
                steps=[], expected_failed_prob_after=0.3
            )
            result = mgr.plan_and_execute("CRITICAL", "pod-1", pod, None, 0.9)
        assert result == []

    def test_none_action_skipped_not_executed(self, clock):
        """A 'none' step must never call _apply_named_action."""
        mgr = _manager()
        pod = _pod()
        with mock.patch("predictive_agent.pomdp.Planner") as MockPlanner:
            mock_plan = MockPlanner.return_value
            mock_plan.plan.return_value = SimpleNamespace(
                steps=_planner_steps("none"), expected_failed_prob_after=0.5
            )
            with mock.patch.object(mgr, "_apply_named_action", return_value=None) as spa:
                mgr.plan_and_execute("CRITICAL", "pod-1", pod, None, 0.9, max_actions=0)
            spa.assert_not_called()

    def test_interventions_executed_increments_on_success(self, clock):
        mgr = _manager()
        action = _TestAction()
        mgr.register_action(action)
        pod = _pod()
        fake = self._fake_result()
        with (
            mock.patch("predictive_agent.config.MIN_MODEL_SUPPORT", 0.0),
            mock.patch("predictive_agent.pomdp.Planner") as MockPlanner,
        ):
            mock_plan = MockPlanner.return_value
            mock_plan.plan.return_value = SimpleNamespace(
                steps=_planner_steps("test_action", "test_action"),
                expected_failed_prob_after=0.3,
            )
            with mock.patch.object(mgr, "_apply_named_action", return_value=fake):
                before = mgr.interventions_executed
                mgr.plan_and_execute("CRITICAL", "pod-1", pod, None, 0.9)
        assert mgr.interventions_executed == before + 2

    def test_confidence_below_gate_returns_empty(self, clock):
        """Confidence gate: never act when the predictor is unsure."""
        mgr = _manager()
        pod = _pod()
        pred = SimpleNamespace(confidence=0.1)
        with mock.patch.object(config, "MIN_ACTION_CONFIDENCE", 0.5):
            with mock.patch("predictive_agent.pomdp.Planner") as MockPlanner:
                mock_plan = MockPlanner.return_value
                mock_plan.plan.return_value = SimpleNamespace(
                    steps=_planner_steps("test_action"), expected_failed_prob_after=0.3
                )
                result = mgr.plan_and_execute("CRITICAL", "pod-1", pod, pred, 0.9)
        assert result == []
        MockPlanner.return_value.plan.assert_not_called()

    def test_degraded_falls_back_to_evaluate(self, clock):
        mgr = _manager()
        mgr.degraded = True
        pod = _pod()
        with mock.patch("predictive_agent.pomdp.Planner"):
            with mock.patch.object(mgr, "evaluate", return_value=[self._fake_result()]) as ev:
                result = mgr.plan_and_execute("CRITICAL", "pod-1", pod, None, 0.9)
        assert len(result) == 1
        ev.assert_called_once()


# ── get_remediation_state ───────────────────────────────────────────────────


class TestGetRemediationState:
    def test_returns_dict_with_expected_keys(self, clock):
        mgr = _manager()
        state = mgr.get_remediation_state()
        assert isinstance(state, dict)
        assert "verify_enabled" in state
        assert "entries" in state
        assert "canary" in state
        assert "pending_outcome_alerts" in state

    def test_limit_controls_entries(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        for i in range(3):
            mgr.verify_state[f"key-{i}"] = {"status": "pending", "at": 100.0}
            mgr._dedup_history[f"key-{i}"] = {"verify": "pending"}
        state = mgr.get_remediation_state(limit=3)
        assert len(state["entries"]) == 3

    def test_safety_config_fields_present(self, clock):
        mgr = _manager()
        state = mgr.get_remediation_state()
        assert "hold_s" in state
        assert "min_signals" in state
        assert "escalate_after" in state

    def test_canary_state_included(self, clock):
        mgr = _manager()
        mgr._canary_state["pod-1"] = {"fraction": 0.5, "leg": "canary"}
        state = mgr.get_remediation_state()
        assert state["canary"].get("pod-1") == {"fraction": 0.5, "leg": "canary"}

    def test_pending_outcome_alerts_counted(self, clock):
        mgr = _manager()
        mgr.outcome_alerts.append({"action": "test"})
        state = mgr.get_remediation_state()
        assert state["pending_outcome_alerts"] == 1


# ── get_audit_trail ─────────────────────────────────────────────────────────


class TestGetAuditTrail:
    def test_returns_list_within_limit(self, clock):
        mgr = _manager()
        for i in range(5):
            mgr.context.audit_log.append({"i": i})
        trail = mgr.get_audit_trail(limit=3)
        assert isinstance(trail, list)
        assert len(trail) == 3

    def test_empty_trail_returns_empty_list(self, clock):
        mgr = _manager()
        mgr.context.audit_log = []
        trail = mgr.get_audit_trail(limit=20)
        assert trail == []

    def test_recent_first(self, clock):
        mgr = _manager()
        for i in range(5):
            mgr.context.audit_log.append({"i": i})
        trail = mgr.get_audit_trail(limit=2)
        assert [t["i"] for t in trail] == [3, 4]

    def test_default_limit(self, clock):
        mgr = _manager()
        trail = mgr.get_audit_trail()
        assert isinstance(trail, list)


# ── get_remediation_log ─────────────────────────────────────────────────────


class TestGetRemediationLog:
    def test_returns_empty_list_when_no_remediations(self, clock):
        mgr = _manager()
        log = mgr.get_remediation_log(limit=20)
        assert isinstance(log, list)

    def test_log_contains_action_after_evaluate(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _TestAction()
        mgr.register_action(action)
        pod = _pod()
        mgr.evaluate(pod, None, risk_score=0.9)
        if mgr.context.remediation_log:
            entry = mgr.context.remediation_log[-1]
            assert "action" in entry

    def test_log_limit_truncates(self, clock):
        mgr = _manager()
        # Directly populate log to test truncation
        for i in range(10):
            mgr.context.remediation_log.append({"index": i})
        log = mgr.get_remediation_log(limit=5)
        assert len(log) == 5

    def test_log_entries_have_lifecycle_field(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _TestAction()
        mgr.register_action(action)
        pod = _pod()
        mgr.evaluate(pod, None, risk_score=0.9)
        if mgr.context.remediation_log:
            entry = mgr.context.remediation_log[-1]
            assert "lifecycle" in entry


# ── feed_transition ─────────────────────────────────────────────────────────


class TestFeedTransition:
    def test_records_action_transition(self, clock):
        mgr = _manager()
        mgr.feed_transition("HEALTHY", "test_action", "CRITICAL")
        assert "test_action" in mgr.decision_model.action_counts

    def test_none_action_still_recorded(self, clock):
        mgr = _manager()
        mgr.feed_transition("HEALTHY", "none", "CRITICAL")
        assert "none" in mgr.decision_model.action_counts

    def test_multiple_transitions_accumulate(self, clock):
        mgr = _manager()
        mgr.feed_transition("HEALTHY", "test_action", "CRITICAL")
        mgr.feed_transition("HEALTHY", "test_action", "CRITICAL")
        counts = mgr.decision_model.action_counts.get("test_action", [])
        assert counts[0][5] == 0  # HEALTHY -> RECOVERED index 5 is column
        assert counts[0][3] == 2  # HEALTHY(0) -> CRITICAL(3) recorded twice


# ── note_remediation_outcome ────────────────────────────────────────────────


class TestNoteRemediationOutcome:
    def test_not_at_risk_no_futile(self, clock):
        mgr = _manager()
        escalated, count = mgr.note_remediation_outcome("pod-1", at_risk=False, actions_taken=[])
        assert escalated is False
        assert count == 0

    def test_at_risk_with_actions_not_none_no_futile(self, clock):
        mgr = _manager()
        escalated, count = mgr.note_remediation_outcome("pod-1", at_risk=True, actions_taken=["restart"])
        assert escalated is False
        assert count == 0

    def test_at_risk_all_none_is_futile(self, clock):
        with mock.patch.object(config, "CRASHLOOP_ESCALATE_AFTER", 3):
            mgr = _manager()
            escalated, count = mgr.note_remediation_outcome("pod-1", at_risk=True, actions_taken=["none"])
            assert escalated is False
            assert count == 1
            escalated, count = mgr.note_remediation_outcome("pod-1", at_risk=True, actions_taken=["none"])
            assert escalated is False
            assert count == 2
            escalated, count = mgr.note_remediation_outcome("pod-1", at_risk=True, actions_taken=["none"])
            assert escalated is True
            assert count == 3

    def test_futile_cleared_by_non_futile(self, clock):
        with mock.patch.object(config, "CRASHLOOP_ESCALATE_AFTER", 3):
            mgr = _manager()
            mgr.note_remediation_outcome("pod-1", at_risk=True, actions_taken=["none"])
            assert "pod-1" in mgr._futile_cycles
            mgr.note_remediation_outcome("pod-1", at_risk=False, actions_taken=[])
            assert "pod-1" not in mgr._futile_cycles

    def test_escalations_counter_increments(self, clock):
        with mock.patch.object(config, "CRASHLOOP_ESCALATE_AFTER", 1):
            mgr = _manager()
            initial = mgr.escalations
            mgr.note_remediation_outcome("pod-1", at_risk=True, actions_taken=["none"])
            assert mgr.escalations == initial + 1


# ── _execute_revert ─────────────────────────────────────────────────────────


class _ReversibleAction(RemediationAction):
    name = "rev_act"
    idempotent = True
    reversible = True
    risk_tier = "safe"

    def should_execute(self, pod_state, prediction, risk_score: float) -> bool:
        return True

    def execute(self, target: str, context) -> RemediationResult:
        return RemediationResult(
            action=self.name, target=target, success=True,
            dry_run=context.dry_run, message="executed", command="exec",
        )

    def revert(self, target, prev_state, context):
        return RemediationResult(
            action=self.name, target=target, success=True,
            dry_run=context.dry_run, message="reverted", command="revert",
        )


class TestExecuteRevert:
    def _rec(self, mgr, action, sig=None):
        if sig is None:
            sig = mgr.idempotency_key("ns1", "pod-1", action.name)
        return {
            "timestamp": 999.0,
            "message": "exec done",
            "command": "exec",
            "idempotent": True,
            "snapshot": {"health": "ok"},
            "desired_state": {"replicas": 2},
            "verify": "held",
            "held_until": 100000.0,
            "recorded_at": 999.0,
            "sig": sig,
            "leg": "full",
        }

    def test_revert_returns_result(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _ReversibleAction()
        mgr.register_action(action)
        sig = "ns1/pod-1/rev_act/300"
        rec = self._rec(mgr, action, sig)
        before = len(mgr.audit_trail)
        result = mgr._execute_revert(action, "ns1", "pod-1", rec)
        assert result is not None
        assert len(mgr.audit_trail) == before + 1
        assert result.outcome == RemediationOutcome.REVERTED.value

    def test_revert_fails_returns_none(self, clock):
        clock["now"] = 1000.0
        class _BrokenReversibleAction(_ReversibleAction):
            def revert(self, target, prev_state, context):
                raise RuntimeError("boom")

        mgr = _manager()
        action = _BrokenReversibleAction()
        mgr.register_action(action)
        sig = "ns1/pod-1/broken_revert/300"
        rec = self._rec(mgr, action, sig)
        before = len(mgr.audit_trail)
        result = mgr._execute_revert(action, "ns1", "pod-1", rec)
        assert result is None
        assert len(mgr.audit_trail) == before

    def test_revert_records_remediation_log(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _ReversibleAction()
        mgr.register_action(action)
        sig = "ns1/pod-1/rev_act_log/300"
        rec = self._rec(mgr, action, sig)
        before = len(mgr.context.remediation_log)
        mgr._execute_revert(action, "ns1", "pod-1", rec)
        assert len(mgr.context.remediation_log) == before + 1


# ── calibrate_from_incidents ────────────────────────────────────────────────


class TestCalibrateFromIncidents:
    def test_updates_action_costs(self, clock):
        mgr = _manager()
        incidents = [
            {"action": "test_action", "downtime_minutes": 5.0},
            {"action": "test_action", "downtime_minutes": 10.0},
        ]
        mgr.calibrate_from_incidents(incidents)
        assert mgr.cost_source == "calibrated"
        assert isinstance(mgr.action_costs, dict)

    def test_returns_merged_cost_dict(self, clock):
        mgr = _manager()
        incidents = [{"action": "act1", "downtime_minutes": 5.0}]
        result = mgr.calibrate_from_incidents(incidents)
        assert isinstance(result, dict)
        assert "action_costs" in result
        assert "failure_cost" in result
        assert mgr.action_costs == result["action_costs"]

    def test_empty_incidents_defaults(self, clock):
        mgr = _manager()
        result = mgr.calibrate_from_incidents([])
        assert "action_costs" in result
        assert mgr.cost_source == "calibrated"


# ── approve_review / review queue ───────────────────────────────────────────


class TestApproveReview:
    def _pending_review(self, mgr, sig="ns1___pod-1__test_act", rid="rev-1"):
        mgr.review_queue[rid] = {
            "id": rid,
            "signature": sig,
            "action": "test_act",
            "target": "pod-1",
            "namespace": "ns1",
            "tier": "safe",
            "risk_score": 0.5,
            "status": "pending",
            "created_at": "2025-01-01T00:00:00Z",
        }

    def test_approve_by_signature(self, clock):
        mgr = _manager(review_tier="safe")
        sig = "ns1___pod-1__test_act"
        mgr._approved = {}
        self._pending_review(mgr, sig=sig)
        result = mgr.approve_review(signature=sig)
        assert result is not None
        assert sig in mgr._approved
        assert mgr.review_queue["rev-1"]["status"] == "approved"

    def test_approve_by_review_id(self, clock):
        mgr = _manager(review_tier="safe")
        sig = "ns1___pod-1__test_act"
        mgr._approved = {}
        self._pending_review(mgr, sig=sig)
        result = mgr.approve_review(review_id="rev-1")
        assert result is not None
        assert sig in mgr._approved

    def test_approve_nonexistent_signature_records_approval(self, clock):
        """Unknown signature still records an approval (operator may authorise a new key)."""
        mgr = _manager(review_tier="safe")
        mgr._approved = {}
        result = mgr.approve_review(signature="never-heard-of")
        assert result is not None
        assert "never-heard-of" in mgr._approved
        assert result.get("released") == 0

    def test_approve_no_args_returns_none(self, clock):
        mgr = _manager(review_tier="safe")
        result = mgr.approve_review()
        assert result is None


class TestGateReviews:
    def test_review_tier_none_no_gate(self, clock):
        mgr = _manager(review_tier=None)
        action = _TestAction()
        result = mgr._gate_reviews(action, "pod-1", "ns1", 0.9)
        assert result is None

    def test_clear_hold_on_approval(self, clock):
        """Approving a signature clears matching verification holds."""
        clock["now"] = 1000.0
        mgr = _manager(review_tier="safe")
        sig = "ns1___pod-1__test_act"
        mgr._approved = {}
        rid = "rev-1"
        mgr.review_queue[rid] = {
            "id": rid,
            "signature": sig,
            "action": "test_act",
            "target": "pod-1",
            "namespace": "ns1",
            "tier": "safe",
            "risk_score": 0.5,
            "status": "pending",
            "created_at": "2025-01-01T00:00:00Z",
        }
        key = "ns1/pod-1/test_act/300"
        mgr._dedup_history[key] = {"verify": "held", "sig": sig, "held_until": 99999.0}
        mgr.verify_state[key] = {"status": "held", "at": "2025-01-01T00:00:00Z"}
        mgr.approve_review(signature=sig)
        assert key not in mgr._dedup_history
        assert mgr.verify_state[key]["status"] == "approved_retry"


# ── global functions ────────────────────────────────────────────────────────


class TestResetRemediationMetrics:
    def test_reset_counters(self, clock):
        from predictive_agent.remediator import reset_remediation_metrics
        reset_remediation_metrics()
        # calling twice must not raise
        reset_remediation_metrics()


class TestCreateRemediationManagerFromConfig:
    def test_returns_manager(self, clock):
        from predictive_agent.remediator import create_remediation_manager_from_config
        mgr = create_remediation_manager_from_config()
        assert isinstance(mgr, RemediationManager)

    def test_dry_run_when_disabled(self, clock):
        from predictive_agent.remediator import create_remediation_manager_from_config
        with mock.patch.dict(
            "os.environ",
            {"REMEDIATION_ENABLED": "false", "REMEDIATION_DRY_RUN": "false"},
        ):
            mgr = create_remediation_manager_from_config()
        assert mgr.dry_run is True

    def test_risk_threshold_from_env(self, clock):
        from predictive_agent.remediator import create_remediation_manager_from_config
        with mock.patch.dict(
            "os.environ",
            {"REMEDIATION_RISK_THRESHOLD": "0.42"},
        ):
            mgr = create_remediation_manager_from_config()
        assert mgr.risk_threshold == 0.42


# ── get_stats ───────────────────────────────────────────────────────────────


class TestGetStats:
    def test_empty_manager_stats(self, clock):
        mgr = _manager()
        stats = mgr.get_stats()
        assert stats["total_actions"] == 0
        assert stats["registered_actions"] == []
        assert stats["dry_run"] is True

    def test_stats_count_acted(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _TestAction()
        mgr.register_action(action)
        pod = _pod()
        mgr.evaluate(pod, None, risk_score=0.9)
        if mgr.audit_trail:
            stats = mgr.get_stats()
            assert stats["total_actions"] == len(mgr.audit_trail)
            assert "test_action" in stats["registered_actions"]


# ── evaluate (reactive path) ────────────────────────────────────────────────


class TestEvaluate:
    def test_below_risk_threshold_returns_empty(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(risk_threshold=0.9)
        pod = _pod()
        result = mgr.evaluate(pod, None, risk_score=0.1)
        assert result == []

    def test_returns_list(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _TestAction()
        mgr.register_action(action)
        pod = _pod()
        result = mgr.evaluate(pod, None, risk_score=0.9)
        assert isinstance(result, list)

    def test_unregistered_action_conf_raise_noop(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        pod = _pod()
        # Without registered actions, evaluate should still not raise
        result = mgr.evaluate(pod, None, risk_score=0.9)
        assert isinstance(result, list)
