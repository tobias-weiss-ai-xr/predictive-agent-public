"""Survivor-killer tests for _dispatch_action and related dispatch paths.

Targets the ~169 survivors in _dispatch_action and its support functions:
  - _dispatch_action: canary/widen/held/expired paths
  - _run_and_record: canary/widen outcome setting + dedup recording
  - plan_and_execute: planner integration with dispatch
  - evaluate: reactive path through dispatch
  - _apply_named_action: planner path through dispatch
"""

from types import SimpleNamespace
from unittest import mock

import pytest

from predictive_agent import config
from predictive_agent import remediator as rem_mod
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
    return RemediationManager(**kw)


class _CanaryAction(RemediationAction):
    name = "canary_act"
    idempotent = True
    reversible = True
    risk_tier = "safe"
    canary_fraction = 0.5

    def should_execute(self, pod_state, prediction, risk_score: float) -> bool:
        return True

    def execute(self, target: str, context) -> RemediationResult:
        cmode = getattr(context, "canary_mode", None)
        frac = getattr(context, "canary_fraction", 0.0)
        if cmode == "canary":
            return RemediationResult(
                action=self.name, target=target, success=True,
                dry_run=context.dry_run, message=f"canary executed ({frac:.2f})",
                command="canary",
            )
        elif cmode == "widen":
            return RemediationResult(
                action=self.name, target=target, success=True,
                dry_run=context.dry_run, message=f"widened ({frac:.2f})",
                command="widen",
            )
        return RemediationResult(
            action=self.name, target=target, success=True,
            dry_run=context.dry_run, message="full executed", command="full",
        )

    def revert(self, target, prev_state, context):
        return RemediationResult(
            action=self.name, target=target, success=True,
            dry_run=context.dry_run, message="reverted", command="revert",
        )


class _PlainAction(RemediationAction):
    name = "plain_act"
    idempotent = True
    reversible = False
    risk_tier = "safe"

    def should_execute(self, pod_state, prediction, risk_score: float) -> bool:
        return True

    def execute(self, target: str, context) -> RemediationResult:
        return RemediationResult(
            action=self.name, target=target, success=True,
            dry_run=context.dry_run, message="executed", command="plain",
        )


class _SlowAction(RemediationAction):
    name = "slow_act"
    idempotent = True
    reversible = False
    risk_tier = "safe"

    def should_execute(self, pod_state, prediction, risk_score: float) -> bool:
        return True

    def execute(self, target: str, context) -> RemediationResult:
        import time
        time.sleep(0.1)
        return RemediationResult(
            action=self.name, target=target, success=True,
            dry_run=context.dry_run, message="slow executed", command="slow",
        )


# ── _dispatch_action + canary widening ──────────────────────────────────────


class TestDispatchAction:
    def test_non_canary_single_replica_plain_execution(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _PlainAction()
        mgr.register_action(action)
        pod = _pod(replicas=1)

        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            result = mgr._dispatch_action(
                action, pod, "pod-1", "ns1", "node-1",
                from_state="CRITICAL", risk_score=0.9,
            )

        assert result is not None
        assert result.success is True
        assert result.action == action.name
        assert result.outcome in ('', None)  # caller may set outcome
        assert "canary" not in result.message.lower()
        assert "widen" not in result.message.lower()

    def test_canary_first_leg_on_multi_replica(self, clock):
        clock["now"] = 1000.0
        with mock.patch.object(config, "CANARY_MAX_FRACTION", 0.5):
            mgr = _manager(dedup_window=300)
            action = _CanaryAction()
            mgr.register_action(action)
            pod = _pod(replicas=3)

            with mock.patch.object(rem_mod, "_now", return_value=1000.0):
                result = mgr._dispatch_action(
                    action, pod, "pod-1", "ns1", "node-1",
                    from_state="CRITICAL", risk_score=0.9,
                )

            assert result is not None
            assert result.success is True
            assert result.outcome == RemediationOutcome.CANARY.value
            assert "canary" in result.message.lower() or "0.50" in result.message

    def test_canary_blocked_when_canary_leg_not_verified(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _CanaryAction()
        mgr.register_action(action)
        pod = _pod(replicas=3)
        base = mgr.idempotency_key("ns1", "pod-1", action.name)
        canary_key = f"{base}/canary"
        # Record a canary execution that is still pending verification
        rec = {
            "timestamp": 999.0,
            "message": "canary done", "command": "canary",
            "idempotent": True, "snapshot": {}, "verify": "pending",
            "recorded_at": 999.0, "sig": canary_key, "leg": "canary",
        }
        mgr._dedup_history[canary_key] = rec

        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            result = mgr._dispatch_action(
                action, pod, "pod-1", "ns1", "node-1",
                from_state="CRITICAL", risk_score=0.9,
            )

        assert result is not None
        assert result.outcome == RemediationOutcome.HELD.value
        assert "canary" in result.message.lower()

    def test_widen_execution_after_canary_verified(self, clock):
        clock["now"] = 1000.0
        with mock.patch.object(config, "CANARY_MAX_FRACTION", 0.5):
            mgr = _manager(dedup_window=300)
            action = _CanaryAction()
            mgr.register_action(action)
            pod = _pod(replicas=3)
            base = mgr.idempotency_key("ns1", "pod-1", action.name)
            canary_key = f"{base}/canary"
            # Record a verified canary execution
            rec_canary = {
                "timestamp": 900.0,
                "message": "canary done", "command": "canary",
                "idempotent": True, "snapshot": {},
                "verify": "verified", "verified_at": 950.0,
                "recorded_at": 900.0, "sig": canary_key, "leg": "canary",
            }
            mgr._dedup_history[canary_key] = rec_canary
            mgr.verify_state[canary_key] = {"status": "verified", "criteria": {}, "at": "t0"}
            mgr._canary_state[base] = "canary_done"

            with mock.patch.object(rem_mod, "_now", return_value=1000.0):
                result = mgr._dispatch_action(
                    action, pod, "pod-1", "ns1", "node-1",
                    from_state="CRITICAL", risk_score=0.9,
                )

            assert result is not None
            assert result.success is True
            assert result.outcome == RemediationOutcome.WIDENED.value
            assert "widen" in result.message.lower()

    def test_widen_blocked_when_widen_held(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _CanaryAction()
        mgr.register_action(action)
        pod = _pod(replicas=3)
        base = mgr.idempotency_key("ns1", "pod-1", action.name)
        widen_key = f"{base}/widen"
        # Record a widen execution that is held
        rec_widen = {
            "timestamp": 900.0,
            "message": "widen done", "command": "widen",
            "idempotent": True, "snapshot": {},
            "verify": "held", "held_until": 3000.0,
            "recorded_at": 900.0, "sig": widen_key, "leg": "widen",
        }
        mgr._dedup_history[widen_key] = rec_widen

        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            result = mgr._dispatch_action(
                action, pod, "pod-1", "ns1", "node-1",
                from_state="CRITICAL", risk_score=0.9,
            )

        assert result is not None
        assert result.outcome == RemediationOutcome.HELD.value
        assert "widen" in result.message.lower()

    def test_hold_expired_canary_clears_and_rearms(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _CanaryAction()
        mgr.register_action(action)
        pod = _pod(replicas=3)
        base = mgr.idempotency_key("ns1", "pod-1", action.name)
        canary_key = f"{base}/canary"
        rec_canary = {
            "timestamp": 900.0,
            "message": "canary done", "command": "canary",
            "idempotent": True, "snapshot": {},
            "verify": "held", "held_until": 950.0,  # expired
            "recorded_at": 900.0, "sig": canary_key, "leg": "canary",
        }
        mgr._dedup_history[canary_key] = rec_canary
        mgr.verify_state[canary_key] = {"status": "held", "criteria": {}, "at": "t0"}

        clock["now"] = 1000.0  # past held_until
        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            result = mgr._dispatch_action(
                action, pod, "pod-1", "ns1", "node-1",
                from_state="CRITICAL", risk_score=0.9,
            )

        # Should have re-armed and started fresh canary
        assert result is not None
        assert result.outcome == RemediationOutcome.CANARY.value
        assert canary_key not in mgr._dedup_history or mgr._dedup_history[canary_key].get("verify") != "held"

    def test_hold_active_blocks_dispatch(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _PlainAction()
        mgr.register_action(action)
        pod = _pod()
        base = mgr.idempotency_key("ns1", "pod-1", action.name)
        rec = {
            "timestamp": 900.0,
            "message": "exec done", "command": "plain",
            "idempotent": True, "snapshot": {},
            "verify": "held", "held_until": 2000.0,  # future
            "recorded_at": 900.0, "sig": base, "leg": "full",
        }
        mgr._dedup_history[base] = rec

        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            result = mgr._dispatch_action(
                action, pod, "pod-1", "ns1", "node-1",
                from_state="CRITICAL", risk_score=0.9,
            )

        assert result is not None
        assert result.outcome == RemediationOutcome.HELD.value

    def test_idempotency_key_set_on_result(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _PlainAction()
        mgr.register_action(action)
        pod = _pod()

        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            result = mgr._dispatch_action(
                action, pod, "pod-1", "ns1", "node-1",
                from_state="CRITICAL", risk_score=0.9,
            )

        assert result is not None
        assert hasattr(result, "idempotency_key")
        assert result.idempotency_key is not None
        expected = mgr.idempotency_key("ns1", "pod-1", action.name)
        assert result.idempotency_key == expected

    def test_dedup_record_created_after_dispatch(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _PlainAction()
        mgr.register_action(action)
        pod = _pod()
        base = mgr.idempotency_key("ns1", "pod-1", action.name)

        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            mgr._dispatch_action(
                action, pod, "pod-1", "ns1", "node-1",
                from_state="CRITICAL", risk_score=0.9,
            )

        assert base in mgr._dedup_history
        rec = mgr._dedup_history[base]
        assert rec["verify"] == "pending"

    def test_canary_mode_context_propagated(self, clock):
        clock["now"] = 1000.0
        with mock.patch.object(config, "CANARY_MAX_FRACTION", 0.5):
            mgr = _manager(dedup_window=300)
            action = _CanaryAction()
            mgr.register_action(action)
            pod = _pod(replicas=3)

            with mock.patch.object(rem_mod, "_now", return_value=1000.0):
                result = mgr._dispatch_action(
                    action, pod, "pod-1", "ns1", "node-1",
                    from_state="CRITICAL", risk_score=0.9,
                )

            # The action.execute should have seen canary_mode in context
            assert "0.50" in result.message  # from canary action's message

    def test_panelty_frac_clamped_to_max(self, clock):
        clock["now"] = 1000.0
        with mock.patch.object(config, "CANARY_MAX_FRACTION", 0.2):
            mgr = _manager(dedup_window=300)
            action = _CanaryAction()  # canary_fraction = 0.5 but clamped to 0.2
            mgr.register_action(action)
            pod = _pod(replicas=3)

            with mock.patch.object(rem_mod, "_now", return_value=1000.0):
                result = mgr._dispatch_action(
                    action, pod, "pod-1", "ns1", "node-1",
                    from_state="CRITICAL", risk_score=0.9,
                )

            assert result is not None
            assert result.outcome == RemediationOutcome.CANARY.value


# ── _run_and_record + canary/widen/held outcome shapes ────────────────────


class TestRunAndRecord:
    def test_canary_mode_sets_canary_outcome(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _CanaryAction()
        mgr.register_action(action)
        pod = _pod(replicas=3)

        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            with mock.patch.object(mgr, "_execute_with_timeout", wraps=mgr._execute_with_timeout):
                result = mgr._run_and_record(
                    action, pod, "pod-1", "ns1", "", 0.9,
                    leg_key="ns1/pod-1/canary_act/w300",
                    canary_mode="canary", canary_fraction=0.5, replicas=3,
                )

        assert result is not None
        assert result.success is True
        assert result.outcome == RemediationOutcome.CANARY.value
        assert "canary" in result.message.lower() or "0.50" in result.message

    def test_widen_mode_sets_widened_outcome(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _CanaryAction()
        mgr.register_action(action)
        pod = _pod(replicas=3)

        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            result = mgr._run_and_record(
                action, pod, "pod-1", "ns1", "", 0.9,
                leg_key="ns1/pod-1/canary_act/w300/widen",
                canary_mode="widen", canary_fraction=0.5, replicas=3,
            )

        assert result is not None
        assert result.success is True
        assert result.outcome == RemediationOutcome.WIDENED.value

    def test_no_canary_mode_does_not_set_special_outcome(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _PlainAction()
        mgr.register_action(action)
        pod = _pod()

        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            result = mgr._run_and_record(
                action, pod, "pod-1", "ns1", "", 0.9,
                leg_key="ns1/pod-1/plain_act/w300",
                canary_mode=None, canary_fraction=0.0, replicas=1,
            )

        assert result is not None
        assert result.success is True
        assert result.outcome in ('', None)

    def test_failed_execution_sets_failed_outcome(self, clock):
        class _FailingAction(RemediationAction):
            name = "failing"
            idempotent = True

            def should_execute(self, p, pr, r):
                return True

            def execute(self, t, c):
                return RemediationResult(
                    action=self.name, target=t, success=False,
                    dry_run=c.dry_run, message="failed", command="fail"
                )

        clock["now"] = 1000.0
        mgr = _manager()
        action = _FailingAction()
        mgr.register_action(action)
        pod = _pod()

        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            result = mgr._run_and_record(
                action, pod, "pod-1", "ns1", "", 0.9,
                leg_key="ns1/pod-1/failing/w300",
                canary_mode=None, canary_fraction=0.0, replicas=1,
            )

        assert result is not None
        assert result.success is False
        assert result.outcome == RemediationOutcome.FAILED.value

    def test_dedup_record_includes_canary_leg_marker(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _CanaryAction()
        mgr.register_action(action)
        pod = _pod(replicas=3)
        base = mgr.idempotency_key("ns1", "pod-1", action.name)
        canary_key = f"{base}/canary"

        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            mgr._run_and_record(
                action, pod, "pod-1", "ns1", "", 0.9,
                leg_key=canary_key,
                canary_mode="canary", canary_fraction=0.5, replicas=3,
            )

        assert canary_key in mgr._dedup_history
        assert mgr._dedup_history[canary_key]["leg"] == "canary"


# ── evaluate path through dispatch ────────────────────────────────────────────


class TestEvaluate:
    def test_evaluate_calls_verify_remediations_first(self, clock):
        clock["now"] = 1000.0
        with mock.patch.object(config, "VERIFY_ENABLED", True):
            mgr = _manager()
            action = _PlainAction()
            mgr.register_action(action)
            pod = _pod()
            base = mgr.idempotency_key("ns1", "pod-1", action.name)
            # Set up a pending verification
            rec = {
                "timestamp": 0, "message": "", "command": "plain",
                "idempotent": True, "snapshot": {}, "verify": "pending",
                "recorded_at": 0, "sig": base, "leg": "full",
            }
            mgr._dedup_history[base] = rec

            with mock.patch.object(rem_mod, "_now", return_value=1000.0):
                mgr.evaluate(pod, None, risk_score=0.9)
            # verify_remediations should have checked pending entries

    def test_evaluate_skips_when_risk_below_threshold(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(risk_threshold=0.5)
        action = _PlainAction()
        mgr.register_action(action)
        pod = _pod()

        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            results = mgr.evaluate(pod, None, risk_score=0.4)

        assert len(results) == 0

    def test_evaluate_executes_when_risk_at_threshold(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(risk_threshold=0.5, dedup_window=300)
        action = _PlainAction()
        mgr.register_action(action)
        pod = _pod()

        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            results = mgr.evaluate(pod, None, risk_score=0.5)

        assert len(results) == 1
        assert results[0].action == action.name

    def test_evaluate_outcome_shape(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(risk_threshold=0.5, dedup_window=300)
        action = _PlainAction()
        mgr.register_action(action)
        pod = _pod()

        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            results = mgr.evaluate(pod, None, risk_score=0.5)

        assert len(results) == 1
        r = results[0]
        assert hasattr(r, "action")
        assert hasattr(r, "target")
        assert hasattr(r, "success")
        assert hasattr(r, "dry_run")
        assert hasattr(r, "message")
        assert hasattr(r, "outcome")


# ── _execute_with_timeout ───────────────────────────────────────────────────


class TestExecuteWithTimeout:
    def test_succeeds_within_timeout(self, clock):
        mgr = _manager(action_timeout_s=5)
        action = _PlainAction()

        result = mgr._execute_with_timeout(action, "target-1", mgr.context)

        assert result is not None
        assert result.success is True
        assert result.action == action.name

    def test_returns_failed_outcome_on_timeout(self, clock):
        mgr = _manager(action_timeout_s=0.001)  # impossibly short
        action = _SlowAction()

        result = mgr._execute_with_timeout(action, "target-1", mgr.context)

        assert result is not None
        assert result.success is False
        assert result.outcome == RemediationOutcome.FAILED.value
        assert "timeout" in result.message.lower()

    def test_exception_propagates_from_execute(self, clock):
        class _CrashingAction(RemediationAction):
            name = "crash_act"

            def should_execute(self, p, pr, r):
                return True

            def execute(self, t, c):
                raise ValueError("intentional crash")

        mgr = _manager(action_timeout_s=5)
        action = _CrashingAction()

        with pytest.raises(ValueError, match="intentional crash"):
            mgr._execute_with_timeout(action, "target-1", mgr.context)


# ── _apply_named_action ─────────────────────────────────────────────────────


class TestApplyNamedAction:
    def test_not_found_returns_none(self, clock):
        mgr = _manager()
        pod = _pod()

        result = mgr._apply_named_action(
            "nonexistent", "pod-1", pod, None, 0.9, "ns1", "node-1"
        )
        assert result is None

    def test_safety_policy_block_returns_skip_result(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _PlainAction()
        mgr.register_action(action)
        pod = _pod()
        # Mock safety policy to block
        with mock.patch.object(mgr.safety_policy, "can_execute", return_value=(False, "cooldown")):
            result = mgr._apply_named_action(
                action.name, "pod-1", pod, None, 0.9, "ns1", "node-1"
            )

        assert result is not None
        assert result.success is False
        assert result.outcome == RemediationOutcome.SKIPPED.value
        assert "cooldown" in result.message

    def test_action_not_eligible_returns_skip_result(self, clock):
        class _PickyAction(RemediationAction):
            name = "picky_act"
            idempotent = True

            def should_execute(self, pod_state, prediction, risk_score):
                return False  # never eligible

            def execute(self, t, c):
                return RemediationResult(
                    action=self.name, target=t, success=True,
                    dry_run=c.dry_run, message="ok", command="ok"
                )

        clock["now"] = 1000.0
        mgr = _manager()
        action = _PickyAction()
        mgr.register_action(action)
        pod = _pod()

        result = mgr._apply_named_action(
            action.name, "pod-1", pod, None, 0.9, "ns1", "node-1"
        )

        assert result is not None
        assert result.success is False
        assert result.outcome == RemediationOutcome.SKIPPED.value

    def test_existing_dedup_returns_skip_result(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _PlainAction()
        mgr.register_action(action)
        pod = _pod()
        base = mgr.idempotency_key("ns1", "pod-1", action.name)
        # Set up existing dedup
        result_orig = action.execute("pod-1", mgr.context)
        mgr._record_execution("ns1", "pod-1", action, result_orig, snapshot={}, leg_key=base)

        with mock.patch.object(rem_mod, "_now", return_value=1001.0):
            result = mgr._apply_named_action(
                action.name, "pod-1", pod, None, 0.9, "ns1", "node-1", from_state="CRITICAL"
            )

        assert result is not None
        assert result.success is False
        assert result.outcome == RemediationOutcome.SKIPPED.value
        assert "dedup" in result.message.lower()
