"""Survivor-killer tests pinning exact messages, keys and boundaries.

Targets the highest-survivor remediator functions from the mutation campaign:
  - _dispatch_action: 132 survivors
  - _apply_named_action: 63 survivors
  - get_remediation_state: 43 survivors
  - _handle_verify_failure: 43 survivors
  - plan_and_execute: 43 survivors
  - _execute_revert: 32 survivors
  - _run_and_record: 31 survivors
  - _hold_skip: 23 survivors
  - _queue_retry_approval: 22 survivors
  - _leg_dedup_skip: 18 survivors
  - _execute_with_timeout: 12 survivors
  - _record_execution: 6 survivors
  - _signal_snapshot: 7 survivors
  - _effective_canary_fraction: 5 survivors

Every test asserts exact strings / dict keys / numeric boundaries so any
mutated constant changes the outcome and kills the mutant.
"""

import time
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

# ── fixtures & helpers ──────────────────────────────────────────────────

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
    kw.setdefault("action_timeout_s", 10)
    return RemediationManager(**kw)


class _CanaryAction(RemediationAction):
    """Reversible action that supports canary legs."""
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
        if cmode == "widen":
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
    """Non-canary, reversible action."""
    name = "plain_act"
    idempotent = True
    reversible = True
    risk_tier = "safe"
    canary_fraction = 0.0

    def should_execute(self, pod_state, prediction, risk_score: float) -> bool:
        return True

    def execute(self, target: str, context) -> RemediationResult:
        return RemediationResult(
            action=self.name, target=target, success=True,
            dry_run=context.dry_run, message="executed plain_act", command="plain_cmd",
        )

    def revert(self, target, prev_state, context):
        return RemediationResult(
            action=self.name, target=target, success=True,
            dry_run=context.dry_run, message="reverted plain_act", command="revert_cmd",
        )


class _IneligibleAction(RemediationAction):
    name = "ineligible_act"
    idempotent = True
    reversible = False
    risk_tier = "safe"
    canary_fraction = 0.0

    def should_execute(self, pod_state, prediction, risk_score: float) -> bool:
        return False

    def execute(self, target: str, context) -> RemediationResult:
        return RemediationResult(
            action=self.name, target=target, success=True,
            dry_run=context.dry_run, message="ok", command="x",
        )


# ── _effective_canary_fraction: exact boundaries ───────────────────────


class TestEffectiveCanaryFraction:
    def test_zero_fraction_returns_zero(self):
        assert RemediationManager._effective_canary_fraction(_PlainAction()) == 0.0

    def test_canary_disabled_returns_zero(self):
        with mock.patch.object(config, "CANARY_MAX_FRACTION", 0.0):
            assert RemediationManager._effective_canary_fraction(_CanaryAction()) == 0.0

    def test_clamps_to_max(self):
        with mock.patch.object(config, "CANARY_MAX_FRACTION", 0.3):
            assert RemediationManager._effective_canary_fraction(_CanaryAction()) == 0.3

    def test_passes_through_below_max(self):
        with mock.patch.object(config, "CANARY_MAX_FRACTION", 0.9):
            assert RemediationManager._effective_canary_fraction(_CanaryAction()) == 0.5


# ── _canary_leg_keys: exact format ──────────────────────────────────────


class TestCanaryLegKeys:
    def test_canary_and_widen_suffixes(self):
        mgr = _manager(dedup_window=300)
        action = _CanaryAction()
        base = mgr.idempotency_key("ns1", "pod-1", action.name)
        assert base == "ns1/pod-1/canary_act/w300"
        canary_key, widen_key = mgr._canary_leg_keys("ns1", "pod-1", action)
        assert canary_key == "ns1/pod-1/canary_act/w300/canary"
        assert widen_key == "ns1/pod-1/canary_act/w300/widen"


# ── _signal_snapshot: exact output ─────────────────────────────────────


class TestSignalSnapshot:
    def test_snapshot_exact_values(self):
        pod = _pod(
            state="DEGRADED", memory_pct=60.5, cpu_pct=40.0,
            restart_count=3, log_errors=10, node_pressure=True,
            node_disk_pressure=True,
        )
        assert RemediationManager._signal_snapshot(pod) == {
            "state": "DEGRADED",
            "memory_pct": 60.5,
            "cpu_pct": 40.0,
            "restart_count": 3,
            "log_errors": 10,
            "node_pressure": True,
            "node_disk_pressure": True,
        }

    def test_snapshot_defensive_defaults(self):
        pod = SimpleNamespace()
        assert RemediationManager._signal_snapshot(pod) == {
            "state": "", "memory_pct": 0.0, "cpu_pct": 0.0,
            "restart_count": 0, "log_errors": 0,
            "node_pressure": False, "node_disk_pressure": False,
        }

    def test_snapshot_coerces_non_numeric(self):
        pod = SimpleNamespace(
            state=123, memory_pct="not-a-number", cpu_pct=None,
            restart_count="3", log_errors=False, node_pressure=1,
            node_disk_pressure=0,
        )
        snap = RemediationManager._signal_snapshot(pod)
        assert snap["state"] == ""
        assert snap["memory_pct"] == 0.0
        assert snap["cpu_pct"] == 0.0
        assert snap["restart_count"] == 3
        # `1 is True` is False — non-bool truthy values do NOT count as pressure
        assert snap["node_pressure"] is False
        assert snap["node_disk_pressure"] is False


# ── _signal_improved / _check_improvement: exact verdicts ──────────────


class TestSignalImproved:
    def test_state_improved(self):
        assert RemediationManager._signal_improved(
            "state", "CRITICAL", "HEALTHY") is True

    def test_state_worsened(self):
        assert RemediationManager._signal_improved(
            "state", "HEALTHY", "CRITICAL") is False

    def test_state_unchanged_neutral(self):
        assert RemediationManager._signal_improved(
            "state", "CRITICAL", "CRITICAL") is None

    def test_state_unknown_neutral(self):
        assert RemediationManager._signal_improved(
            "state", "BOGUS", "HEALTHY") is None

    def test_lower_is_better(self):
        assert RemediationManager._signal_improved("memory_pct", 90.0, 40.0) is True
        assert RemediationManager._signal_improved("memory_pct", 40.0, 90.0) is False
        assert RemediationManager._signal_improved("memory_pct", 50.0, 50.0) is None

    def test_recorded_zero_is_neutral(self):
        assert RemediationManager._signal_improved("memory_pct", 0, 40.0) is None

    def test_current_none_is_neutral(self):
        assert RemediationManager._signal_improved("memory_pct", 90.0, None) is None

    def test_node_pressure_true_to_false_improved(self):
        assert RemediationManager._signal_improved(
            "node_pressure", True, False) is True
        assert RemediationManager._signal_improved(
            "node_pressure", True, True) is False

    def test_node_pressure_no_recorded_neutral(self):
        assert RemediationManager._signal_improved(
            "node_pressure", False, True) is None

    def test_check_improvement_counts_true_verdicts(self, clock):
        mgr = _manager()
        rec = {
            "snapshot": {
                "state": "CRITICAL", "memory_pct": 90.0, "cpu_pct": 80.0,
                "restart_count": 2, "log_errors": 5,
                "node_pressure": True, "node_disk_pressure": False,
            }
        }
        pod = _pod(
            state="HEALTHY", memory_pct=40.0, cpu_pct=30.0,
            restart_count=0, log_errors=0, node_pressure=False,
            node_disk_pressure=False,
        )
        improved, criteria = mgr._check_improvement(rec, pod)
        # 7 signals: state, memory_pct, cpu_pct, restart_count, log_errors,
        # node_pressure all improve; node_disk_pressure False->False is neutral
        assert improved == 6
        assert criteria["state"] is True
        assert criteria["memory_pct"] is True
        assert criteria["node_pressure"] is True
        assert criteria["node_disk_pressure"] is None  # no recorded pressure

    def test_check_improvement_respects_resolution_signals(self, clock):
        mgr = _manager()
        rec = {"snapshot": {"state": "CRITICAL", "memory_pct": 90.0}}
        pod = _pod(state="HEALTHY", memory_pct=40.0)

        class _SignalsAction(RemediationAction):
            name = "sig"
            idempotent = True
            resolution_signals = ("state", "memory_pct")

            def should_execute(self, pod_state, prediction, risk_score):
                return True

            def execute(self, target, context):
                return RemediationResult(
                    action=self.name, target=target, success=True,
                    dry_run=context.dry_run, message="ok", command="x",
                )

        improved, criteria = mgr._check_improvement(rec, pod, _SignalsAction())
        assert improved == 2


# ── _record_execution: exact keys ──────────────────────────────────────


class TestRecordExecution:
    def test_recorded_fields_exact(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _PlainAction()
        result = RemediationResult(
            action=action.name, target="pod-1", success=True,
            dry_run=True, message="exec msg", command="exec cmd",
        )
        snapshot = {"state": "HEALTHY", "cpu_pct": 50.0}
        leg_key = "ns1/pod-1/plain_act/w300"

        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            with mock.patch.object(rem_mod, "_utc_now", return_value="2025-01-01T1000.0Z"):
                mgr._record_execution(
                    "ns1", "pod-1", action, result,
                    snapshot=snapshot, leg_key=leg_key,
                )

        rec = mgr._dedup_history[leg_key]
        assert rec["timestamp"] == 1000.0
        assert rec["message"] == "exec msg"
        assert rec["command"] == "exec cmd"
        assert rec["desired_state"] is None
        assert rec["idempotent"] is True
        assert rec["snapshot"] == snapshot
        assert rec["verify"] == "pending"
        assert rec["recorded_at"] == 1000.0
        assert rec["sig"] == "ns1/pod-1/plain_act"
        assert rec["leg"] == "full"
        assert result.idempotency_key == leg_key
        assert mgr.verify_state[leg_key] == {"status": "pending", "at": "2025-01-01T1000.0Z"}

    def test_leg_classification(self, clock):
        mgr = _manager(dedup_window=300)
        action = _PlainAction()
        result = RemediationResult(
            action=action.name, target="pod-1", success=True,
            dry_run=True, message="m", command="c",
        )
        with mock.patch.object(rem_mod, "_now", return_value=1.0):
            mgr._record_execution("ns1", "pod-1", action, result, snapshot={}, leg_key="k/canary")
        assert mgr._dedup_history["k/canary"]["leg"] == "canary"

        with mock.patch.object(rem_mod, "_now", return_value=2.0):
            mgr._record_execution("ns1", "pod-1", action, result, snapshot={}, leg_key="k/widen")
        assert mgr._dedup_history["k/widen"]["leg"] == "widen"


# ── _dedup_skip_result: exact message + window semantics ───────────────


class TestDedupSkipResult:
    def test_within_window_returns_skip(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _PlainAction()
        mgr.register_action(action)
        key = "ns1/pod-1/plain_act/w300"
        mgr._dedup_history[key] = {
            "timestamp": 900.0, "message": "m", "command": "c",
            "desired_state": None, "verify": "pending", "recorded_at": 900.0,
        }
        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            result = mgr._dedup_skip_result("ns1", "pod-1", action)

        assert result is not None
        assert result.outcome == RemediationOutcome.SKIPPED.value
        assert result.message.startswith("[op] dedup: action already executed")
        assert "idempotency_key=ns1/pod-1/plain_act/w300" in result.message
        assert mgr.deduped_count == 1

    def test_outside_window_returns_none(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _PlainAction()
        key = "ns1/pod-1/plain_act/w300"
        mgr._dedup_history[key] = {"timestamp": 100.0, "verify": "verified"}
        mgr.verify_state[key] = {"status": "verified"}
        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            result = mgr._dedup_skip_result("ns1", "pod-1", action)

        assert result is None
        assert key not in mgr._dedup_history  # pruned
        assert key not in mgr.verify_state

    def test_outside_window_hold_still_active_keeps(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _PlainAction()
        key = "ns1/pod-1/plain_act/w300"
        mgr._dedup_history[key] = {
            "timestamp": 100.0, "verify": "held", "held_until": 3000.0,
        }
        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            result = mgr._dedup_skip_result("ns1", "pod-1", action)

        assert result is None
        assert key in mgr._dedup_history  # kept for hold gate


# ── _hold_skip: exact message & expiry semantics ───────────────────────


class TestHoldSkip:
    def test_active_hold_message_exact(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _PlainAction()
        key = "ns1/pod-1/plain_act/w300"
        mgr._dedup_history[key] = {"verify": "held", "held_until": 3000.0}
        result = mgr._hold_skip(action, "pod-1", "ns1")

        assert result is not None
        assert result.outcome == RemediationOutcome.HELD.value
        assert result.message == (
            "held: verification failed; waiting for hold expiry "
            f"or operator approval (key={key})"
        )

    def test_expired_hold_returns_none_and_clears(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _PlainAction()
        key = "ns1/pod-1/plain_act/w300"
        mgr._dedup_history[key] = {"verify": "held", "held_until": 950.0}
        result = mgr._hold_skip(action, "pod-1", "ns1")

        assert result is None
        assert key not in mgr._dedup_history

    def test_no_hold_returns_none(self):
        mgr = _manager()
        assert mgr._hold_skip(_PlainAction(), "pod-1", "ns1") is None


# ── _leg_dedup_skip: exact message ─────────────────────────────────────


class TestLegDedupSkip:
    def test_within_window_returns_skip(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _CanaryAction()
        rec = {"timestamp": 900.0, "verify": "pending"}
        leg_key = "ns1/pod-1/canary_act/w300/canary"
        result = mgr._leg_dedup_skip(action, "pod-1", leg_key, rec)

        assert result is not None
        assert result.outcome == RemediationOutcome.SKIPPED.value
        assert result.idempotency_key == leg_key
        assert result.message.startswith("[op] dedup: action already executed")
        assert f"idempotency_key={leg_key}" in result.message
        assert mgr.deduped_count == 1

    def test_outside_window_returns_none(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=100)
        action = _CanaryAction()
        rec = {"timestamp": 100.0, "verify": "verified"}
        leg_key = "ns1/pod-1/canary_act/w100/canary"
        mgr._dedup_history[leg_key] = rec
        mgr.verify_state[leg_key] = {"status": "verified"}
        result = mgr._leg_dedup_skip(action, "pod-1", leg_key, rec)

        assert result is None
        assert leg_key not in mgr._dedup_history
        assert leg_key not in mgr.verify_state


# ── _queue_retry_approval: exact review item ───────────────────────────


class TestQueueRetryApproval:
    def test_review_item_fields_exact(self, clock):
        clock["now"] = 1000.0
        with mock.patch.object(rem_mod, "_utc_now", return_value="2025-01-01T1000.0Z"):
            mgr = _manager()
            action = _PlainAction()
            mgr.register_action(action)
            mgr._queue_retry_approval(action, "ns1", "pod-1", "ns1/pod-1/plain_act/w300")

        assert len(mgr.review_queue) == 1
        rid = list(mgr.review_queue.keys())[0]
        assert rid == "rev-1"
        item = mgr.review_queue[rid]
        assert item["id"] == "rev-1"
        assert item["signature"] == "ns1/pod-1/plain_act"
        assert item["action"] == "plain_act"
        assert item["target"] == "pod-1"
        assert item["namespace"] == "ns1"
        assert item["tier"] == "safe"
        assert item["risk_score"] is None
        assert item["status"] == "pending"
        assert item["reason"] == "verification_failed"
        assert item["created_at"] == "2025-01-01T1000.0Z"
        # sequence increments
        mgr._queue_retry_approval(action, "ns1", "pod-1", "k2")
        assert "rev-2" in mgr.review_queue


# ── _execute_with_timeout: exact behavior ──────────────────────────────


class TestExecuteWithTimeout:
    def test_direct_execute_when_timeout_disabled(self):
        mgr = _manager(action_timeout_s=0)
        action = _PlainAction()

        class _Ctx:
            dry_run = True

        mgr.register_action(action)
        result = mgr._execute_with_timeout(action, "pod-1", _Ctx())
        assert result.success is True
        assert result.message == "executed plain_act"

    def test_timeout_message_uses_floor_format(self):
        mgr = _manager(action_timeout_s=0.2)

        class _SlowAction(RemediationAction):
            name = "slow_act"
            idempotent = True
            reversible = False

            def should_execute(self, pod_state, prediction, risk_score):
                return True

            def execute(self, target, context):
                time.sleep(0.4)
                return RemediationResult(
                    action=self.name, target=target, success=True,
                    dry_run=context.dry_run, message="done", command="x",
                )

        action = _SlowAction()
        mgr.register_action(action)
        ctx = SimpleNamespace(dry_run=True)
        result = mgr._execute_with_timeout(action, "pod-1", ctx)
        assert result.success is False
        assert result.outcome == RemediationOutcome.FAILED.value
        # {:.0f} of 0.2 -> "0"
        assert result.message == "timeout: action did not complete within 0s"

    def test_error_raised_from_execute_propagates(self):
        mgr = _manager(action_timeout_s=5)

        class _BoomAction(RemediationAction):
            name = "boom_act"
            idempotent = True

            def should_execute(self, pod_state, prediction, risk_score):
                return True

            def execute(self, target, context):
                raise RuntimeError("boom")

        action = _BoomAction()
        mgr.register_action(action)
        ctx = SimpleNamespace(dry_run=True)
        with pytest.raises(RuntimeError):
            mgr._execute_with_timeout(action, "pod-1", ctx)


# ── _execute_revert: exact behavior ────────────────────────────────────


class TestExecuteRevert:
    def test_revert_sets_outcome_reverted(self, clock):
        mgr = _manager()
        action = _PlainAction()
        rec = {"snapshot": {"state": "HEALTHY"}, "desired_state": None}
        result = mgr._execute_revert(action, "ns1", "pod-1", rec)

        assert result is not None
        assert result.outcome == RemediationOutcome.REVERTED.value
        assert result in mgr.audit_trail
        assert len(mgr.context.remediation_log) == 1

    def test_revert_none_returns_none(self, clock):
        mgr = _manager()

        class _NoRevert(RemediationAction):
            name = "no_revert"
            idempotent = True
            reversible = True

            def should_execute(self, pod_state, prediction, risk_score):
                return True

            def execute(self, target, context):
                return RemediationResult(
                    action=self.name, target=target, success=True,
                    dry_run=context.dry_run, message="done", command="x",
                )

            def revert(self, target, prev_state, context):
                return None

        action = _NoRevert()
        result = mgr._execute_revert(action, "ns1", "pod-1", {"snapshot": {}})
        assert result is None

    def test_revert_exception_returns_none(self, clock):
        mgr = _manager()

        class _BadRevert(RemediationAction):
            name = "bad_revert"
            idempotent = True
            reversible = True

            def should_execute(self, pod_state, prediction, risk_score):
                return True

            def execute(self, target, context):
                return RemediationResult(
                    action=self.name, target=target, success=True,
                    dry_run=context.dry_run, message="done", command="x",
                )

            def revert(self, target, prev_state, context):
                raise ValueError("nope")

        action = _BadRevert()
        result = mgr._execute_revert(action, "ns1", "pod-1", {"snapshot": {}})
        assert result is None


# ── _handle_verify_failure: exact state transitions ────────────────────


class TestHandleVerifyFailure:
    def test_sets_held_state_exact(self, clock):
        clock["now"] = 1000.0
        with mock.patch.object(config, "VERIFY_HOLD_S", 60):
            with mock.patch.object(config, "REMEDITATION_REVERT_ENABLED", False):
                with mock.patch.object(rem_mod, "_utc_now", return_value="2025-01-01T1000.0Z"):
                    mgr = _manager()
                    action = _PlainAction()
                    mgr.register_action(action)
                    key = "ns1/pod-1/plain_act/w300"
                    rec = {"verify": "pending", "held_until": 0}
                    mgr._dedup_history[key] = rec
                    mgr._handle_verify_failure(action, "ns1", "pod-1", key, rec, {"state": False})

        assert mgr._verify_failures[key] == 1
        assert rec["verify"] == "held"
        assert rec["held_until"] == 1060.0
        assert mgr.verify_state[key]["status"] == "held"
        assert mgr.verify_state[key]["failures"] == 1
        assert mgr.verify_state[key]["criteria"] == {"state": False}
        assert mgr.verify_state[key]["at"] == "2025-01-01T1000.0Z"
        # no revert, no approval for reversible below escalate threshold
        alerts = [a["kind"] for a in mgr.outcome_alerts]
        assert "remediation_held" in alerts
        assert "remediation_reverted" not in alerts

    def test_reversible_action_triggers_revert_when_enabled(self, clock):
        clock["now"] = 1000.0
        with mock.patch.object(config, "VERIFY_HOLD_S", 60):
            with mock.patch.object(config, "REMEDITATION_REVERT_ENABLED", True):
                mgr = _manager()
                action = _PlainAction()  # reversible=True
                mgr.register_action(action)
                key = "ns1/pod-1/plain_act/w300"
                rec = {"verify": "pending", "snapshot": {"state": "HEALTHY"}}
                mgr._dedup_history[key] = rec
                mgr._handle_verify_failure(action, "ns1", "pod-1", key, rec, {"state": False})

        kinds = [a["kind"] for a in mgr.outcome_alerts]
        assert "remediation_reverted" in kinds
        assert len(mgr.audit_trail) >= 1

    def test_escalates_and_queues_approval_for_irreversible(self, clock):
        clock["now"] = 1000.0
        with mock.patch.object(config, "VERIFY_HOLD_S", 60):
            with mock.patch.object(config, "REMEDITATION_REVERT_ENABLED", False):
                with mock.patch.object(config, "VERIFY_ESCALATE_AFTER", 1):
                    mgr = _manager()
                    action = _IneligibleAction()  # reversible=False
                    mgr.register_action(action)
                    key = "ns1/pod-1/ineligible_act/w300"
                    rec = {"verify": "pending"}
                    mgr._dedup_history[key] = rec
                    mgr._handle_verify_failure(action, "ns1", "pod-1", key, rec, {})

        assert mgr._verify_failures[key] == 1
        kinds = [a["kind"] for a in mgr.outcome_alerts]
        assert "remediation_exhausted" in kinds
        assert len(mgr.review_queue) == 1


# ── get_remediation_state: exact keys ──────────────────────────────────


class TestGetRemediationState:
    def test_structure_keys_exact(self, clock):
        mgr = _manager(dedup_window=300)
        state = mgr.get_remediation_state(limit=50)

        assert set(state.keys()) == {
            "verify_enabled", "settling_s", "min_signals", "hold_s",
            "escalate_after", "canary_max_fraction", "revert_enabled",
            "entries", "canary", "pending_outcome_alerts",
        }
        assert state["verify_enabled"] is True
        assert state["settling_s"] == config.VERIFY_SETTLING_S
        assert state["min_signals"] == config.VERIFY_MIN_SIGNALS
        assert state["hold_s"] == config.VERIFY_HOLD_S
        assert state["escalate_after"] == config.VERIFY_ESCALATE_AFTER
        assert state["canary_max_fraction"] == config.CANARY_MAX_FRACTION
        assert state["revert_enabled"] is True
        assert state["entries"] == []
        assert state["canary"] == {}
        assert state["pending_outcome_alerts"] == 0

    def test_entry_fields_exact(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        key = "ns1/pod-1/plain_act/w300"
        mgr.verify_state[key] = {"status": "held", "criteria": {"state": False}, "at": "2025-01-01T900.0Z"}
        mgr._dedup_history[key] = {"verify": "pending", "held_until": 0, "leg": "full"}
        mgr._verify_failures[key] = 2
        mgr.outcome_alerts.append({"kind": "x"})

        state = mgr.get_remediation_state(limit=50)
        entry = state["entries"][0]
        assert entry == {
            "idempotency_key": key,
            "status": "held",
            "verify": "pending",
            "leg": "full",
            "failures": 2,
            "hold_remaining_s": 0.0,
            "criteria": {"state": False},
            "at": "2025-01-01T900.0Z",
        }
        assert state["pending_outcome_alerts"] == 1

    def test_entry_hold_remaining_computed(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        key = "ns1/pod-1/plain_act/w300"
        mgr.verify_state[key] = {"status": "held", "criteria": None, "at": "t"}
        mgr._dedup_history[key] = {"verify": "held", "held_until": 1500.0, "leg": "canary"}
        state = mgr.get_remediation_state(limit=50)
        assert state["entries"][0]["hold_remaining_s"] == 500.0
        assert state["entries"][0]["leg"] == "canary"

    def test_limit_respected(self, clock):
        mgr = _manager(dedup_window=300)
        for i in range(5):
            mgr.verify_state[f"key-{i}"] = {"status": "pending"}
        state = mgr.get_remediation_state(limit=2)
        assert len(state["entries"]) == 2


# ── _apply_named_action: exact messages ────────────────────────────────


class TestApplyNamedAction:
    def test_unregistered_returns_none(self):
        mgr = _manager()
        result = mgr._apply_named_action(
            "ghost", "pod-1", _pod(), {}, 0.5, "ns1", "node-1",
        )
        assert result is None

    def test_ineligible_message_exact(self):
        mgr = _manager()
        action = _IneligibleAction()
        mgr.register_action(action)
        result = mgr._apply_named_action(
            action.name, "pod-1", _pod(), {}, 0.5, "ns1", "node-1",
        )
        assert result is not None
        assert result.outcome == RemediationOutcome.SKIPPED.value
        assert result.message == "skipped: action ineligible"

    def test_safety_block_message_prefix(self):
        mgr = _manager()
        action = _PlainAction()
        mgr.register_action(action)
        # max_per_minute=0 -> even an empty limiter already over the limit
        mgr.safety_policy = SafetyPolicy(
            cooldown_seconds=3600, max_per_minute=0, max_per_hour=0,
        )
        result = mgr._apply_named_action(
            action.name, "pod-1", _pod(), {}, 0.5, "ns1", "node-1",
        )
        assert result is not None
        assert result.success is False
        assert result.outcome == RemediationOutcome.SKIPPED.value
        assert result.message.startswith("skipped:")

    def test_dispatch_success_message_exact(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _PlainAction()
        mgr.register_action(action)
        result = mgr._apply_named_action(
            action.name, "pod-1", _pod(replicas=1), {}, 0.5, "ns1", "node-1",
        )
        assert result is not None
        assert result.success is True
        assert result.outcome == RemediationOutcome.ACTED.value
        assert result.message == "executed plain_act"
        assert len(mgr.audit_trail) == 1
        assert mgr.context.audit_log[-1]["outcome"] == RemediationOutcome.ACTED.value

    def test_dispatch_failure_sets_failed_outcome(self):
        class _FailAction(RemediationAction):
            name = "fail_act"
            idempotent = True
            reversible = False
            canary_fraction = 0.0

            def should_execute(self, pod_state, prediction, risk_score):
                return True

            def execute(self, target, context):
                return RemediationResult(
                    action=self.name, target=target, success=False,
                    dry_run=context.dry_run, message="failed run", command="fail",
                )

        mgr = _manager()
        action = _FailAction()
        mgr.register_action(action)
        result = mgr._apply_named_action(
            action.name, "pod-1", _pod(replicas=1), {}, 0.5, "ns1", "node-1",
        )
        assert result is not None
        assert result.success is False
        assert result.outcome == RemediationOutcome.FAILED.value


# ── _dispatch_action: exact held messages (canary path) ────────────────


class TestDispatchActionHeldMessages:
    def test_held_widen_message_exact(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _CanaryAction()
        mgr.register_action(action)
        base = mgr.idempotency_key("ns1", "pod-1", action.name)
        widen_key = f"{base}/widen"
        mgr._dedup_history[widen_key] = {
            "timestamp": 900.0, "verify": "held", "held_until": 3000.0,
        }
        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            result = mgr._dispatch_action(
                action, _pod(replicas=3), "pod-1", "ns1", "node-1",
                from_state="CRITICAL", risk_score=0.9,
            )
        assert result is not None
        assert result.outcome == RemediationOutcome.HELD.value
        assert result.message == "held: widen verification failed — hold active"

    def test_held_canary_blocked_message_exact(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _CanaryAction()
        mgr.register_action(action)
        base = mgr.idempotency_key("ns1", "pod-1", action.name)
        canary_key = f"{base}/canary"
        mgr._dedup_history[canary_key] = {
            "timestamp": 900.0, "verify": "held", "held_until": 3000.0,
        }
        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            result = mgr._dispatch_action(
                action, _pod(replicas=3), "pod-1", "ns1", "node-1",
                from_state="CRITICAL", risk_score=0.9,
            )
        assert result is not None
        assert result.outcome == RemediationOutcome.HELD.value
        assert result.message == "held: canary verification failed — widen blocked until hold expiry"

    def test_held_canary_not_verified_message_exact(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _CanaryAction()
        mgr.register_action(action)
        base = mgr.idempotency_key("ns1", "pod-1", action.name)
        canary_key = f"{base}/canary"
        mgr._dedup_history[canary_key] = {
            "timestamp": 900.0, "verify": "pending", "recorded_at": 900.0,
        }
        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            result = mgr._dispatch_action(
                action, _pod(replicas=3), "pod-1", "ns1", "node-1",
                from_state="CRITICAL", risk_score=0.9,
            )
        assert result is not None
        assert result.outcome == RemediationOutcome.HELD.value
        assert result.message == "held: canary leg not yet verified — widen deferred"

    def test_widen_hold_expired_rearms_whole_cycle(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _CanaryAction()
        mgr.register_action(action)
        base = mgr.idempotency_key("ns1", "pod-1", action.name)
        widen_key = f"{base}/widen"
        canary_key = f"{base}/canary"
        mgr._dedup_history[widen_key] = {
            "timestamp": 900.0, "verify": "held", "held_until": 500.0,
        }
        mgr._dedup_history[canary_key] = {
            "timestamp": 900.0, "verify": "verified", "recorded_at": 900.0,
        }
        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            result = mgr._dispatch_action(
                action, _pod(replicas=3), "pod-1", "ns1", "node-1",
                from_state="CRITICAL", risk_score=0.9,
            )
        # Expired widen hold -> re-arm whole cycle -> fresh canary leg re-executes
        assert result is not None
        assert result.success is True
        assert result.outcome == RemediationOutcome.CANARY.value
        assert widen_key not in mgr._dedup_history
        fresh = mgr._dedup_history[canary_key]
        assert fresh["verify"] == "pending"
        assert fresh["timestamp"] == 1000.0
        assert mgr.verify_state[widen_key] == {"status": "rearmed", "at": "2025-01-01T1000Z"}

    def test_canary_hold_expired_rearms_canary(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _CanaryAction()
        mgr.register_action(action)
        base = mgr.idempotency_key("ns1", "pod-1", action.name)
        canary_key = f"{base}/canary"
        mgr._dedup_history[canary_key] = {
            "timestamp": 900.0, "verify": "held", "held_until": 500.0,
        }
        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            result = mgr._dispatch_action(
                action, _pod(replicas=3), "pod-1", "ns1", "node-1",
                from_state="CRITICAL", risk_score=0.9,
            )
        # Expired canary hold -> re-arm -> fresh canary leg re-executes and is
        # recorded back under canary_key with a new pending record.
        assert result is not None
        assert result.success is True
        assert result.outcome == RemediationOutcome.CANARY.value
        fresh = mgr._dedup_history[canary_key]
        assert fresh["verify"] == "pending"
        assert fresh["timestamp"] == 1000.0
        assert mgr.verify_state[canary_key]["status"] == "pending"


# ── plan_and_execute: exact guardrails ────────────────────────────────


class TestPlanAndExecute:
    def test_confidence_below_gate_returns_empty(self, clock):
        with mock.patch.object(config, "MIN_ACTION_CONFIDENCE", 0.7):
            mgr = _manager()
            action = _PlainAction()
            mgr.register_action(action)
            pod = _pod(state="CRITICAL", replicas=1)
            results = mgr.plan_and_execute(
                "CRITICAL", "pod-1", pod, SimpleNamespace(confidence=0.3),
                0.9, namespace="ns1", node_name="node-1",
            )
        assert results == []

    def test_confidence_none_proceeds(self, clock):
        mgr = _manager()
        # Force degraded→evaluate path to make deterministic
        mgr.degraded = True
        action = _PlainAction()
        mgr.register_action(action)
        results = mgr.plan_and_execute(
            "CRITICAL", "pod-1", _pod(state="CRITICAL", replicas=1),
            SimpleNamespace(confidence=None), 0.9, namespace="ns1", node_name="node-1",
        )
        # degraded -> evaluate -> executes plain_act
        assert any(r.success for r in results)

    def test_planner_none_result_appends_none_result(self, clock):
        with mock.patch.object(config, "MIN_ACTION_CONFIDENCE", 0.0):
            with mock.patch.object(config, "MIN_MODEL_SUPPORT", 0):
                with mock.patch.object(config, "PLANNING_ENABLED", True):
                    mgr = _manager()
                    mgr.degraded = False
                    action = _PlainAction()
                    mgr.register_action(action)
                    pod = _pod(state="HEALTHY", replicas=1)
                    results = mgr.plan_and_execute(
                        "HEALTHY", "pod-1", pod, SimpleNamespace(confidence=1.0),
                        0.9, namespace="ns1", node_name="node-1",
                    )
                    # No beneficial plan -> explicit "none" result surfaced
                    assert len(results) == 1
                    assert results[0].action == "none"
                    assert results[0].success is False
                    assert results[0].message == (
                        "planner: no beneficial action \u2014 restart yields no durable "
                        "P(FAILED) reduction"
                    )


# ── _run_and_record: exact outcome/alert behavior ──────────────────────


class TestRunAndRecord:
    def _ctx_pod(self):
        return _pod(state="CRITICAL", replicas=3)

    def test_canary_mode_sets_outcome_and_alert(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _CanaryAction()
        mgr.register_action(action)
        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            result = mgr._run_and_record(
                action, self._ctx_pod(), "pod-1", "ns1", "CRITICAL", 0.9,
                leg_key="ns1/pod-1/canary_act/w300/canary",
                canary_mode="canary", canary_fraction=0.5, replicas=3,
            )
        assert result.success is True
        assert result.outcome == RemediationOutcome.CANARY.value
        assert mgr.outcome_alerts[0]["kind"] == "remediation_canary"
        assert mgr.outcome_alerts[0]["action"] == "canary_act"
        assert mgr.outcome_alerts[0]["risk_score"] == 0.9
        assert mgr.outcome_alerts[0]["details"] == (
            "canary leg executed (fraction=0.50, replicas=3); widen awaits verification"
        )

    def test_widen_mode_sets_outcome_and_canary_state(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _CanaryAction()
        mgr.register_action(action)
        base = mgr.idempotency_key("ns1", "pod-1", action.name)
        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            result = mgr._run_and_record(
                action, self._ctx_pod(), "pod-1", "ns1", "CRITICAL", 0.9,
                leg_key="ns1/pod-1/canary_act/w300/widen",
                canary_mode="widen", canary_fraction=0.5, replicas=3,
            )
        assert result.outcome == RemediationOutcome.WIDENED.value
        assert mgr.outcome_alerts[0]["kind"] == "remediation_widened"
        assert mgr.outcome_alerts[0]["details"] == "widen leg executed after canary leg verified"
        assert mgr._canary_state[base] == "widened"

    def test_failed_without_outcome_sets_failed(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)

        class _FailAct(RemediationAction):
            name = "fail_run"
            idempotent = True
            canary_fraction = 0.0

            def should_execute(self, pod_state, prediction, risk_score):
                return True

            def execute(self, target, context):
                return RemediationResult(
                    action=self.name, target=target, success=False,
                    dry_run=context.dry_run, message="nope", command="x",
                )

        mgr.register_action(_FailAct())
        result = mgr._run_and_record(
            _FailAct(), self._ctx_pod(), "pod-1", "ns1", "CRITICAL", 0.9,
            leg_key="k", canary_mode=None, canary_fraction=0.0, replicas=1,
        )
        assert result.success is False
        assert result.outcome == RemediationOutcome.FAILED.value
        # failure is NOT recorded in dedup history
        assert "k" not in mgr._dedup_history

    def test_records_cooldown_and_pending_feedback(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _PlainAction()
        mgr.register_action(action)
        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            mgr._run_and_record(
                action, self._ctx_pod(), "pod-1", "ns1", "CRITICAL", 0.9,
                leg_key="ns1/pod-1/plain_act/w300",
                canary_mode=None, canary_fraction=0.0, replicas=1,
            )
        assert mgr._dedup_history["ns1/pod-1/plain_act/w300"]["verify"] == "pending"
        # _record_pending: (from_state, action, predicted_p_failed)
        assert mgr._pending_feedback["pod-1"] == [("CRITICAL", "plain_act", 0.9)]
        # safety_policy.record_action populated the context limiter (real-time ts)
        assert len(mgr.context.rate_limiter.get("minute", [])) == 1
        assert len(mgr.context.rate_limiter.get("hour", [])) == 1
        assert "plain_act:pod-1" in mgr.context.cooldowns


# ── record_observations / _record_pending: exact learn-loop behavior ────


class TestRecordObservations:
    def test_feeds_transitions_and_clears(self, clock):
        mgr = _manager()
        mgr._pending_feedback["pod-1"] = [("CRITICAL", "plain_act", 0.8)]
        model = mock.MagicMock()
        mgr.decision_model = model
        mgr.record_observations({"pod-1": "HEALTHY"})
        model.record_transition.assert_called_once_with("CRITICAL", "HEALTHY", "plain_act")
        assert "pod-1" not in mgr._pending_feedback

    def test_calibration_and_rl_hooks(self, clock):
        mgr = _manager()
        mgr._pending_feedback["pod-1"] = [
            ("CRITICAL", "plain_act", 0.8),
            ("DEGRADED", "plain_act", 0.2),
        ]
        cal = mock.MagicMock()
        rl = mock.MagicMock()
        mgr.calibration_monitor = cal
        mgr.rl_policy = rl
        mgr.record_observations({"pod-1": "CRITICAL"})
        # first: predicted 0.8, observed CRITICAL -> actual_failed True
        cal.record.assert_any_call(0.8, True)
        # second: predicted 0.2, observed CRITICAL -> actual_failed True
        cal.record.assert_any_call(0.2, True)
        assert rl.update.call_count == 2
        assert "pod-1" not in mgr._pending_feedback

    def test_skips_when_no_pending(self):
        mgr = _manager()
        mgr.record_observations({"pod-1": "HEALTHY"})  # should not raise

    def test_record_pending_ignores_none_state(self):
        mgr = _manager()
        mgr._record_pending("pod-1", "", "plain_act", 0.5)
        mgr._record_pending("pod-1", "CRITICAL", "none", 0.5)
        assert "pod-1" not in mgr._pending_feedback or mgr._pending_feedback.get("pod-1") == []

    def test_record_pending_appends(self):
        mgr = _manager()
        mgr._record_pending("pod-1", "CRITICAL", "plain_act", 0.5)
        mgr._record_pending("pod-1", "CRITICAL", "plain_act", 0.7)
        assert mgr._pending_feedback["pod-1"] == [
            ("CRITICAL", "plain_act", 0.5),
            ("CRITICAL", "plain_act", 0.7),
        ]

    def test_clear_pending(self):
        mgr = _manager()
        mgr._pending_feedback["pod-1"] = [("CRITICAL", "plain_act", 0.5)]
        mgr.clear_pending("pod-1")
        assert "pod-1" not in mgr._pending_feedback


# ── verify_remediations: exact verdict transitions ─────────────────────


class TestVerifyRemediations:
    def test_disabled_returns_early(self, clock):
        mgr = _manager()
        with mock.patch.object(config, "VERIFY_ENABLED", False):
            mgr.verify_remediations(_pod())  # no raise, nothing changes
        assert mgr.outcome_alerts == []

    def test_none_pod_returns_early(self, clock):
        mgr = _manager()
        mgr.verify_remediations(None)
        assert mgr.outcome_alerts == []

    def test_not_due_is_noop(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _PlainAction()
        mgr.register_action(action)
        key = "ns1/pod-1/plain_act/w300"
        mgr._dedup_history[key] = {
            "timestamp": 999.0, "verify": "pending", "recorded_at": 999.0,
            "snapshot": {"state": "CRITICAL", "memory_pct": 90.0},
        }
        mgr.verify_state[key] = {"status": "pending", "at": "t"}
        with mock.patch.object(config, "VERIFY_SETTLING_S", 600):
            with mock.patch.object(rem_mod, "_now", return_value=1000.0):
                mgr.verify_remediations(_pod(state="HEALTHY"))
        assert mgr.verify_state[key]["status"] == "pending"
        assert mgr.outcome_alerts == []

    def test_improved_marks_verified(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _PlainAction()
        mgr.register_action(action)
        key = "ns1/pod-1/plain_act/w300"
        mgr._dedup_history[key] = {
            "timestamp": 100.0, "verify": "pending", "recorded_at": 100.0,
            "snapshot": {
                "state": "CRITICAL", "memory_pct": 90.0, "cpu_pct": 80.0,
                "restart_count": 2, "log_errors": 5,
                "node_pressure": False, "node_disk_pressure": False,
            },
        }
        mgr.verify_state[key] = {"status": "pending", "at": "t"}
        mgr._verify_failures[key] = 3  # should be broken by success
        improved_pod = _pod(
            state="HEALTHY", memory_pct=30.0, cpu_pct=20.0,
            restart_count=0, log_errors=0, node_pressure=False,
            node_disk_pressure=False,
        )
        with mock.patch.object(config, "VERIFY_MIN_SIGNALS", 3):
            with mock.patch.object(config, "VERIFY_SETTLING_S", 1):
                with mock.patch.object(rem_mod, "_now", return_value=1000.0):
                    mgr.verify_remediations(improved_pod)

        assert mgr._dedup_history[key]["verify"] == "verified"
        assert key not in mgr._verify_failures  # chain broken
        assert mgr.verify_state[key]["status"] == "verified"
        kinds = [a["kind"] for a in mgr.outcome_alerts]
        assert "remediation_verified" in kinds

    def test_unimproved_marks_held(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _PlainAction()
        mgr.register_action(action)
        key = "ns1/pod-1/plain_act/w300"
        mgr._dedup_history[key] = {
            "timestamp": 100.0, "verify": "pending", "recorded_at": 100.0,
            "snapshot": {
                "state": "HEALTHY", "memory_pct": 30.0, "cpu_pct": 20.0,
                "restart_count": 0, "log_errors": 0,
                "node_pressure": False, "node_disk_pressure": False,
            },
        }
        mgr.verify_state[key] = {"status": "pending", "at": "t"}
        worse_pod = _pod(
            state="CRITICAL", memory_pct=95.0, cpu_pct=90.0,
            restart_count=5, log_errors=8, node_pressure=True,
            node_disk_pressure=False,
        )
        with mock.patch.object(config, "VERIFY_HOLD_S", 60):
            with mock.patch.object(config, "VERIFY_SETTLING_S", 1):
                with mock.patch.object(config, "REMEDITATION_REVERT_ENABLED", False):
                    with mock.patch.object(rem_mod, "_now", return_value=1000.0):
                        mgr.verify_remediations(worse_pod)

        assert mgr._dedup_history[key]["verify"] == "held"
        assert mgr._verify_failures[key] == 1
        assert mgr.verify_state[key]["status"] == "held"


# ── create_remediation_manager_from_config: exact env parsing ──────────


class TestCreateManagerFromConfig:
    def test_disabled_forces_dry_run(self, monkeypatch):
        monkeypatch.setenv("REMEDIATION_ENABLED", "false")
        monkeypatch.setenv("REMEDIATION_DRY_RUN", "false")
        mgr = rem_mod.create_remediation_manager_from_config()
        assert mgr.dry_run is True
        assert mgr.risk_threshold == 0.7

    def test_enabled_dry_run_true(self, monkeypatch):
        monkeypatch.setenv("REMEDIATION_ENABLED", "true")
        monkeypatch.setenv("REMEDIATION_DRY_RUN", "true")
        mgr = rem_mod.create_remediation_manager_from_config()
        assert mgr.dry_run is True

    def test_parses_limits_and_cooldown(self, monkeypatch):
        monkeypatch.delenv("REMEDIATION_ENABLED", raising=False)
        monkeypatch.setenv("REMEDIATION_DRY_RUN", "true")
        monkeypatch.setenv("REMEDIATION_MAX_PER_MIN", "3")
        monkeypatch.setenv("REMEDIATION_MAX_PER_HOUR", "10")
        monkeypatch.setenv("REMEDIATION_COOLDOWN_S", "120")
        monkeypatch.setenv("REMEDIATION_PROTECTED_NS", "kube-system,prod")
        monkeypatch.setenv("REMEDIATION_RISK_THRESHOLD", "0.9")
        monkeypatch.setenv("REMEDIATION_DEDUP_WINDOW", "120")
        mgr = rem_mod.create_remediation_manager_from_config()
        assert mgr.safety_policy.max_per_minute == 3
        assert mgr.safety_policy.max_per_hour == 10
        assert mgr.safety_policy.cooldown_seconds == 120
        assert {"kube-system", "prod"} == mgr.safety_policy.protected_namespaces
        assert mgr.risk_threshold == 0.9
        assert mgr.dedup_window == 120


# ── note_remediation_outcome: futile-cycle & escalation exact semantics ──


class TestNoteRemediationOutcome:
    def test_not_at_risk_clears_and_returns_false(self):
        mgr = _manager()
        mgr._futile_cycles["pod-1"] = 2
        escalated, count = mgr.note_remediation_outcome("pod-1", False, ["none"])
        assert escalated is False
        assert count == 0
        assert "pod-1" not in mgr._futile_cycles

    def test_no_actions_taken_is_not_futile(self):
        mgr = _manager()
        escalated, count = mgr.note_remediation_outcome("pod-1", True, [])
        assert escalated is False
        assert count == 0

    def test_actions_taken_are_not_futile(self):
        mgr = _manager()
        escalated, count = mgr.note_remediation_outcome("pod-1", True, ["restart", "scale"])
        assert escalated is False
        assert count == 0

    def test_futile_increments_without_escalating(self):
        mgr = _manager()
        with mock.patch.object(config, "CRASHLOOP_ESCALATE_AFTER", 3):
            escalated, count = mgr.note_remediation_outcome("pod-1", True, ["none"])
            assert escalated is False
            assert count == 1
            assert mgr.escalations == 0

    def test_futile_escalates_exactly_once_at_threshold(self):
        mgr = _manager()
        with mock.patch.object(config, "CRASHLOOP_ESCALATE_AFTER", 3):
            mgr.note_remediation_outcome("pod-1", True, ["none"])  # 1
            mgr.note_remediation_outcome("pod-1", True, ["none"])  # 2
            escalated, count = mgr.note_remediation_outcome("pod-1", True, ["none"])  # 3
            assert escalated is True
            assert count == 3
            assert mgr.escalations == 1
            # subsequent futile cycles do NOT re-escalate
            escalated, count = mgr.note_remediation_outcome("pod-1", True, ["none"])  # 4
            assert escalated is False
            assert count == 4
            assert mgr.escalations == 1


# ── _gate_reviews / approve_review / list_reviews: exact gating ─────────


class _DestructiveAction(RemediationAction):
    name = "destructive_act"
    idempotent = True
    reversible = False
    risk_tier = "destructive"
    canary_fraction = 0.0

    def should_execute(self, pod_state, prediction, risk_score):
        return True

    def execute(self, target, context):
        return RemediationResult(
            action=self.name, target=target, success=True,
            dry_run=context.dry_run, message="done", command="x",
        )


class TestGateReviews:
    def test_gate_disabled_when_tier_none(self):
        mgr = _manager(review_tier="none")
        action = _DestructiveAction()
        result = mgr._gate_reviews(action, "pod-1", "ns1", 0.9)
        assert result is None

    def test_below_tier_returns_none(self):
        mgr = _manager(review_tier="destructive")
        action = _PlainAction()  # safe < destructive
        result = mgr._gate_reviews(action, "pod-1", "ns1", 0.9)
        assert result is None

    def test_at_tier_defers_with_exact_message(self, clock):
        clock["now"] = 1000.0
        with mock.patch.object(rem_mod, "_utc_now", return_value="2025-01-01T1000.0Z"):
            mgr = _manager(review_tier="moderate")
            action = _DestructiveAction()
            result = mgr._gate_reviews(action, "pod-1", "ns1", 0.9)

        assert result is not None
        assert result.outcome == RemediationOutcome.REVIEW_REQUIRED.value
        assert result.success is False
        assert result.message == (
            "review_required: tier destructive >= moderate; "
            "approve signature ns1/pod-1/destructive_act (review id rev-1)"
        )
        assert mgr.reviews_deferred == 1
        item = mgr.review_queue["rev-1"]
        assert item["tier"] == "destructive"
        assert item["risk_score"] == 0.9
        assert item["status"] == "pending"

    def test_approved_still_valid_returns_none(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(review_tier="moderate", review_ttl_s=900)
        action = _DestructiveAction()
        mgr._approved["ns1/pod-1/destructive_act"] = 500.0
        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            result = mgr._gate_reviews(action, "pod-1", "ns1", 0.9)
        assert result is None

    def test_approved_expired_defers_again(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(review_tier="moderate", review_ttl_s=100)
        action = _DestructiveAction()
        mgr._approved["ns1/pod-1/destructive_act"] = 800.0  # 200s ago > 100s ttl
        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            result = mgr._gate_reviews(action, "pod-1", "ns1", 0.9)
        assert result is not None
        assert result.outcome == RemediationOutcome.REVIEW_REQUIRED.value


class TestApproveReview:
    def test_unknown_review_id_returns_none(self):
        mgr = _manager()
        assert mgr.approve_review(review_id="nope") is None

    def test_empty_signature_returns_none(self):
        mgr = _manager()
        assert mgr.approve_review() is None

    def test_approve_signature_releases_items(self, clock):
        clock["now"] = 1000.0
        with mock.patch.object(rem_mod, "_utc_now", return_value="2025-01-01T1000.0Z"):
            mgr = _manager(review_tier="moderate")
            action = _DestructiveAction()
            mgr._gate_reviews(action, "pod-1", "ns1", 0.9)
            result = mgr.approve_review(signature="ns1/pod-1/destructive_act")

        assert result == {"signature": "ns1/pod-1/destructive_act", "released": 1}
        assert mgr.review_queue["rev-1"]["status"] == "approved"
        assert mgr.review_queue["rev-1"]["approved_at"] == "2025-01-01T1000.0Z"
        assert mgr._approved["ns1/pod-1/destructive_act"] == 1000.0

    def test_approve_clears_held_dedup(self, clock):
        clock["now"] = 1000.0
        with mock.patch.object(rem_mod, "_utc_now", return_value="2025-01-01T1000.0Z"):
            mgr = _manager(review_tier="moderate")
            key = "ns1/pod-1/destructive_act/w300"
            mgr._dedup_history[key] = {
                "verify": "held", "held_until": 5000.0,
                "sig": "ns1/pod-1/destructive_act",
            }
            mgr.approve_review(signature="ns1/pod-1/destructive_act")

        assert key not in mgr._dedup_history
        assert mgr.verify_state[key] == {"status": "approved_retry", "at": "2025-01-01T1000.0Z"}


class TestListReviews:
    def test_pending_only_by_default(self, clock):
        mgr = _manager(review_tier="moderate")
        action = _DestructiveAction()
        mgr._gate_reviews(action, "pod-1", "ns1", 0.9)
        mgr._gate_reviews(action, "pod-2", "ns1", 0.9)
        mgr.approve_review(signature="ns1/pod-1/destructive_act")
        items = mgr.list_reviews()
        assert len(items) == 1
        assert items[0]["id"] == "rev-2"
        assert items[0]["status"] == "pending"

    def test_include_resolved_returns_all(self, clock):
        mgr = _manager(review_tier="moderate")
        action = _DestructiveAction()
        mgr._gate_reviews(action, "pod-1", "ns1", 0.9)
        mgr.approve_review(signature="ns1/pod-1/destructive_act")
        assert len(mgr.list_reviews(include_resolved=True)) == 1


# ── get_stats / get_audit_trail / get_remediation_log / plan_remediation ──


class TestStatsAndLogs:
    def test_get_stats_exact_keys(self, clock):
        mgr = _manager(dedup_window=300)
        action = _PlainAction()
        mgr.register_action(action)
        mgr._apply_named_action(action.name, "pod-1", _pod(replicas=1), {}, 0.5, "ns1", "node-1")
        stats = mgr.get_stats()

        assert stats["total_actions"] == 1
        assert stats["successful_actions"] == 1
        assert stats["failed_actions"] == 0
        assert stats["dry_run_actions"] == 1
        assert stats["registered_actions"] == ["plain_act"]
        assert stats["risk_threshold"] == 0.0
        assert stats["dry_run"] is True
        assert stats["dedup_window"] == 300
        assert stats["deduped_actions"] == 0
        assert stats["outcomes"] == {
            "acted": 1, "skipped": 0, "failed": 0, "held": 0,
            "canary": 0, "widened": 0, "reverted": 0,
        }
        assert stats["verifications"] == {
            "tracked": 1, "pending": 1, "verified": 0, "held": 0,
        }

    def test_get_audit_trail_limit(self, clock):
        mgr = _manager()
        for i in range(5):
            r = RemediationResult(
                action="a", target="t", success=True, dry_run=True,
                message=f"m{i}", command="c",
            )
            mgr.context.audit_log.append({
                "action": r.action, "target": r.target, "success": r.success,
                "dry_run": r.dry_run, "message": r.message,
                "timestamp": r.timestamp, "command": r.command,
            })
        trail = mgr.get_audit_trail(limit=2)
        assert len(trail) == 2
        assert trail[-1]["message"] == "m4"

    def test_get_remediation_log_limit_and_fields(self, clock):
        mgr = _manager()
        # The structured remediation log is populated by _execute_revert /
        # _queue_retry_approval / _gate_reviews (not plain successful dispatch).
        action = _PlainAction()
        mgr.register_action(action)
        rec = {"snapshot": {"state": "HEALTHY"}, "desired_state": None}
        mgr._execute_revert(action, "ns1", "pod-1", rec)
        log = mgr.get_remediation_log(limit=20)
        assert len(log) == 1
        entry = log[0]
        assert entry["action"] == "plain_act"
        assert entry["outcome"] == RemediationOutcome.REVERTED.value
        assert entry["lifecycle"] == RemediationOutcome.REVERTED.value
        assert entry["namespace"] == "ns1"
        assert entry["dry_run"] is True
        assert entry["message"] == "reverted plain_act"
        assert "idempotency_key" in entry
        assert "timestamp" in entry
        assert "command" in entry
        assert "success" in entry

    def test_plan_remediation_preview_shape(self):
        mgr = _manager()
        plan = mgr.plan_remediation("CRITICAL", horizon=2)
        assert isinstance(plan, dict)
        assert "steps" in plan

    def test_feed_transition_passes_args(self):
        mgr = _manager()
        model = mock.MagicMock()
        mgr.decision_model = model
        mgr.feed_transition("CRITICAL", "restart", "HEALTHY")
        model.record_transition.assert_called_once_with("CRITICAL", "HEALTHY", "restart")
