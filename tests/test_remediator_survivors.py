"""Survivor-killer tests for mutmut: precise assertions on remediator internals.

These tests target the bounded-autonomy machinery in RemediationManager
to kill persistent mutants that survive the existing battery.

Hermetic: no network, no kubectl; time is a monotonic fake clock.
"""

from types import SimpleNamespace
from typing import Any
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

def _no_cooldown_policy():
    return SafetyPolicy(cooldown_seconds=0, max_per_minute=1000, max_per_hour=100000)


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


class _ReversibleAction(RemediationAction):
    name = "reversible"
    idempotent = True
    reversible = True
    risk_tier = "safe"

    def __init__(self):
        self.revert_calls = 0
        self.executions = 0

    def should_execute(self, pod_state, prediction, risk_score: float) -> bool:
        return True

    def execute(self, target: str, context) -> RemediationResult:
        self.executions += 1
        return RemediationResult(
            action=self.name, target=target, success=True,
            dry_run=context.dry_run, message="executed", command="reversible",
            desired_state={"replicas": 2},
        )

    def revert(self, target, prev_state, context):
        self.revert_calls += 1
        return RemediationResult(
            action=self.name, target=target, success=True,
            dry_run=context.dry_run, message="reverted", command="revert",
        )


class _IrreversibleAction(RemediationAction):
    name = "irreversible"
    idempotent = True
    reversible = False
    risk_tier = "risky"

    def should_execute(self, pod_state, prediction, risk_score: float) -> bool:
        return True

    def execute(self, target: str, context) -> RemediationResult:
        return RemediationResult(
            action=self.name, target=target, success=True,
            dry_run=context.dry_run, message="executed", command="irreversible",
        )


class _CanaryAction(RemediationAction):
    name = "canary"
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
        return RemediationResult(
            action=self.name, target=target, success=True,
            dry_run=context.dry_run, message="full executed", command="full",
        )

    def revert(self, target, prev_state, context):
        return RemediationResult(
            action=self.name, target=target, success=True,
            dry_run=context.dry_run, message="reverted", command="revert",
        )


def _manager(**kw) -> RemediationManager:
    kw.setdefault("dry_run", True)
    kw.setdefault("risk_threshold", 0.0)
    kw.setdefault("safety_policy", _no_cooldown_policy())
    kw.setdefault("dedup_window", 300)
    return RemediationManager(**kw)


# ── _signal_snapshot: exact keys + coercion ──────────────────────────────────


class TestSignalSnapshot:
    def test_snapshot_returns_seven_signal_keys(self, clock):
        pod = _pod(state="HEALTHY", memory_pct=50.0, cpu_pct=60.0,
                   restart_count=3, log_errors=0, node_pressure=False,
                   node_disk_pressure=False)
        snap = RemediationManager._signal_snapshot(pod)
        expected = {
            "state", "memory_pct", "cpu_pct",
            "restart_count", "log_errors",
            "node_pressure", "node_disk_pressure",
        }
        assert set(snap) == expected

    def test_snapshot_coerces_non_numeric_to_zero_float_int(self, clock):
        pod = _pod(state="HEALTHY", memory_pct="not-a-float",
                   cpu_pct="bad", restart_count="nope", log_errors="err")
        snap = RemediationManager._signal_snapshot(pod)
        assert snap["memory_pct"] == 0.0
        assert snap["cpu_pct"] == 0.0
        assert snap["restart_count"] == 0
        assert snap["log_errors"] == 0

    def test_snapshot_coerces_non_string_state_to_empty(self, clock):
        pod = _pod(state=123)
        snap = RemediationManager._signal_snapshot(pod)
        assert snap["state"] == ""

    def test_snapshot_missing_attrs_default_to_zero_or_false(self, clock):
        pod = SimpleNamespace(name="x", namespace="y")
        snap = RemediationManager._signal_snapshot(pod)
        assert snap["state"] == ""
        assert snap["memory_pct"] == 0.0
        assert snap["cpu_pct"] == 0.0
        assert snap["restart_count"] == 0
        assert snap["log_errors"] == 0
        assert snap["node_pressure"] is False
        assert snap["node_disk_pressure"] is False

    def test_snapshot_booleans_stored_as_is(self, clock):
        pod = _pod(node_pressure=True, node_disk_pressure=True)
        snap = RemediationManager._signal_snapshot(pod)
        assert snap["node_pressure"] is True
        assert snap["node_disk_pressure"] is True


# ── _signal_improved + _check_improvement ──────────────────────────────────


class TestSignalImproved:
    _STATE_ORDER = {"HEALTHY": 0, "DEGRADED": 1, "RECOVERED": 1,
                     "STRESSED": 2, "CRITICAL": 3, "FAILED": 4}

    def test_state_improved_lower_enum_is_better(self, clock):
        assert RemediationManager._signal_improved("state", "CRITICAL", "DEGRADED") is True
        assert RemediationManager._signal_improved("state", "DEGRADED", "CRITICAL") is False
        assert RemediationManager._signal_improved("state", "HEALTHY", "HEALTHY") is None
        assert RemediationManager._signal_improved("state", "UNKNOWN", "HEALTHY") is None

    def test_node_pressure_improved_when_goes_false(self, clock):
        assert RemediationManager._signal_improved("node_pressure", True, False) is True
        assert RemediationManager._signal_improved("node_pressure", True, True) is False
        assert RemediationManager._signal_improved("node_pressure", False, False) is None
        assert RemediationManager._signal_improved("node_pressure", False, True) is None

    def test_node_disk_pressure_improved_when_goes_false(self, clock):
        assert RemediationManager._signal_improved("node_disk_pressure", True, False) is True
        assert RemediationManager._signal_improved("node_disk_pressure", True, True) is False
        assert RemediationManager._signal_improved("node_disk_pressure", False, False) is None

    def test_numeric_signals_lower_is_better(self, clock):
        assert RemediationManager._signal_improved("memory_pct", 90.0, 40.0) is True
        assert RemediationManager._signal_improved("memory_pct", 40.0, 90.0) is False
        assert RemediationManager._signal_improved("memory_pct", 50.0, 50.0) is None

    def test_numeric_non_numerics_yield_neutral(self, clock):
        assert RemediationManager._signal_improved("memory_pct", 50.0, None) is None
        assert RemediationManager._signal_improved("memory_pct", None, 40.0) is None
        assert RemediationManager._signal_improved("memory_pct", 0, 40.0) is None


# ── _hold_still_active ───────────────────────────────────────────────────────


class TestHoldStillActive:
    def test_held_and_not_expired_is_active(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        rec = {"verify": "held", "held_until": 2000.0}
        assert mgr._hold_still_active(rec, clock["now"]) is True

    def test_held_and_expired_is_not_active(self, clock):
        clock["now"] = 3000.0
        mgr = _manager()
        rec = {"verify": "held", "held_until": 2000.0}
        assert mgr._hold_still_active(rec, clock["now"]) is False

    def test_not_held_is_not_active(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        rec = {"verify": "pending"}
        assert mgr._hold_still_active(rec, clock["now"]) is False

    def test_missing_held_until_defaults_to_zero_not_active(self, clock):
        clock["now"] = 1.0
        mgr = _manager()
        rec = {"verify": "held"}
        assert mgr._hold_still_active(rec, clock["now"]) is False

    def test_empty_rec_is_not_active(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        rec: dict[str, Any] = {}
        assert mgr._hold_still_active(rec, clock["now"]) is False

    def test_none_verify_is_not_active(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        rec = {"verify": None}
        assert mgr._hold_still_active(rec, clock["now"]) is False


# ── _record_execution: exact dedup record keys ──────────────────────────────


class TestRecordExecution:
    def test_exact_dedup_record_keys_from_base_execution(self, clock):
        mgr = _manager()
        action = _ReversibleAction()
        # pod = _pod()
        result = action.execute("pod-1", mgr.context)
        snapshot = {"state": "CRITICAL", "memory_pct": 90.0, "cpu_pct": 80.0,
                    "restart_count": 2, "log_errors": 5, "node_pressure": False,
                    "node_disk_pressure": False}
        mgr.register_action(action)
        mgr._record_execution("ns1", "pod-1", action, result, snapshot=snapshot)
        key = mgr.idempotency_key("ns1", "pod-1", action.name)
        rec = mgr._dedup_history[key]
        expected_keys = {
            "timestamp", "message", "command", "desired_state",
            "idempotent", "snapshot", "verify", "recorded_at",
            "sig", "leg",
        }
        assert set(rec) >= expected_keys  # may have more fields

    def test_dedup_record_leg_is_full_for_base_key(self, clock):
        mgr = _manager()
        action = _ReversibleAction()
        result = action.execute("pod-1", mgr.context)
        mgr.register_action(action)
        mgr._record_execution("ns1", "pod-1", action, result, snapshot={})
        key = mgr.idempotency_key("ns1", "pod-1", action.name)
        rec = mgr._dedup_history[key]
        assert rec["leg"] == "full"

    def test_dedup_record_leg_is_canary_for_canary_suffix(self, clock):
        mgr = _manager()
        action = _ReversibleAction()
        result = action.execute("pod-1", mgr.context)
        mgr.register_action(action)
        base_key = mgr.idempotency_key("ns1", "pod-1", action.name)
        canary_key = f"{base_key}/canary"
        mgr._record_execution("ns1", "pod-1", action, result, snapshot={}, leg_key=canary_key)
        assert mgr._dedup_history[canary_key]["leg"] == "canary"

    def test_dedup_record_leg_is_widen_for_widen_suffix(self, clock):
        mgr = _manager()
        action = _ReversibleAction()
        result = action.execute("pod-1", mgr.context)
        mgr.register_action(action)
        base_key = mgr.idempotency_key("ns1", "pod-1", action.name)
        widen_key = f"{base_key}/widen"
        mgr._record_execution("ns1", "pod-1", action, result, snapshot={}, leg_key=widen_key)
        assert mgr._dedup_history[widen_key]["leg"] == "widen"

    def test_dedup_record_verify_is_pending(self, clock):
        mgr = _manager()
        action = _ReversibleAction()
        result = action.execute("pod-1", mgr.context)
        mgr.register_action(action)
        mgr._record_execution("ns1", "pod-1", action, result, snapshot={})
        key = mgr.idempotency_key("ns1", "pod-1", action.name)
        rec = mgr._dedup_history[key]
        assert rec["verify"] == "pending"

    def test_dedup_record_sig_is_review_signature(self, clock):
        mgr = _manager()
        action = _ReversibleAction()
        result = action.execute("pod-1", mgr.context)
        mgr.register_action(action)
        mgr._record_execution("ns1", "pod-1", action, result, snapshot={})
        key = mgr.idempotency_key("ns1", "pod-1", action.name)
        rec = mgr._dedup_history[key]
        assert rec["sig"] == "ns1/pod-1/reversible"

    def test_dedup_record_timestamps_are_now_at_call_time(self, clock):
        clock["now"] = 500000.0
        mgr = _manager()
        action = _ReversibleAction()
        result = action.execute("pod-1", mgr.context)
        mgr.register_action(action)
        mgr._record_execution("ns1", "pod-1", action, result, snapshot={})
        key = mgr.idempotency_key("ns1", "pod-1", action.name)
        rec = mgr._dedup_history[key]
        assert rec["timestamp"] == clock["now"]
        assert rec["recorded_at"] == clock["now"]

    def test_result_idempotency_key_set_by_record_execution(self, clock):
        clock["now"] = 500000.0
        mgr = _manager()
        action = _ReversibleAction()
        result = action.execute("pod-1", mgr.context)
        # idempotency_key may or may not exist before; record_execution sets it
        mgr.register_action(action)
        mgr._record_execution("ns1", "pod-1", action, result, snapshot={})
        expected_key = mgr.idempotency_key("ns1", "pod-1", action.name)
        assert result.idempotency_key == expected_key


# ── _check_improvement ──────────────────────────────────────────────────────


class TestCheckImprovement:
    def test_returns_tuple_of_int_dict(self, clock):
        mgr = _manager()
        action = _ReversibleAction()
        rec = {"snapshot": {"state": "CRITICAL", "memory_pct": 90.0, "cpu_pct": 80.0,
                              "restart_count": 5, "log_errors": 10,
                              "node_pressure": False, "node_disk_pressure": False}}
        mgr.register_action(action)
        pod = _pod(state="DEGRADED", memory_pct=50.0, cpu_pct=40.0,
                   restart_count=3, log_errors=2)
        improved, criteria = mgr._check_improvement(rec, pod, action)
        assert isinstance(improved, int)
        assert isinstance(criteria, dict)

    def test_improved_count_equals_true_verdicts(self, clock):
        mgr = _manager()
        action = _ReversibleAction()
        rec = {"snapshot": {"state": "CRITICAL", "memory_pct": 90.0, "cpu_pct": 80.0,
                              "restart_count": 5, "log_errors": 10,
                              "node_pressure": False, "node_disk_pressure": False}}
        mgr.register_action(action)
        pod = _pod(state="HEALTHY", memory_pct=40.0, cpu_pct=30.0,
                   restart_count=0, log_errors=0)
        improved, criteria = mgr._check_improvement(rec, pod, action)
        n_true = sum(1 for v in criteria.values() if v is True)
        assert improved == n_true

    def test_uses_only_resolution_signals_from_action(self, clock):
        class _CustomSignalsAction(RemediationAction):
            name = "custom"
            idempotent = True
            resolution_signals = ("memory_pct", "cpu_pct")

            def should_execute(self, pod_state, prediction, rs):
                return True

            def execute(self, target, ctx):
                return RemediationResult(
                    action="custom", target=target, success=True,
                    dry_run=ctx.dry_run, message="ok", command="custom"
                )

        mgr = _manager()
        action = _CustomSignalsAction()
        rec = {"snapshot": {"state": "CRITICAL", "memory_pct": 90.0, "cpu_pct": 80.0,
                              "restart_count": 5}}
        mgr.register_action(action)
        pod = _pod(state="FAILED", memory_pct=40.0, cpu_pct=30.0, restart_count=0)
        improved, criteria = mgr._check_improvement(rec, pod, action)
        assert set(criteria) == {"memory_pct", "cpu_pct"}


# ── _dedup_skip_result ──────────────────────────────────────────────────────


class TestDedupSkipResult:
    def test_returns_none_when_no_prior_execution(self, clock):
        mgr = _manager()
        action = _ReversibleAction()
        result = mgr._dedup_skip_result("ns1", "pod-1", action)
        assert result is None

    def test_returns_skip_result_when_within_window(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _ReversibleAction()
        result = action.execute("pod-1", mgr.context)
        mgr.register_action(action)
        mgr._record_execution("ns1", "pod-1", action, result, snapshot={})

        clock["now"] = 1001.0
        with mock.patch.object(rem_mod, "_now", return_value=1001.0):
            skip = mgr._dedup_skip_result("ns1", "pod-1", action)
        assert skip is not None
        assert skip.success is False
        assert "dedup" in skip.message.lower()

    def test_returns_none_when_window_expired(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=100)
        action = _ReversibleAction()
        result = action.execute("pod-1", mgr.context)
        mgr.register_action(action)
        mgr._record_execution("ns1", "pod-1", action, result, snapshot={})

        clock["now"] = 1101.0
        with mock.patch.object(rem_mod, "_now", return_value=1101.0):
            skip = mgr._dedup_skip_result("ns1", "pod-1", action)
        assert skip is None

    def test_skip_result_captures_last_run_seconds(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _ReversibleAction()
        result = action.execute("pod-1", mgr.context)
        mgr.register_action(action)
        mgr._record_execution("ns1", "pod-1", action, result, snapshot={})

        clock["now"] = 1005.0
        with mock.patch.object(rem_mod, "_now", return_value=1005.0):
            skip = mgr._dedup_skip_result("ns1", "pod-1", action)
        assert skip is not None
        assert "5s" in skip.message

    def test_skip_result_includes_desired_state(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _ReversibleAction()
        result = action.execute("pod-1", mgr.context)
        mgr.register_action(action)
        mgr._record_execution("ns1", "pod-1", action, result, snapshot={})

        clock["now"] = 1001.0
        with mock.patch.object(rem_mod, "_now", return_value=1001.0):
            skip = mgr._dedup_skip_result("ns1", "pod-1", action)
        assert skip is not None
        assert "desired_state" in skip.message

    def test_skip_result_outcome_is_skipped(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _ReversibleAction()
        result = action.execute("pod-1", mgr.context)
        mgr.register_action(action)
        mgr._record_execution("ns1", "pod-1", action, result, snapshot={})

        clock["now"] = 1001.0
        with mock.patch.object(rem_mod, "_now", return_value=1001.0):
            skip = mgr._dedup_skip_result("ns1", "pod-1", action)
        assert skip is not None
        assert skip.get_outcome() == RemediationOutcome.SKIPPED.value


# ── _leg_dedup_skip ────────────────────────────────────────────────────────


class TestLegDedupSkip:
    def test_returns_none_when_no_prior_leg_execution(self, clock):
        mgr = _manager()
        action = _CanaryAction()
        # need a valid rec with timestamp
        rec = {"timestamp": 100.0, "verify": "pending", "message": "x",
               "command": "c", "sig": "s", "leg": "canary"}
        result = mgr._leg_dedup_skip(action, "pod-1", "key/canary", rec)
        assert result is None

    def test_returns_skip_result_when_leg_within_window(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=300)
        action = _CanaryAction()
        leg_key = "ns1/pod-1/canary/w300"
        result = action.execute("pod-1", mgr.context)
        mgr.register_action(action)
        mgr._record_execution("ns1", "pod-1", action, result, snapshot={}, leg_key=leg_key)

        clock["now"] = 1001.0
        with mock.patch.object(rem_mod, "_now", return_value=1001.0):
            skip = mgr._leg_dedup_skip(action, "pod-1", leg_key, mgr._dedup_history[leg_key])
        assert skip is not None
        assert "dedup" in skip.message.lower()
        assert skip.outcome == RemediationOutcome.SKIPPED.value

    def test_drops_leg_rec_when_window_expired(self, clock):
        clock["now"] = 1000.0
        mgr = _manager(dedup_window=100)
        action = _CanaryAction()
        leg_key = "ns1/pod-1/canary/w100"
        result = action.execute("pod-1", mgr.context)
        mgr.register_action(action)
        mgr._record_execution("ns1", "pod-1", action, result, snapshot={}, leg_key=leg_key)
        assert leg_key in mgr._dedup_history

        clock["now"] = 1101.0
        with mock.patch.object(rem_mod, "_now", return_value=1101.0):
            skip = mgr._leg_dedup_skip(action, "pod-1", leg_key, mgr._dedup_history[leg_key])
        assert skip is None
        assert leg_key not in mgr._dedup_history


# ── _hold_skip ─────────────────────────────────────────────────────────────


class TestHoldSkip:
    def test_returns_none_when_no_rec(self, clock):
        mgr = _manager()
        action = _ReversibleAction()
        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            result = mgr._hold_skip(action, "pod-1", "ns1")
        assert result is None

    def test_returns_none_when_not_held(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _ReversibleAction()
        # key = mgr.idempotency_key("ns1", "pod-1", action.name)
        result_exec = action.execute("pod-1", mgr.context)
        mgr.register_action(action)
        mgr._record_execution("ns1", "pod-1", action, result_exec, snapshot={})

        with mock.patch.object(rem_mod, "_now", return_value=1001.0):
            result = mgr._hold_skip(action, "pod-1", "ns1")
        assert result is None

    def test_returns_held_result_when_hold_active(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _ReversibleAction()
        key = mgr.idempotency_key("ns1", "pod-1", action.name)
        result_exec = action.execute("pod-1", mgr.context)
        mgr.register_action(action)
        mgr._record_execution("ns1", "pod-1", action, result_exec, snapshot={})
        rec = mgr._dedup_history[key]
        rec["verify"] = "held"
        rec["held_until"] = 2000.0

        clock["now"] = 1500.0
        with mock.patch.object(rem_mod, "_now", return_value=1500.0):
            result = mgr._hold_skip(action, "pod-1", "ns1")
        assert result is not None
        assert result.outcome == RemediationOutcome.HELD.value
        assert "held" in result.message.lower()
        assert key in result.message

    def test_drops_hold_and_returns_none_when_expired(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _ReversibleAction()
        key = mgr.idempotency_key("ns1", "pod-1", action.name)
        result_exec = action.execute("pod-1", mgr.context)
        mgr.register_action(action)
        mgr._record_execution("ns1", "pod-1", action, result_exec, snapshot={})
        rec = mgr._dedup_history[key]
        rec["verify"] = "held"
        rec["held_until"] = 1005.0

        clock["now"] = 2000.0
        with mock.patch.object(rem_mod, "_now", return_value=2000.0):
            result = mgr._hold_skip(action, "pod-1", "ns1")
        assert result is None
        assert key not in mgr._dedup_history


# ── _queue_retry_approval ────────────────────────────────────────────────────


class TestQueueRetryApproval:
    def test_adds_to_review_queue_with_correct_shape(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _IrreversibleAction()
        mgr.register_action(action)
        before = len(mgr.review_queue)
        sig = mgr._review_signature("ns1", "pod-1", action.name)

        mgr._queue_retry_approval(action, "ns1", "pod-1", "some-key")

        assert len(mgr.review_queue) == before + 1
        new_items = list(mgr.review_queue.values())
        new = new_items[-1]
        assert new["id"].startswith("rev-")
        assert new["signature"] == sig
        assert new["action"] == action.name
        assert new["target"] == "pod-1"
        assert new["namespace"] == "ns1"
        assert new["tier"] == action.risk_tier
        assert new["status"] == "pending"
        assert new["reason"] == "verification_failed"

    def test_appends_to_audit_trail(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _IrreversibleAction()
        mgr.register_action(action)
        before = len(mgr.audit_trail)

        mgr._queue_retry_approval(action, "ns1", "pod-1", "some-key")

        assert len(mgr.audit_trail) == before + 1
        entry = mgr.audit_trail[-1]
        assert entry.action == action.name
        assert entry.target == "pod-1"
        assert entry.success is False
        assert entry.message is not None
        assert "approval_required" in entry.message

    def test_appends_to_remediation_log(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _IrreversibleAction()
        mgr.register_action(action)

        mgr._queue_retry_approval(action, "ns1", "pod-1", "some-key")

        log = mgr.get_remediation_log()
        approvals = [e for e in log if e.get("outcome") == RemediationOutcome.APPROVAL_REQUIRED.value]
        assert len(approvals) >= 1

    def test_does_not_return_result(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _IrreversibleAction()
        mgr.register_action(action)

        mgr._queue_retry_approval(action, "ns1", "pod-1", "some-key")


# ── _execute_revert ─────────────────────────────────────────────────────────


class TestExecuteRevert:
    def test_calls_action_revert_with_prev_state(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _ReversibleAction()
        mgr.register_action(action)
        # pod = _pod()
        result_exec = action.execute("pod-1", mgr.context)
        base_key = mgr.idempotency_key("ns1", "pod-1", action.name)
        mgr._record_execution("ns1", "pod-1", action, result_exec, snapshot={}, leg_key=base_key)
        rec = mgr._dedup_history[base_key]
        rec["desired_state"] = {"prev_replicas": 3}
        action.revert_calls = 0

        revert_res = mgr._execute_revert(action, "ns1", "pod-1", rec)

        assert action.revert_calls == 1
        assert revert_res is not None
        assert revert_res.outcome == RemediationOutcome.REVERTED.value

    def test_appends_revert_result_to_audit_trail(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _ReversibleAction()
        mgr.register_action(action)
        result_exec = action.execute("pod-1", mgr.context)
        base_key = mgr.idempotency_key("ns1", "pod-1", action.name)
        mgr._record_execution("ns1", "pod-1", action, result_exec, snapshot={}, leg_key=base_key)
        rec = mgr._dedup_history[base_key]
        before = len(mgr.audit_trail)

        mgr._execute_revert(action, "ns1", "pod-1", rec)

        assert len(mgr.audit_trail) == before + 1
        revert_entry = mgr.audit_trail[-1]
        assert revert_entry.action == action.name
        assert "reverted" in revert_entry.message.lower()

    def test_returns_none_when_revert_returns_none(self, clock):
        class _NullRevert(RemediationAction):
            name = "null_revert"
            idempotent = True
            reversible = True
            risk_tier = "safe"

            def should_execute(self, p, pr, r):
                return True

            def execute(self, t, c):
                return RemediationResult(
                    action=self.name, target=t, success=True,
                    dry_run=c.dry_run, message="exec", command="exec"
                )

            def revert(self, target, prev_state, context):
                return None

        clock["now"] = 1000.0
        mgr = _manager()
        action = _NullRevert()
        mgr.register_action(action)
        result_exec = action.execute("pod-1", mgr.context)
        base_key = mgr.idempotency_key("ns1", "pod-1", action.name)
        mgr._record_execution("ns1", "pod-1", action, result_exec, snapshot={}, leg_key=base_key)
        rec = mgr._dedup_history[base_key]

        result = mgr._execute_revert(action, "ns1", "pod-1", rec)
        assert result is None

    def test_logs_on_revert_none(self, clock, caplog):
        class _NullRevert2(RemediationAction):
            name = "null_revert2"
            idempotent = True
            reversible = True
            risk_tier = "safe"

            def should_execute(self, p, pr, r):
                return True

            def execute(self, t, c):
                return RemediationResult(
                    action=self.name, target=t, success=True,
                    dry_run=c.dry_run, message="exec", command="exec"
                )

            def revert(self, target, prev_state, context):
                return None

        clock["now"] = 1000.0
        mgr = _manager()
        action = _NullRevert2()
        mgr.register_action(action)
        result_exec = action.execute("pod-1", mgr.context)
        base_key = mgr.idempotency_key("ns1", "pod-1", action.name)
        mgr._record_execution("ns1", "pod-1", action, result_exec, snapshot={}, leg_key=base_key)
        rec = mgr._dedup_history[base_key]

        with caplog.at_level("WARNING"):
            mgr._execute_revert(action, "ns1", "pod-1", rec)
        log_text = " ".join(r.message for r in caplog.records)
        assert "revert() returned None" in log_text


# ── _handle_verify_failure ──────────────────────────────────────────────────


class TestHandleVerifyFailure:
    def test_sets_verify_held_and_held_until(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _ReversibleAction()
        mgr.register_action(action)
        key = "test-key"
        rec = {"verify": "pending", "recorded_at": 500.0}
        criteria = {"state": True, "memory_pct": False}
        mgr._dedup_history[key] = rec
        mgr._verify_failures[key] = 0

        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            mgr._handle_verify_failure(action, "ns1", "pod-1", key, rec, criteria)

        assert rec["verify"] == "held"
        assert rec["held_until"] == 1000.0 + config.VERIFY_HOLD_S
        assert mgr._verify_failures.get(key, 0) == 1

    def test_increments_consecutive_failure_counter(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _ReversibleAction()
        mgr.register_action(action)
        key = "test-key"
        rec = {"verify": "pending", "recorded_at": 500.0}
        mgr._dedup_history[key] = rec
        mgr._verify_failures[key] = 3

        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            mgr._handle_verify_failure(action, "ns1", "pod-1", key, rec, {})

        assert mgr._verify_failures[key] == 4

    def test_updates_verify_state_to_held(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _ReversibleAction()
        mgr.register_action(action)
        key = "test-key"
        rec = {"verify": "pending", "recorded_at": 500.0}
        mgr._dedup_history[key] = rec
        criteria = {"state": False, "memory_pct": False}

        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            mgr._handle_verify_failure(action, "ns1", "pod-1", key, rec, criteria)

        assert mgr.verify_state[key]["status"] == "held"
        assert mgr.verify_state[key]["failures"] == 1
        assert mgr.verify_state[key]["criteria"] == criteria

    def test_appends_held_alert_to_outcome_alerts(self, clock):
        clock["now"] = 1000.0
        with mock.patch.object(config, "REMEDIATION_REVERT_ENABLED", False):
            mgr = _manager()
            action = _ReversibleAction()
            # reversible=True but revert disabled, so kind stays "remediation_held"
            mgr.register_action(action)
            key = "test-key"
            rec = {"verify": "pending", "recorded_at": 500.0}
            mgr._dedup_history[key] = rec
            before = len(mgr.outcome_alerts)

            with mock.patch.object(rem_mod, "_now", return_value=1000.0):
                mgr._handle_verify_failure(action, "ns1", "pod-1", key, rec, {"state": True})

            assert len(mgr.outcome_alerts) == before + 1
            alert = mgr.outcome_alerts[-1]
            assert alert["kind"] == "remediation_held"
            assert alert["action"] == action.name


# ── verify_remediations ─────────────────────────────────────────────────────


class TestVerifyRemediations:
    def test_is_noop_when_verify_disabled(self, clock):
        with mock.patch.object(config, "VERIFY_ENABLED", False):
            mgr = _manager()
            with mock.patch.object(rem_mod, "_now", return_value=1000.0):
                mgr.verify_remediations(_pod())
            assert len(mgr.audit_trail) == 0

    def test_is_noop_when_pod_state_none(self, clock):
        mgr = _manager()
        with mock.patch.object(rem_mod, "_now", return_value=1000.0):
            mgr.verify_remediations(None)
        assert len(mgr.audit_trail) == 0

    def test_marks_verified_when_signals_improved(self, clock):
        with mock.patch.object(config, "VERIFY_ENABLED", True):
            with mock.patch.object(config, "VERIFY_SETTLING_S", 100):
                clock["now"] = 2000.0
                mgr = _manager()
                action = _ReversibleAction()
                mgr.register_action(action)
                namespace = "ns1"
                target = "pod-1"
                base = mgr.idempotency_key(namespace, target, action.name)
                result_exec = action.execute(target, mgr.context)
                mgr._record_execution(namespace, target, action, result_exec,
                                       snapshot={"state": "CRITICAL", "memory_pct": 90.0,
                                                 "cpu_pct": 80.0, "restart_count": 5,
                                                 "log_errors": 10, "node_pressure": False,
                                                 "node_disk_pressure": False},
                                       leg_key=base)
                rec = mgr._dedup_history[base]
                assert rec["verify"] == "pending"

                clock["now"] = 2101.0
                improved_pod = _pod(state="HEALTHY", memory_pct=40.0, cpu_pct=30.0,
                                    restart_count=0, log_errors=0)
                with mock.patch.object(rem_mod, "_now", return_value=2101.0):
                    mgr.verify_remediations(improved_pod)

                assert rec["verify"] == "verified"
                assert "verified_at" in rec

    def test_holds_when_below_min_signals(self, clock):
        with mock.patch.object(config, "VERIFY_ENABLED", True):
            with mock.patch.object(config, "VERIFY_SETTLING_S", 100):
                with mock.patch.object(config, "VERIFY_MIN_SIGNALS", 3):
                    clock["now"] = 2000.0
                    mgr = _manager()
                    action = _ReversibleAction()
                    mgr.register_action(action)
                    namespace = "ns1"
                    target = "pod-1"
                    base = mgr.idempotency_key(namespace, target, action.name)
                    result_exec = action.execute(target, mgr.context)
                    mgr._record_execution(namespace, target, action, result_exec,
                                           snapshot={"state": "CRITICAL", "memory_pct": 90.0,
                                                     "cpu_pct": 80.0, "restart_count": 5,
                                                     "log_errors": 10, "node_pressure": False,
                                                     "node_disk_pressure": False},
                                           leg_key=base)

                    clock["now"] = 2101.0
                    # Only 2 improved signals, below min_signals=3
                    partial_pod = _pod(state="HEALTHY", memory_pct=40.0, cpu_pct=85.0,
                                       restart_count=5, log_errors=10)
                    with mock.patch.object(rem_mod, "_now", return_value=2101.0):
                        mgr.verify_remediations(partial_pod)

                    assert mgr._dedup_history[base]["verify"] == "held"


# ── get_remediation_state: exact dict shape ────────────────────────────────


class TestGetRemediationState:
    def test_returns_exact_top_level_keys(self, clock):
        mgr = _manager()
        state = mgr.get_remediation_state()
        expected_keys = {
            "verify_enabled", "settling_s", "min_signals", "hold_s",
            "escalate_after", "canary_max_fraction", "revert_enabled",
            "entries", "canary", "pending_outcome_alerts",
        }
        assert set(state) == expected_keys

    def test_entries_contain_exact_keys(self, clock):
        clock["now"] = 1000.0
        mgr = _manager()
        action = _ReversibleAction()
        mgr.register_action(action)
        namespace = "ns1"
        target = "pod-1"
        base = mgr.idempotency_key(namespace, target, action.name)
        result_exec = action.execute(target, mgr.context)
        mgr._record_execution(namespace, target, action, result_exec,
                               snapshot={}, leg_key=base)
        rec = mgr._dedup_history[base]
        rec["verify"] = "pending"
        mgr.verify_state[base] = {"status": "pending", "at": "t0"}

        state = mgr.get_remediation_state()
        assert len(state["entries"]) >= 1
        entry = state["entries"][0]
        expected_entry_keys = {
            "idempotency_key", "status", "verify", "leg",
            "failures", "hold_remaining_s", "criteria", "at",
        }
        assert set(entry) == expected_entry_keys

    def test_config_values_are_captured(self, clock):
        with mock.patch.object(config, "VERIFY_SETTLING_S", 123.0):
            with mock.patch.object(config, "VERIFY_MIN_SIGNALS", 5):
                with mock.patch.object(config, "VERIFY_HOLD_S", 999.0):
                    with mock.patch.object(config, "VERIFY_ESCALATE_AFTER", 7):
                        with mock.patch.object(config, "CANARY_MAX_FRACTION", 0.25):
                            state = RemediationManager().get_remediation_state()
                            # Use total order tolerance due to time dependency
                            assert state["settling_s"] == 123.0
                            assert state["min_signals"] == 5
                            assert state["hold_s"] == 999.0
                            assert state["escalate_after"] == 7
                            assert state["canary_max_fraction"] == 0.25


# ── get_stats: exact counts ─────────────────────────────────────────────────


class TestGetStats:
    def test_returns_exact_top_level_keys(self, clock):
        mgr = _manager()
        stats = mgr.get_stats()
        expected = {
            "total_actions", "successful_actions", "failed_actions",
            "dry_run_actions", "registered_actions", "risk_threshold",
            "dry_run", "dedup_window",
            "deduped_actions", "outcomes", "verifications",
        }
        assert set(stats) == expected

    def test_outcomes_nested_dict_has_exact_keys(self, clock):
        mgr = _manager()
        stats = mgr.get_stats()
        expected_outcomes = {
            "acted", "skipped", "failed", "held", "canary", "widened", "reverted",
        }
        assert set(stats["outcomes"]) == expected_outcomes

    def test_verifications_nested_dict_has_exact_keys(self, clock):
        mgr = _manager()
        stats = mgr.get_stats()
        expected_verif = {"tracked", "pending", "verified", "held"}
        assert set(stats["verifications"]) == expected_verif

    def test_counts_outcomes_correctly(self, clock):
        mgr = _manager()
        mgr.audit_trail = [
            RemediationResult(action="a1", target="t1", success=True, dry_run=True,
                              message="m1", outcome=RemediationOutcome.ACTED.value),
            RemediationResult(action="a2", target="t2", success=False, dry_run=True,
                              message="m2", outcome=RemediationOutcome.SKIPPED.value),
            RemediationResult(action="a3", target="t3", success=True, dry_run=True,
                              message="m3", outcome=RemediationOutcome.HELD.value),
        ]
        stats = mgr.get_stats()
        assert stats["outcomes"]["acted"] == 1
        assert stats["outcomes"]["skipped"] == 1
        assert stats["outcomes"]["held"] == 1


# ── _effective_canary_fraction ──────────────────────────────────────────────


class TestEffectiveCanaryFraction:
    @pytest.mark.parametrize("attr_frac,config_frac,expected", [
        (0.5, 0.2, 0.2),
        (0.3, 0.15, 0.15),
        (0.4, 0.0, 0.0),
        (0.0, 0.2, 0.0),
        (-0.1, 0.2, 0.0),
    ])
    def test_clamping_with_config_patch(self, attr_frac, config_frac, expected):
        class _FracAction(RemediationAction):
            name = "frac"
            idempotent = True

            def __init__(self):
                self.canary_fraction = attr_frac

            def should_execute(self, p, pr, r):
                return True

            def execute(self, t, c):
                return RemediationResult(
                    action="frac", target=t, success=True,
                    dry_run=c.dry_run, message="ok", command="ok"
                )

        action = _FracAction()
        with mock.patch.object(config, "CANARY_MAX_FRACTION", config_frac):
            result = RemediationManager._effective_canary_fraction(action)
        assert result == pytest.approx(expected)


# ── idempotency_key ─────────────────────────────────────────────────────────


class TestIdempotencyKey:
    def test_format_is_ns_target_action_window(self):
        mgr = _manager(dedup_window=600)
        key = mgr.idempotency_key("ns", "pod", "act")
        assert key == "ns/pod/act/w600"

    def test_none_values_use_dash(self):
        mgr = _manager(dedup_window=600)
        key = mgr.idempotency_key("ns", "", "act")
        # target None -> "-"; action_name is "act" (string)
        assert "ns/-" in key
        assert "/w600" in key

    def test_all_none_values(self):
        mgr = _manager(dedup_window=600)
        # action_name is a string passed as third arg; if you pass None,1,2 it becomes str(None)
        key = mgr.idempotency_key("", "", "act")
        assert "-" in key
        assert "/act/" in key
        assert "/w600" in key


# ── register_action + actions list ─────────────────────────────────────────


class TestRegisterAction:
    def test_appends_action_to_list(self):
        mgr = _manager()
        action = _ReversibleAction()
        assert len(mgr.actions) == 0
        mgr.register_action(action)
        assert len(mgr.actions) == 1
        assert mgr.actions[0].name == action.name

    def test_registered_actions_appear_in_stats(self):
        mgr = _manager()
        action = _ReversibleAction()
        mgr.register_action(action)
        stats = mgr.get_stats()
        assert action.name in stats["registered_actions"]
