"""Verify-after-act & reversible remediation — closed-loop contract tests.

Covers the ``verify-after-act-reversible-remediation`` change:
  * verification config knobs (task 1.1)
  * action contract: reversible / canary_fraction / resolution_signals /
    revert default (task 1.2)
  * execution snapshot persistence in the dedup record (task 1.3)
  * verification lifecycle: pending → verified | held (tasks 2.1-2.3)
  * hold gate: no re-fire while held, re-arm after expiry (task 3.1)
  * escalation after consecutive failed verifications (task 3.2)
  * reversibility: auto-revert, irreversible → approval, disabled → hold
    (tasks 4.1-4.3)
  * canary widening: leg → verify → widen, abort on failure, clamps
    (tasks 5.1-5.3)
  * operator surfacing: alerts + /remediation/state + lifecycle field
    (tasks 6.1-6.2)

Hermetic: no network, no kubectl; time is a monotonic fake clock.
"""

import io
import json
import os
from types import SimpleNamespace
from unittest import mock

import pytest

from predictive_agent import config
from predictive_agent import remediator as rem_mod
from predictive_agent import server as server_mod
from predictive_agent.remediator import (
    RemediationAction,
    RemediationManager,
    RemediationOutcome,
    RemediationResult,
    SafetyPolicy,
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _no_cooldown_policy():
    """Safety policy that never rate-limits/cooldowns."""
    return SafetyPolicy(cooldown_seconds=0, max_per_minute=1000, max_per_hour=100000)


@pytest.fixture
def clock(monkeypatch):
    """Deterministic monotonic clock for remediator._now."""
    c = {"now": 1_000_000.0}
    monkeypatch.setattr(rem_mod, "_now", lambda: c["now"])
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
        replicas=2,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _recovered_pod(**kw) -> SimpleNamespace:
    """A pod whose signals improved vs the stressed default._pod()."""
    return _pod(state="DEGRADED", memory_pct=40.0, cpu_pct=30.0, restart_count=2, **kw)


class _MinimalAction(RemediationAction):
    """Minimal concrete action that does NOT override the closed-loop hooks."""

    name = "minimal"
    idempotent = True
    risk_tier = "safe"

    def should_execute(self, pod_state, prediction, risk_score: float) -> bool:
        return True

    def execute(self, target: str, context) -> RemediationResult:
        return RemediationResult(
            action=self.name, target=target, success=True,
            dry_run=context.dry_run, message="minimal executed", command="minimal",
        )


class _FakeAction(RemediationAction):
    """Always-executes action with configurable closed-loop properties."""

    name = "fake"
    idempotent = False
    risk_tier = "safe"

    def __init__(
        self,
        *,
        success: bool = True,
        reversible: bool = False,
        canary_fraction: float = 0.0,
        resolution_signals=None,
        should: bool = True,
        revert_result=None,
    ):
        self.reversible = reversible
        self.canary_fraction = canary_fraction
        if resolution_signals is not None:
            self.resolution_signals = tuple(resolution_signals)
        self._success = success
        self._should = should
        self._revert_result = revert_result
        self.executions = 0
        self.revert_calls = 0

    # -- abstract impls ----------------------------------------------------
    def should_execute(self, pod_state, prediction, risk_score: float) -> bool:
        return self._should

    def execute(self, target: str, context) -> RemediationResult:
        self.executions += 1
        return RemediationResult(
            action=self.name,
            target=target,
            success=self._success,
            dry_run=context.dry_run,
            message="fake executed",
            command="fake",
        )

    def revert(self, target, prev_state, context):
        self.revert_calls += 1
        if self._revert_result is not None:
            return self._revert_result
        return RemediationResult(
            action=self.name,
            target=target,
            success=True,
            dry_run=context.dry_run,
            message="fake reverted",
            command="fake revert",
        )


def _manager(**kw) -> RemediationManager:
    kw.setdefault("dry_run", True)
    kw.setdefault("risk_threshold", 0.0)
    kw.setdefault("safety_policy", _no_cooldown_policy())
    kw.setdefault("dedup_window", 300)
    return RemediationManager(**kw)


def _outcomes(manager) -> list:
    return [r.get_outcome() for r in manager.audit_trail]


def _alerts(manager) -> list:
    return [a["kind"] for a in manager.outcome_alerts]


# ── task 1.1: config knobs ───────────────────────────────────────────────────

class TestConfigKnobs:
    def test_verification_knobs_defined_with_documented_defaults(self):
        assert config.VERIFY_ENABLED is True
        assert config.VERIFY_SETTLING_S == 90.0
        assert config.VERIFY_MIN_SIGNALS == 2
        assert config.VERIFY_HOLD_S == 600.0
        assert config.VERIFY_ESCALATE_AFTER == 2
        assert config.CANARY_MAX_FRACTION == 0.15
        assert config.REMEDITATION_REVERT_ENABLED is True

    def test_knobs_readable_from_env(self):
        # The config layer reads the documented env overrides (same pattern the
        # existing suite uses for REMEDIATION_DEDUP_WINDOW).
        assert str(config.VERIFY_ENABLED).lower() == os.environ.get("VERIFY_ENABLED", "true").lower()
        assert float(os.environ.get("VERIFY_SETTLING_S", "90")) == config.VERIFY_SETTLING_S
        assert float(os.environ.get("VERIFY_HOLD_S", "600")) == config.VERIFY_HOLD_S
        assert float(os.environ.get("CANARY_MAX_FRACTION", "0.15")) == config.CANARY_MAX_FRACTION
        assert int(os.environ.get("VERIFY_MIN_SIGNALS", "2")) == config.VERIFY_MIN_SIGNALS


# ── task 1.2: action contract ───────────────────────────────────────────────

class TestActionContract:
    def test_base_action_defaults_are_closed_loop_safe(self):
        action = _MinimalAction()
        assert action.reversible is False
        assert action.canary_fraction == 0.0
        assert "memory_pct" in action.resolution_signals
        assert action.revert("pod-1", {}, None) is None  # default: not undoable

    def test_all_shipped_actions_construct(self):
        from predictive_agent.actions.container_restart import ContainerRestartAction
        from predictive_agent.actions.node_cordon import NodeCordonAction
        from predictive_agent.actions.pod_restart import PodRestartAction
        from predictive_agent.actions.right_size import RightSizeAction
        from predictive_agent.actions.rollout_restart import RolloutRestartAction
        from predictive_agent.actions.scale import DeploymentScaleAction
        from predictive_agent.actions.tune_resources import ResourceTunerAction

        for action in (
            ContainerRestartAction(),
            PodRestartAction(),
            NodeCordonAction(),
            RightSizeAction(),
            RolloutRestartAction(),
            DeploymentScaleAction(),
            ResourceTunerAction(),
        ):
            assert action.name

    def test_scale_declares_reversible_and_canary_capable(self):
        from predictive_agent.actions.scale import DeploymentScaleAction

        action = DeploymentScaleAction()
        assert action.reversible is True
        assert action.canary_fraction == 0.15

    def test_rollout_restart_documented_canary_capable_but_off(self):
        from predictive_agent.actions.rollout_restart import RolloutRestartAction

        action = RolloutRestartAction()
        assert action.canary_fraction == 0.0  # operator-enable only
        assert action.reversible is False

    def test_scale_revert_restores_prev_replicas(self):
        from predictive_agent.actions.scale import DeploymentScaleAction
        from predictive_agent.remediator import ActionContext

        action = DeploymentScaleAction()
        ctx = ActionContext(dry_run=True, namespace="ns1")
        prev = {"desired_state": {"replicas": 5, "prev_replicas": 3}}
        with pytest.MonkeyPatch.context() as mp:
            calls = {}

            def fake_run(cmd, **kw):
                calls["cmd"] = cmd
                return SimpleNamespace(returncode=0, stdout="scaled", stderr="")

            mp.setattr("predictive_agent.actions.scale.subprocess.run", fake_run)
            result = action.revert("backend", prev, ctx)
        assert result.success is True
        assert "--replicas=3" in " ".join(calls["cmd"])


# ── task 1.3: execution snapshot ────────────────────────────────────────────

class TestExecutionSnapshot:
    def test_dedup_record_persists_signal_snapshot_and_verify_state(self, clock):
        mgr = _manager()
        action = _FakeAction()
        mgr.register_action(action)
        pod = _pod()

        mgr.evaluate(pod, None, 0.9)
        rec = list(mgr._dedup_history.values())[0]
        assert rec["verify"] == "pending"
        assert rec["leg"] == "full"
        assert rec["recorded_at"] == clock["now"]
        snap = rec["snapshot"]
        assert snap["state"] == "CRITICAL"
        assert snap["memory_pct"] == 90.0
        assert snap["cpu_pct"] == 80.0
        assert mgr.verify_state[list(mgr._dedup_history)[0]]["status"] == "pending"


# ── tasks 2.1-2.3: verification lifecycle ───────────────────────────────────

class TestVerificationLifecycle:
    def test_not_due_yet_is_observe_only(self, clock):
        mgr = _manager()
        action = _FakeAction()
        mgr.register_action(action)
        pod = _pod()

        mgr.evaluate(pod, None, 0.9)
        clock["now"] += config.VERIFY_SETTLING_S - 1  # just before due
        mgr.evaluate(_recovered_pod(), None, 0.9)
        rec = list(mgr._dedup_history.values())[0]
        assert rec["verify"] == "pending"  # not due → observe-only no-op
        assert _alerts(mgr) == []
        # The re-evaluation was dedup-skipped (within window), no re-execution.
        assert action.executions == 1
        assert _alerts(mgr) == []

    def test_verified_when_signals_improve(self, clock):
        mgr = _manager()
        action = _FakeAction()
        mgr.register_action(action)
        mgr.evaluate(_pod(), None, 0.9)
        clock["now"] += config.VERIFY_SETTLING_S + 1
        mgr.evaluate(_recovered_pod(), None, 0.9)

        rec = list(mgr._dedup_history.values())[0]
        assert rec["verify"] == "verified"
        state = list(mgr.verify_state.values())[0]
        assert state["status"] == "verified"
        assert state["improved"] >= config.VERIFY_MIN_SIGNALS
        assert "remediation_verified" in _alerts(mgr)

    def test_single_signal_improvement_is_not_enough(self, clock):
        mgr = _manager()
        action = _FakeAction()
        mgr.register_action(action)
        mgr.evaluate(_pod(), None, 0.9)
        clock["now"] += config.VERIFY_SETTLING_S + 1
        # Only the markov state improved; memory/cpu unchanged (~neutral for the
        # fake action's default resolution set: restart_count equal → neutral).
        mgr.evaluate(_pod(state="DEGRADED"), None, 0.9)

        rec = list(mgr._dedup_history.values())[0]
        assert rec["verify"] == "held"
        assert "remediation_held" in _alerts(mgr)

    def test_verification_disabled_is_noop(self, clock, monkeypatch):
        monkeypatch.setattr(config, "VERIFY_ENABLED", False)
        mgr = _manager()
        action = _FakeAction()
        mgr.register_action(action)
        mgr.evaluate(_pod(), None, 0.9)
        clock["now"] += config.VERIFY_SETTLING_S + 1
        mgr.evaluate(_recovered_pod(), None, 0.9)

        rec = list(mgr._dedup_history.values())[0]
        assert rec["verify"] == "pending"
        assert _alerts(mgr) == []

    def test_resolution_signals_drive_verification(self, clock, monkeypatch):
        monkeypatch.setattr(config, "VERIFY_MIN_SIGNALS", 1)
        mgr = _manager()
        action = _FakeAction(resolution_signals=("memory_pct",))
        mgr.register_action(action)
        mgr.evaluate(_pod(), None, 0.9)
        clock["now"] += config.VERIFY_SETTLING_S + 1
        mgr.evaluate(_pod(memory_pct=30.0), None, 0.9)

        rec = list(mgr._dedup_history.values())[0]
        assert rec["verify"] == "verified"


# ── task 3.1: hold gate ─────────────────────────────────────────────────────

class TestHoldGate:
    def test_held_action_not_refired_within_hold(self, clock):
        mgr = _manager()
        action = _FakeAction()
        mgr.register_action(action)
        mgr.evaluate(_pod(), None, 0.9)  # t0: executes
        clock["now"] += config.VERIFY_SETTLING_S + 1  # t=91: verification due
        mgr.evaluate(_pod(), None, 0.9)  # fails verification → held
        assert action.executions == 1  # second eval was within dedup window
        held_until = list(mgr._dedup_history.values())[0]["held_until"]

        # After the dedup window but still inside the hold → HELD, no re-fire.
        clock["now"] = 1_000_000 + 301  # t=301: dedup(300) expired, hold active
        results = mgr.evaluate(_pod(), None, 0.9)
        assert action.executions == 1
        assert results[0].get_outcome() == RemediationOutcome.HELD.value
        assert held_until > clock["now"]

    def test_hold_expiry_rearms_and_refires(self, clock):
        mgr = _manager()
        action = _FakeAction()
        mgr.register_action(action)
        mgr.evaluate(_pod(), None, 0.9)  # t0: executes
        clock["now"] += config.VERIFY_SETTLING_S + 1
        mgr.evaluate(_pod(), None, 0.9)  # fails verification → held
        held_until = list(mgr._dedup_history.values())[0]["held_until"]
        assert action.executions == 1

        # Advance past hold expiry (also past dedup window) → re-arm + re-fire.
        clock["now"] = held_until + 1
        mgr.evaluate(_pod(), None, 0.9)
        assert action.executions == 2
        rec = list(mgr._dedup_history.values())[-1]
        assert rec["verify"] == "pending"  # fresh verification cycle
        # the old held record is gone (dropped at re-arm), so the next failure
        # counts as a fresh cycle, not an escalation chain
        assert not any(v.get("status") == "approved_retry" for v in mgr.verify_state.values())


# ── task 3.2: escalation ────────────────────────────────────────────────────

class TestEscalation:
    def test_escalates_after_consecutive_failures(self, clock):
        mgr = _manager()
        action = _FakeAction()
        mgr.register_action(action)
        mgr.evaluate(_pod(), None, 0.9)  # t0: executes

        # failure 1 → hold
        clock["now"] += config.VERIFY_SETTLING_S + 1
        mgr.evaluate(_pod(), None, 0.9)
        assert _alerts(mgr) == ["remediation_held"]
        held_until = list(mgr._dedup_history.values())[0]["held_until"]

        # re-arm (hold + dedup expired) → re-fires, then fails again → escalate
        clock["now"] = held_until + 1
        mgr.evaluate(_pod(), None, 0.9)
        assert action.executions == 2
        clock["now"] += config.VERIFY_SETTLING_S + 1
        mgr.evaluate(_pod(), None, 0.9)

        assert _alerts(mgr) == ["remediation_held", "remediation_exhausted"]
        key = list(mgr._dedup_history.keys())[-1]
        assert mgr._verify_failures.get(key, 0) >= 2
        # manager fed the shared futile-remediation channel on each failure
        assert mgr._futile_cycles.get("pod-1", 0) >= 2

    def test_irreversible_escalation_queues_approval(self, clock):
        mgr = _manager()
        action = _FakeAction(reversible=False)
        mgr.register_action(action)
        mgr.evaluate(_pod(), None, 0.9)

        clock["now"] += config.VERIFY_SETTLING_S + 1
        mgr.evaluate(_pod(), None, 0.9)  # failure 1 → hold
        held_until = list(mgr._dedup_history.values())[0]["held_until"]

        clock["now"] = held_until + 1
        mgr.evaluate(_pod(), None, 0.9)  # re-fire
        clock["now"] += config.VERIFY_SETTLING_S + 1
        mgr.evaluate(_pod(), None, 0.9)  # failure 2 → escalate + approval

        # An operator-approval review item is queued and surfaced in the audit.
        assert len(mgr.review_queue) >= 1
        item = list(mgr.review_queue.values())[-1]
        assert item["reason"] == "verification_failed"
        assert RemediationOutcome.APPROVAL_REQUIRED.value in _outcomes(mgr)

    def test_verified_resets_consecutive_failure_counter(self, clock):
        mgr = _manager()
        action = _FakeAction()
        mgr.register_action(action)
        mgr.evaluate(_pod(), None, 0.9)
        clock["now"] += config.VERIFY_SETTLING_S + 1
        mgr.evaluate(_pod(), None, 0.9)  # fail 1 → held
        key = list(mgr._dedup_history.keys())[-1]
        held_until = mgr._dedup_history[key]["held_until"]
        assert mgr._verify_failures.get(key, 0) == 1

        # re-arm then show improvement → verified, counter reset.
        clock["now"] = held_until + 1
        mgr.evaluate(_pod(), None, 0.9)
        clock["now"] += config.VERIFY_SETTLING_S + 1
        mgr.evaluate(_recovered_pod(), None, 0.9)
        assert mgr._verify_failures.get(key, 0) == 0
        assert list(mgr._dedup_history.values())[-1]["verify"] == "verified"


# ── tasks 4.1-4.3: reversibility ────────────────────────────────────────────

class TestReversibility:
    def test_reversible_action_reverted_once_on_failed_verify(self, clock):
        mgr = _manager()
        action = _FakeAction(reversible=True)
        mgr.register_action(action)
        mgr.evaluate(_pod(), None, 0.9)
        assert action.revert_calls == 0

        clock["now"] += config.VERIFY_SETTLING_S + 1
        mgr.evaluate(_pod(), None, 0.9)  # fails → revert

        assert action.revert_calls == 1
        assert RemediationOutcome.REVERTED.value in _outcomes(mgr)
        assert "remediation_reverted" in _alerts(mgr)

    def test_revert_happens_once_not_repeatedly(self, clock):
        mgr = _manager()
        action = _FakeAction(reversible=True)
        mgr.register_action(action)
        mgr.evaluate(_pod(), None, 0.9)
        clock["now"] += config.VERIFY_SETTLING_S + 1
        mgr.evaluate(_pod(), None, 0.9)
        assert action.revert_calls == 1
        # Subsequent cycles (still held) don't re-revert.
        clock["now"] += 100
        mgr.evaluate(_pod(), None, 0.9)
        assert action.revert_calls == 1

    def test_irreversible_action_not_reverted(self, clock):
        mgr = _manager()
        action = _FakeAction(reversible=False)
        mgr.register_action(action)
        mgr.evaluate(_pod(), None, 0.9)
        clock["now"] += config.VERIFY_SETTLING_S + 1
        mgr.evaluate(_pod(), None, 0.9)
        assert action.revert_calls == 0

    def test_revert_disabled_degrades_to_hold(self, clock, monkeypatch):
        monkeypatch.setattr(config, "REMEDITATION_REVERT_ENABLED", False)
        mgr = _manager()
        action = _FakeAction(reversible=True)
        mgr.register_action(action)
        mgr.evaluate(_pod(), None, 0.9)
        clock["now"] += config.VERIFY_SETTLING_S + 1
        mgr.evaluate(_pod(), None, 0.9)

        assert action.revert_calls == 0
        assert "remediation_reverted" not in _alerts(mgr)
        assert "remediation_held" in _alerts(mgr)


# ── tasks 5.1-5.3: canary widening ──────────────────────────────────────────

class TestCanaryWidening:
    def _canary_manager(self, clock, monkeypatch, **action_kw):
        monkeypatch.setattr(config, "CANARY_MAX_FRACTION", 0.5)
        mgr = _manager()
        action = _FakeAction(canary_fraction=0.5, **action_kw)
        mgr.register_action(action)
        return mgr, action

    def test_canary_leg_then_verify_then_widen(self, clock, monkeypatch):
        mgr, action = self._canary_manager(clock, monkeypatch)

        # 1) canary leg only
        mgr.evaluate(_pod(), None, 0.9)
        assert action.executions == 1
        assert _alerts(mgr) == ["remediation_canary"]
        canary_key = next(k for k in mgr._dedup_history if k.endswith("/canary"))
        assert mgr._dedup_history[canary_key]["verify"] == "pending"

        # 2) still verifying → no widen, no re-run (deferred-wait is silent)
        clock["now"] += 10
        mgr.evaluate(_pod(), None, 0.9)
        assert action.executions == 1
        assert _alerts(mgr) == ["remediation_canary"]

        # 3) due + improved → verify then widen in same cycle
        clock["now"] += config.VERIFY_SETTLING_S + 1
        mgr.evaluate(_recovered_pod(), None, 0.9)
        assert action.executions == 2
        assert "remediation_verified" in _alerts(mgr)
        assert "remediation_widened" in _alerts(mgr)
        widen_key = next(k for k in mgr._dedup_history if k.endswith("/widen"))
        assert mgr._dedup_history[widen_key]["verify"] == "pending"

    def test_canary_failure_blocks_widen(self, clock, monkeypatch):
        mgr, action = self._canary_manager(clock, monkeypatch)
        mgr.evaluate(_pod(), None, 0.9)
        clock["now"] += config.VERIFY_SETTLING_S + 1
        mgr.evaluate(_pod(), None, 0.9)  # canary verification fails

        assert action.executions == 1  # widen NEVER executes
        assert "remediation_widened" not in _alerts(mgr)
        assert "remediation_held" in _alerts(mgr)

    def test_single_replica_skips_canary(self, clock, monkeypatch):
        monkeypatch.setattr(config, "CANARY_MAX_FRACTION", 0.5)
        mgr = _manager()
        action = _FakeAction(canary_fraction=0.5)
        mgr.register_action(action)
        mgr.evaluate(_pod(replicas=1), None, 0.9)

        assert action.executions == 1
        assert _alerts(mgr) == []
        assert RemediationOutcome.ACTED.value in _outcomes(mgr)

    def test_canary_fraction_clamped(self):
        mgr = _manager()
        action = _FakeAction(canary_fraction=0.9)
        assert mgr._effective_canary_fraction(action) == 0.15
        action2 = _FakeAction(canary_fraction=0.05)
        assert mgr._effective_canary_fraction(action2) == 0.05
        action3 = _FakeAction(canary_fraction=0.0)
        assert mgr._effective_canary_fraction(action3) == 0.0

    def test_canary_and_base_keys_distinct(self, clock, monkeypatch):
        """Widen must not collide with the base dedup key."""
        mgr, action = self._canary_manager(clock, monkeypatch)
        mgr.evaluate(_pod(), None, 0.9)
        keys = set(mgr._dedup_history)
        assert len(keys) == 1
        assert next(iter(keys)).endswith("/canary")


# ── task 6.2: operator surfacing (unit level) ───────────────────────────────

class TestOperatorSurfacing:
    def test_get_remediation_state_shape(self, clock):
        mgr = _manager()
        action = _FakeAction()
        mgr.register_action(action)
        mgr.evaluate(_pod(), None, 0.9)
        clock["now"] += config.VERIFY_SETTLING_S + 1
        mgr.evaluate(_pod(), None, 0.9)  # → held

        state = mgr.get_remediation_state()
        assert state["verify_enabled"] is True
        assert state["settling_s"] == 90.0
        assert state["hold_s"] == 600.0
        assert state["revert_enabled"] is True
        assert state["canary_max_fraction"] == 0.15
        assert len(state["entries"]) >= 1
        entry = state["entries"][-1]
        assert entry["status"] == "held"
        assert entry["hold_remaining_s"] > 0
        assert "idempotency_key" in entry

    def test_remediation_log_rows_carry_lifecycle(self, clock):
        mgr = _manager()
        action = _FakeAction()
        mgr.register_action(action)
        mgr.evaluate(_pod(), None, 0.9)  # t0: acted
        clock["now"] += config.VERIFY_SETTLING_S + 1
        mgr.evaluate(_pod(), None, 0.9)  # fails verification → held
        # beyond dedup (300) but inside the hold → HELD result hits the log
        clock["now"] = 1_000_000 + 301
        mgr.evaluate(_pod(), None, 0.9)

        log = mgr.get_remediation_log(limit=50)
        assert any("lifecycle" in entry for entry in log)
        assert any(entry.get("lifecycle") == "held" for entry in log)
        # lifecycle mirrors outcome on the held row
        held = [e for e in log if e.get("lifecycle") == "held"][-1]
        assert held["outcome"] == "held"

    def test_approve_review_clears_hold(self, clock):
        mgr = _manager()
        action = _FakeAction(reversible=False)
        mgr.register_action(action)
        mgr.evaluate(_pod(), None, 0.9)
        clock["now"] += config.VERIFY_SETTLING_S + 1
        mgr.evaluate(_pod(), None, 0.9)  # held
        key = list(mgr._dedup_history.keys())[0]
        rec = mgr._dedup_history[key]
        assert rec["verify"] == "held"

        # simulate an operator approving the hold's signature
        mgr.approve_review(rec["sig"])
        assert key not in mgr._dedup_history
        assert mgr.verify_state[key]["status"] == "approved_retry"

        # and the action may now re-execute
        mgr.evaluate(_pod(), None, 0.9)
        assert action.executions == 2


# ── task 6.2: operator surfacing (HTTP endpoint level) ──────────────────────

class _NoCloseBytesIO(io.BytesIO):
    def close(self):
        pass


class _FakeSocket:
    def __init__(self, raw_request):
        self.rfile = io.BytesIO(raw_request)
        self.wfile = _NoCloseBytesIO()

    def makefile(self, mode, *args, **kwargs):
        return self.rfile if mode.startswith("r") else self.wfile

    def sendall(self, data):
        self.wfile.write(data)

    def shutdown(self, how):
        pass

    def close(self):
        pass


class _TestRequestHandler(server_mod.RequestHandler):
    wbufsize = 65536


def _dispatch(method: str, path: str) -> (int, dict):
    raw = f"{method} {path} HTTP/1.0\r\n\r\n".encode()
    sock = _FakeSocket(raw)
    _TestRequestHandler(sock, ("127.0.0.1", 0), mock.Mock())
    head, _, payload = sock.wfile.getvalue().partition(b"\r\n\r\n")
    status_line = head.split(b"\r\n", 1)[0].decode("utf-8", "replace")
    status = int(status_line.split(" ", 2)[1])
    return status, json.loads(payload.decode("utf-8", "replace"))


class TestRemediationStateEndpoint:
    @pytest.fixture
    def srv(self, monkeypatch):
        """Stub the module-level server context and restore after the test."""
        fresh = server_mod._ServerState()
        with monkeypatch.context() as mp:
            mp.setattr(server_mod, "_context", fresh)
            yield fresh

    def test_remediation_state_endpoint_disabled_without_manager(self, srv):
        status, body = _dispatch("GET", "/remediation/state")
        assert status == 200
        assert body["status"] == "disabled"

    def test_remediation_state_endpoint_surfaces_holds(self, srv, clock):
        mgr = _manager()
        action = _FakeAction()
        mgr.register_action(action)
        srv.remediation_manager = mgr
        mgr.evaluate(_pod(), None, 0.9)
        clock["now"] += config.VERIFY_SETTLING_S + 1
        mgr.evaluate(_pod(), None, 0.9)  # → held

        status, body = _dispatch("GET", "/remediation/state")
        assert status == 200
        assert body["verify_enabled"] is True
        assert body["min_signals"] == 2
        assert body["hold_s"] == 600.0
        assert len(body["entries"]) >= 1
        assert body["entries"][-1]["status"] == "held"
        assert body["entries"][-1]["hold_remaining_s"] > 0

    def test_remediation_log_endpoint_lifecycle_field(self, srv, clock):
        mgr = _manager()
        action = _FakeAction()
        mgr.register_action(action)
        srv.remediation_manager = mgr
        mgr.evaluate(_pod(), None, 0.9)

        status, body = _dispatch("GET", "/remediation-log")
        assert status == 200
        assert body["remediation_log"]
        entry = body["remediation_log"][-1]
        assert "lifecycle" in entry
        assert entry["lifecycle"] == entry["outcome"]
