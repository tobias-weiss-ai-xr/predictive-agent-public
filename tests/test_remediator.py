"""Tests for the remediation framework (REM-1, P2 Binary Observability)."""

import os
import time
from unittest.mock import MagicMock, patch

import pytest

from predictive_agent.remediator import (
    ActionContext,
    RemediationAction,
    RemediationManager,
    RemediationOutcome,
    RemediationResult,
    SafetyPolicy,
    create_remediation_manager_from_config,
    get_remediation_metrics,
    record_remediation_metric,
    reset_remediation_metrics,
)


class TestRemediationActionABC:
    """Test that RemediationAction is an abstract base class."""

    def test_cannot_instantiate_abc_directly(self):
        """RemediationAction is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            RemediationAction()

    def test_subclass_must_implement_methods(self):
        """Subclass must implement should_execute and execute."""

        class IncompleteAction(RemediationAction):
            name = "incomplete"

        with pytest.raises(TypeError):
            IncompleteAction()

    def test_subclass_with_implementations_works(self):
        """A complete subclass can be instantiated."""

        class TestAction(RemediationAction):
            name = "test"
            def should_execute(self, pod_state, prediction, risk_score):
                return risk_score > 50
            def execute(self, target, context):
                return RemediationResult(
                    action=self.name, target=target, success=True,
                    dry_run=context.dry_run, message="test action"
                )

        action = TestAction()
        assert action.name == "test"
        assert action.should_execute(None, None, 60) is True
        assert action.should_execute(None, None, 40) is False


class TestActionContext:
    """Test ActionContext defaults and behavior."""

    def test_defaults(self):
        ctx = ActionContext()
        assert ctx.namespace == "opendesk-predictive-agent"
        assert ctx.dry_run is True
        assert ctx.audit_log == []
        assert ctx.cooldowns == {}
        assert ctx.rate_limiter == {}
        assert ctx.node_name == ""
        assert ctx.pod_phase == "Running"
        assert ctx.cpu_trend == 0.0
        assert ctx.mem_trend == 0.0

    def test_custom_values(self):
        ctx = ActionContext(namespace="custom", dry_run=False)
        assert ctx.namespace == "custom"
        assert ctx.dry_run is False


class TestRemediationResult:
    """Test RemediationResult dataclass."""

    def test_fields(self):
        result = RemediationResult(
            action="pod_restart", target="pod-1", success=True,
            dry_run=False, message="deleted", command="kubectl delete pod pod-1",
            outcome="acted"
        )
        assert result.action == "pod_restart"
        assert result.target == "pod-1"
        assert result.success is True
        assert result.dry_run is False
        assert result.message == "deleted"
        assert result.command == "kubectl delete pod pod-1"
        assert result.outcome == "acted"
        assert result.timestamp  # auto-generated

    def test_timestamp_auto_generated(self):
        result = RemediationResult(
            action="test", target="t", success=True, dry_run=True, message="m"
        )
        assert result.timestamp is not None
        assert "T" in result.timestamp  # ISO format

    def test_get_outcome_acted(self):
        result = RemediationResult(
            action="restart", target="pod-1", success=True, dry_run=False, message="restarted"
        )
        assert result.get_outcome() == RemediationOutcome.ACTED.value

    def test_get_outcome_skipped(self):
        result = RemediationResult(
            action="restart", target="pod-1", success=False, dry_run=True, message="skipped: protected namespace"
        )
        assert result.get_outcome() == RemediationOutcome.SKIPPED.value

    def test_get_outcome_failed(self):
        result = RemediationResult(
            action="restart", target="pod-1", success=False, dry_run=False, message="error: timeout"
        )
        assert result.get_outcome() == RemediationOutcome.FAILED.value

    def test_get_outcome_explicit(self):
        result = RemediationResult(
            action="restart", target="pod-1", success=True, dry_run=False,
            message="restarted", outcome="acted"
        )
        assert result.get_outcome() == "acted"


class TestSafetyPolicy:
    """Test SafetyPolicy rate limiting, cooldowns, and protected namespaces."""

    def test_protected_namespace_blocked(self):
        policy = SafetyPolicy(protected_namespaces={"kube-system"})
        ctx = ActionContext()
        can, reason = policy.can_execute("pod-1", "pod_restart", ctx, "kube-system")
        assert can is False
        assert "protected" in reason

    def test_non_protected_namespace_allowed(self):
        policy = SafetyPolicy(protected_namespaces={"kube-system"})
        ctx = ActionContext()
        can, reason = policy.can_execute("pod-1", "pod_restart", ctx, "default")
        assert can is True
        assert reason == "ok"

    def test_cooldown_blocks_repeated_action(self):
        policy = SafetyPolicy(cooldown_seconds=60)
        ctx = ActionContext()
        # First action: allowed
        can1, _ = policy.can_execute("pod-1", "pod_restart", ctx)
        assert can1 is True
        policy.record_action("pod-1", "pod_restart", ctx)
        # Second action immediately: blocked by cooldown
        can2, reason2 = policy.can_execute("pod-1", "pod_restart", ctx)
        assert can2 is False
        assert "cooldown" in reason2

    def test_cooldown_expires(self):
        policy = SafetyPolicy(cooldown_seconds=1)
        ctx = ActionContext()
        policy.record_action("pod-1", "pod_restart", ctx)
        # Wait for cooldown to expire
        time.sleep(1.1)
        can, reason = policy.can_execute("pod-1", "pod_restart", ctx)
        assert can is True

    def test_per_minute_rate_limit(self):
        policy = SafetyPolicy(max_per_minute=3)
        ctx = ActionContext()
        for i in range(3):
            can, _ = policy.can_execute(f"pod-{i}", "pod_restart", ctx)
            assert can is True
            policy.record_action(f"pod-{i}", "pod_restart", ctx)
        # 4th action: blocked
        can, reason = policy.can_execute("pod-4", "pod_restart", ctx)
        assert can is False
        assert "rate limit" in reason.lower()

    def test_per_hour_rate_limit(self):
        policy = SafetyPolicy(max_per_hour=2)
        ctx = ActionContext()
        for i in range(2):
            can, _ = policy.can_execute(f"pod-{i}", "pod_restart", ctx)
            assert can is True
            policy.record_action(f"pod-{i}", "pod_restart", ctx)
        # 3rd action: blocked
        can, reason = policy.can_execute("pod-3", "pod_restart", ctx)
        assert can is False
        assert "rate limit" in reason.lower()

    def test_different_targets_not_blocked_by_cooldown(self):
        policy = SafetyPolicy(cooldown_seconds=60)
        ctx = ActionContext()
        policy.record_action("pod-1", "pod_restart", ctx)
        # Different target: not blocked
        can, _ = policy.can_execute("pod-2", "pod_restart", ctx)
        assert can is True

    def test_different_action_types_not_blocked_by_cooldown(self):
        policy = SafetyPolicy(cooldown_seconds=60)
        ctx = ActionContext()
        policy.record_action("pod-1", "pod_restart", ctx)
        # Different action type: not blocked
        can, _ = policy.can_execute("pod-1", "node_cordon", ctx)
        assert can is True

    def test_default_protected_namespaces(self):
        policy = SafetyPolicy()
        assert "kube-system" in policy.protected_namespaces
        assert "opendesk-predictive-agent" in policy.protected_namespaces


class TestRemediationManager:
    """Test RemediationManager registration, evaluation, and audit trail."""

    def setup_method(self):
        """Reset metrics before each test."""
        reset_remediation_metrics()

    def _make_action(self, name="test_action", should_exec=True, exec_result=None):
        class TestAction(RemediationAction):
            def __init__(self):
                self.name = name
            def should_execute(self, pod_state, prediction, risk_score):
                return should_exec
            def execute(self, target, context):
                if exec_result:
                    return exec_result
                return RemediationResult(
                    action=self.name, target=target, success=True,
                    dry_run=context.dry_run, message="executed", outcome="acted"
                )
        return TestAction()

    def test_register_action(self):
        manager = RemediationManager()
        action = self._make_action()
        manager.register_action(action)
        assert len(manager.actions) == 1
        assert manager.actions[0].name == "test_action"

    def test_evaluate_below_threshold(self):
        manager = RemediationManager(risk_threshold=0.7)
        manager.register_action(self._make_action())
        # Risk below threshold: no actions
        results = manager.evaluate(None, None, 0.5)
        assert results == []
        # Verify no metrics recorded for below-threshold decisions
        metrics = get_remediation_metrics()
        assert len(metrics) == 0

    def test_evaluate_above_threshold(self):
        manager = RemediationManager(risk_threshold=0.7)
        manager.register_action(self._make_action())
        results = manager.evaluate(None, None, 0.8)
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].get_outcome() == RemediationOutcome.ACTED.value
        # Verify metric was recorded
        metrics = get_remediation_metrics()
        assert metrics.get(("test_action", "acted"), 0) == 1

    def test_evaluate_action_not_triggered(self):
        manager = RemediationManager(risk_threshold=0.7)
        manager.register_action(self._make_action(should_exec=False))
        results = manager.evaluate(None, None, 0.8)
        assert len(results) == 1  # should_execute=False now produces a skipped entry
        assert results[0].get_outcome() == RemediationOutcome.SKIPPED.value
        assert "should_execute returned False" in results[0].message
        # Verify metric was recorded
        metrics = get_remediation_metrics()
        assert metrics.get(("test_action", "skipped"), 0) == 1

    def test_audit_trail(self):
        manager = RemediationManager(risk_threshold=0.7)
        manager.register_action(self._make_action())
        manager.evaluate(None, None, 0.8)
        assert len(manager.audit_trail) == 1
        assert manager.audit_trail[0].action == "test_action"
        assert manager.audit_trail[0].get_outcome() == RemediationOutcome.ACTED.value

    def test_dry_run_mode(self):
        manager = RemediationManager(dry_run=True, risk_threshold=0.7)
        manager.register_action(self._make_action())
        results = manager.evaluate(None, None, 0.8)
        assert results[0].dry_run is True
        assert results[0].get_outcome() == RemediationOutcome.ACTED.value
        # Verify metric was recorded
        metrics = get_remediation_metrics()
        assert metrics.get(("test_action", "acted"), 0) == 1

    def test_get_audit_trail_limit(self):
        # Use a policy with no cooldown to allow multiple actions on different targets
        policy = SafetyPolicy(cooldown_seconds=0, max_per_minute=100, max_per_hour=1000)
        manager = RemediationManager(risk_threshold=0.0, safety_policy=policy)
        manager.register_action(self._make_action())
        for i in range(25):
            manager.evaluate(MagicMock(name=f"pod-{i}", namespace="default"), None, 0.8)
        trail = manager.get_audit_trail(limit=10)
        assert len(trail) == 10

    def test_get_remediation_log(self):
        """Test structured remediation log for binary observability."""
        policy = SafetyPolicy(cooldown_seconds=0, max_per_minute=100, max_per_hour=1000)
        manager = RemediationManager(risk_threshold=0.0, safety_policy=policy)
        manager.register_action(self._make_action())
        # Execute some actions
        manager.evaluate(MagicMock(name="pod-1", namespace="default"), None, 0.8)
        manager.evaluate(MagicMock(name="pod-2", namespace="monitoring"), None, 0.8)

        remediation_log = manager.get_remediation_log()
        assert len(remediation_log) == 2
        # Check first entry has required fields for binary observability
        assert "action" in remediation_log[0]
        assert "target" in remediation_log[0]
        assert "namespace" in remediation_log[0]
        assert "outcome" in remediation_log[0]
        assert remediation_log[0]["outcome"] in ["acted", "skipped", "failed"]

    def test_remediation_log_limit(self):
        """Test remediation log respects limit."""
        policy = SafetyPolicy(cooldown_seconds=0, max_per_minute=100, max_per_hour=1000)
        manager = RemediationManager(risk_threshold=0.0, safety_policy=policy)
        manager.register_action(self._make_action())
        for i in range(15):
            manager.evaluate(MagicMock(name=f"pod-{i}", namespace="default"), None, 0.8)

        remediation_log = manager.get_remediation_log(limit=10)
        assert len(remediation_log) == 10

    def test_get_stats(self):
        manager = RemediationManager(dry_run=True, risk_threshold=0.7)
        manager.register_action(self._make_action())
        manager.evaluate(None, None, 0.8)
        stats = manager.get_stats()
        assert stats["total_actions"] == 1
        assert stats["successful_actions"] == 1
        assert stats["dry_run_actions"] == 1
        assert "test_action" in stats["registered_actions"]
        # Check outcomes breakdown
        assert "outcomes" in stats
        assert stats["outcomes"]["acted"] == 1
        assert stats["outcomes"]["skipped"] == 0
        assert stats["outcomes"]["failed"] == 0

    def test_evaluate_exception_handling(self):
        manager = RemediationManager(risk_threshold=0.7)

        class FailingAction(RemediationAction):
            name = "failing"
            def should_execute(self, pod_state, prediction, risk_score):
                return True
            def execute(self, target, context):
                raise RuntimeError("boom")

        manager.register_action(FailingAction())
        results = manager.evaluate(None, None, 0.8)
        assert len(results) == 1
        assert results[0].success is False
        assert "boom" in results[0].message
        assert results[0].get_outcome() == RemediationOutcome.FAILED.value
        # Verify metric was recorded
        metrics = get_remediation_metrics()
        assert metrics.get(("failing", "failed"), 0) == 1

    def test_action_receives_pods_namespace_and_node(self):
        """Regression: actions must receive the pod's actual namespace and
        node_name, NOT the operator's namespace."""
        captured = {}

        class CaptureAction(RemediationAction):
            name = "capture"
            def should_execute(self, pod_state, prediction, risk_score):
                return True
            def execute(self, target, context):
                captured["namespace"] = context.namespace
                captured["node_name"] = context.node_name
                captured["pod_phase"] = context.pod_phase
                return RemediationResult(
                    action=self.name, target=target, success=True,
                    dry_run=context.dry_run, message="captured",
                )

        manager = RemediationManager(risk_threshold=0.7)
        manager.register_action(CaptureAction())

        pod_state = MagicMock()
        pod_state.name = "mariadb-galera-1"
        pod_state.namespace = "opendesk"
        pod_state.node_name = "clrz14-07"
        pod_state.state = "STRESSED"
        pod_state.cpu_trend = 5.0
        pod_state.memory_trend = 2.0

        manager.evaluate(pod_state, None, 0.8)
        assert captured["namespace"] == "opendesk"
        assert captured["namespace"] != "opendesk-predictive-agent"
        assert captured["node_name"] == "clrz14-07"
        assert captured["pod_phase"] == "STRESSED"

    def test_action_receives_default_namespace_when_pod_has_none(self):
        """When the pod state has no namespace, fall back gracefully."""
        captured = {}

        class CaptureAction(RemediationAction):
            name = "capture"
            def should_execute(self, pod_state, prediction, risk_score):
                return True
            def execute(self, target, context):
                captured["namespace"] = context.namespace
                return RemediationResult(
                    action=self.name, target=target, success=True,
                    dry_run=context.dry_run, message="captured",
                )

        manager = RemediationManager(risk_threshold=0.7)
        manager.register_action(CaptureAction())
        manager.evaluate(None, None, 0.8)
        assert captured["namespace"] in ("", "opendesk-predictive-agent")


class TestCreateFromConfig:
    """Test create_remediation_manager_from_config()."""

    def test_defaults(self):
        reset_remediation_metrics()
        with patch.dict(os.environ, {
            "REMEDIATION_ENABLED": "false",
            "REMEDIATION_DRY_RUN": "true",
            "REMEDIATION_MAX_PER_MIN": "5",
            "REMEDIATION_MAX_PER_HOUR": "50",
            "REMEDIATION_COOLDOWN_S": "300",
            "REMEDIATION_RISK_THRESHOLD": "0.7",
            "REMEDIATION_PROTECTED_NS": "kube-system,opendesk-predictive-agent",
        }):
            manager = create_remediation_manager_from_config()
            assert manager.dry_run is True
            assert manager.risk_threshold == 0.7
            assert "kube-system" in manager.safety_policy.protected_namespaces


class TestRemediationMetrics:
    """Test Prometheus metrics for binary observability."""

    def setup_method(self):
        """Reset metrics before each test."""
        reset_remediation_metrics()

    def test_record_metric_acted(self):
        record_remediation_metric("pod_restart", RemediationOutcome.ACTED)
        metrics = get_remediation_metrics()
        assert metrics.get(("pod_restart", "acted"), 0) == 1

    def test_record_metric_skipped(self):
        record_remediation_metric("pod_restart", RemediationOutcome.SKIPPED)
        metrics = get_remediation_metrics()
        assert metrics.get(("pod_restart", "skipped"), 0) == 1

    def test_record_metric_failed(self):
        record_remediation_metric("node_cordon", RemediationOutcome.FAILED)
        metrics = get_remediation_metrics()
        assert metrics.get(("node_cordon", "failed"), 0) == 1

    def test_multiple_records_same_action_outcome(self):
        record_remediation_metric("container_restart", RemediationOutcome.ACTED)
        record_remediation_metric("container_restart", RemediationOutcome.ACTED)
        metrics = get_remediation_metrics()
        assert metrics.get(("container_restart", "acted"), 0) == 2

    def test_multiple_actions_outcomes(self):
        record_remediation_metric("pod_restart", RemediationOutcome.ACTED)
        record_remediation_metric("pod_restart", RemediationOutcome.SKIPPED)
        record_remediation_metric("node_cordon", RemediationOutcome.FAILED)
        metrics = get_remediation_metrics()
        assert len(metrics) == 3
        assert metrics.get(("pod_restart", "acted"), 0) == 1
        assert metrics.get(("pod_restart", "skipped"), 0) == 1
        assert metrics.get(("node_cordon", "failed"), 0) == 1

    def test_reset_metrics(self):
        record_remediation_metric("test", RemediationOutcome.ACTED)
        assert len(get_remediation_metrics()) == 1
        reset_remediation_metrics()
        assert len(get_remediation_metrics()) == 0


class TestOutcomeEnum:
    """Test RemediationOutcome enum."""

    def test_enum_values(self):
        assert RemediationOutcome.ACTED.value == "acted"
        assert RemediationOutcome.SKIPPED.value == "skipped"
        assert RemediationOutcome.FAILED.value == "failed"

    def test_enum_members(self):
        assert "ACTED" in RemediationOutcome.__members__
        assert "SKIPPED" in RemediationOutcome.__members__
        assert "FAILED" in RemediationOutcome.__members__

    def test_enabled_with_dry_run_false(self):
        with patch.dict(os.environ, {
            "REMEDIATION_ENABLED": "true",
            "REMEDIATION_DRY_RUN": "false",
            "REMEDIATION_MAX_PER_MIN": "5",
            "REMEDIATION_MAX_PER_HOUR": "50",
            "REMEDIATION_COOLDOWN_S": "300",
            "REMEDIATION_RISK_THRESHOLD": "0.7",
            "REMEDIATION_PROTECTED_NS": "kube-system",
        }):
            manager = create_remediation_manager_from_config()
            assert manager.dry_run is False

    def test_disabled_forces_dry_run(self):
        with patch.dict(os.environ, {
            "REMEDIATION_ENABLED": "false",
            "REMEDIATION_DRY_RUN": "false",
            "REMEDIATION_MAX_PER_MIN": "5",
            "REMEDIATION_MAX_PER_HOUR": "50",
            "REMEDIATION_COOLDOWN_S": "300",
            "REMEDIATION_RISK_THRESHOLD": "0.7",
            "REMEDIATION_PROTECTED_NS": "kube-system",
        }):
            manager = create_remediation_manager_from_config()
            assert manager.dry_run is True  # forced dry_run when disabled


class TestNamedActionSafetyGates(TestRemediationManager):
    """Pins for _apply_named_action's guard chain (planner path).

    Mutation testing (mutmut, remediator campaign) showed the planner-path
    safety gates had no test executing them: the SafetyPolicy veto, the
    per-action eligibility guard, and the dedup skip all produced mutant
    survivors that changed what actually executes on the cluster.
    """

    def _manager(self, **kw):
        policy = SafetyPolicy(cooldown_seconds=0, max_per_minute=100, max_per_hour=1000)
        manager = RemediationManager(risk_threshold=0.0, safety_policy=policy, **kw)
        manager.register_action(self._make_action())
        return manager

    def _apply(self, manager, target="pod-1", namespace="default", pod_state=None):
        return manager._apply_named_action(
            action_name="test_action",
            target=target,
            pod_state=pod_state or MagicMock(),
            prediction=None,
            risk_score=0.9,
            namespace=namespace,
            node_name="node-a",
        )

    def test_safety_policy_veto_produces_skipped_result_and_audit(self):
        """A protected-namespace veto on the planner path must skip + audit."""
        manager = self._manager()
        result = self._apply(manager, namespace="kube-system")
        assert result is not None and result.success is False
        assert "skipped" in result.message
        assert result.outcome == "skipped"
        assert len(manager.audit_trail) == 1
        assert manager.context.audit_log[-1]["action"] == "test_action"
        assert manager.context.audit_log[-1]["success"] is False

    def test_ineligible_action_skipped_without_execution(self):
        """should_execute=False must skip BEFORE the executor runs."""
        manager = RemediationManager(risk_threshold=0.0)
        manager.register_action(self._make_action(should_exec=False))
        result = self._apply(manager)
        assert result.success is False
        assert result.message == "skipped: action ineligible"
        assert result.outcome == "skipped"

    def test_dedup_second_execution_within_window_is_skipped(self):
        """Same ns/target/action inside REMEDIATION_DEDUP_WINDOW runs once."""
        manager = self._manager()
        first = self._apply(manager)
        assert first.success is True
        second = self._apply(manager)
        assert second.success is False
        assert second.outcome == "skipped"
        assert "already executed" in second.message
        assert second.idempotency_key == first.idempotency_key

    def test_dedup_expiry_allows_rerun(self):
        """Outside the window the stale record is pruned and the action reruns."""
        manager = self._manager()
        first = self._apply(manager)
        key = first.idempotency_key
        manager._dedup_history[key]["timestamp"] -= manager.dedup_window + 1
        second = self._apply(manager)
        assert second.success is True
        assert second.idempotency_key == key  # same key, stale record pruned

    def test_unregistered_action_returns_none(self):
        manager = self._manager()
        assert manager._apply_named_action(
            action_name="nope", target="pod-1", pod_state=MagicMock(),
            prediction=None, risk_score=0.9, namespace="default", node_name="n",
        ) is None


# ── Nightly-campaign survivor killers (2026-09-17) ──────────────────────────
# The first real mutation-campaign run left 398 remediator survivors. These
# tests pin the classes that were genuine tested-behavior gaps: exact audit-log
# key contracts on every gate path, planner wiring/arg fidelity, P(FAILED)
# reduction bookkeeping, timeout-result fields, observation-learning arg
# passthrough, and the review/dedup record shapes. Accepted survivors (not
# killed here): docstring mutants, logger-format strings, and equivalent
# mutants (truthiness-only sums, getattr defaults never exercised).

from types import SimpleNamespace as _NS

from predictive_agent import config as _config
from predictive_agent.pomdp import ActionAwareTransitionModel, MarkovChain


def _plan_step(action, pf=1.0):
    return _NS(action=action, expected_failed_prob_after=pf)


class _FakePlan:
    def __init__(self, steps):
        self.steps = steps

    def to_dict(self):
        return {"steps": [{"action": s.action} for s in self.steps]}


def _fake_planner_cls(recorder, plan):
    class FakePlanner:
        def __init__(self, **kw):
            recorder.append(kw)

        def plan(self, current_state, horizon):
            recorder.append((current_state, horizon))
            return plan

    return FakePlanner


AUDIT_KEYS = {"action", "target", "success", "dry_run", "message", "timestamp", "command"}


class TestMutationSurvivorKillers(TestNamedActionSafetyGates):
    """Kill classes from the first nightly mutation run (remediator campaign)."""

    def _action(self, name="act", should_exec=True, result=None,
                on_execute=None, risk_tier=None, idempotent=None):
        class T(RemediationAction):
            pass
        if risk_tier is not None:
            T.risk_tier = risk_tier
        if idempotent is not None:
            T.idempotent = idempotent
        T.__abstractmethods__ = frozenset()  # methods assigned below

        def __init__(s):
            s.name = name
        T.__init__ = __init__

        def should_execute(s, pod_state, prediction, risk_score):
            return should_exec
        T.should_execute = should_execute

        def execute(s, target, context):
            if on_execute is not None:
                on_execute(target, context)
            if result is not None:
                return result
            return RemediationResult(
                action=name, target=target, success=True,
                dry_run=context.dry_run, message="executed",
                outcome="acted", command=f"cmd-{name}",
            )
        T.execute = execute
        return T()

    def _apply(self, manager, target="pod-1", namespace="default", pod_state=None, node_name="node-a"):
        return manager._apply_named_action(
            action_name="test_action", target=target,
            pod_state=pod_state if pod_state is not None else _NS(),
            prediction=None, risk_score=0.9, namespace=namespace,
            node_name=node_name)

    def _fresh_manager(self, **kw):
        """Manager with permissive policy and NO default registered action."""
        policy = SafetyPolicy(cooldown_seconds=0, max_per_minute=100, max_per_hour=1000)
        kw.setdefault("risk_threshold", 0.0)
        kw.setdefault("safety_policy", policy)
        return RemediationManager(**kw)

    def _planner_manager(self, plan, **kw):
        """Manager + started Planner patch (caller must exit the patcher)."""
        rec = []
        manager = RemediationManager(
            risk_threshold=kw.pop("risk_threshold", 0.0),
            decision_model=kw.pop("model", None),
            action_costs=kw.pop("action_costs", {"act": 1.0}),
            failure_cost=kw.pop("failure_cost", 0.4), **kw)
        manager.evaluate = MagicMock(return_value=[_NS(success=False)])
        patcher = patch("predictive_agent.pomdp.Planner", _fake_planner_cls(rec, plan))
        return manager, rec, patcher

    # ── audit-log dict contract on every _apply_named_action path ──────────

    def test_audit_log_exact_keys_on_all_gate_paths(self):
        cases = []
        manager = self._manager()
        result = self._apply(manager, namespace="kube-system")
        cases.append(("veto", manager, result))
        manager = RemediationManager(risk_threshold=0.0)
        manager.register_action(self._action(name="test_action", should_exec=False))
        cases.append(("ineligible", manager, self._apply(manager)))
        manager = self._manager()
        self._apply(manager)
        cases.append(("dedup", manager, self._apply(manager)))
        manager = RemediationManager(risk_threshold=0.0, review_tier="moderate")
        manager.register_action(self._action(name="test_action", risk_tier="moderate"))
        cases.append(("gated", manager, self._apply(manager)))
        manager = self._manager()
        cases.append(("success", manager, self._apply(manager)))

        for path, manager, result in cases:
            entry = manager.context.audit_log[-1]
            # Closed loop: the planner success path now records the outcome too
            # (unified with the gated path for binary observability).
            expected = AUDIT_KEYS | ({"outcome"} if path in ("gated", "success") else set())
            assert set(entry.keys()) == expected, f"path={path} keys={sorted(entry)}"
            assert entry["action"] == result.action, path
            assert entry["target"] == result.target, path
            assert entry["success"] == result.success, path
            assert entry["dry_run"] == result.dry_run, path
            assert entry["message"] == result.message, path
            assert entry["timestamp"] == result.timestamp, path
            assert entry["command"] == result.command, path

    def test_audit_trail_holds_result_identity(self):
        manager = self._manager()
        first = self._apply(manager)
        second = self._apply(manager)  # dedup-skipped
        assert manager.audit_trail[0] is first
        assert manager.audit_trail[-1] is second
        assert second.action == "test_action"
        assert second.target == "pod-1"
        assert second.dry_run == manager.dry_run

    def test_skipped_paths_report_metric_with_action_name(self):
        with patch("predictive_agent.remediator.record_remediation_metric") as metric:
            manager = self._manager()
            manager.register_action(self._action(name="test_action"))
            self._apply(manager, namespace="kube-system")
            manager2 = RemediationManager(risk_threshold=0.0)
            manager2.register_action(self._action(name="test_action", should_exec=False))
            self._apply(manager2)
            assert metric.call_args_list[0][0] == ("test_action", RemediationOutcome.SKIPPED)
            assert metric.call_args_list[1][0] == ("test_action", RemediationOutcome.SKIPPED)

    def test_context_wired_from_pod_state(self):
        seen = []

        def capture(target, context):
            seen.append((context.namespace, context.node_name, context.pod_phase,
                         context.cpu_trend, context.mem_trend))
        pod = _NS(state="DEGRADED", cpu_trend=2.5, memory_trend=1.5)
        manager = self._fresh_manager()
        manager.register_action(self._action(name="test_action", on_execute=capture))
        self._apply(manager, pod_state=pod, namespace="ns1", node_name="n1")
        assert seen[-1] == ("ns1", "n1", "DEGRADED", 2.5, 1.5)
        self._apply(manager, pod_state=_NS(), namespace="ns2", node_name="n2")
        assert seen[-1] == ("ns2", "n2", "Running", 0.0, 0.0)

    # ── dedup execution record shape ────────────────────────────────────────

    def test_execution_record_shape(self):
        manager = self._fresh_manager()
        res = RemediationResult(
            action="test_action", target="pod-1", success=True,
            dry_run=True, message="executed", command="kubectl scale",
            desired_state={"replicas": "3"})
        manager.register_action(self._action(name="test_action", result=res))
        out = self._apply(manager)
        rec = manager._dedup_history[out.idempotency_key]
        assert set(rec.keys()) == {
            "timestamp", "message", "command", "desired_state", "idempotent",
            # verify-after-act: signal snapshot + verification lifecycle.
            "snapshot", "verify", "recorded_at", "sig", "leg",
        }
        assert rec["message"] == "executed"
        assert rec["command"] == "kubectl scale"
        assert rec["desired_state"] == {"replicas": "3"}
        assert rec["idempotent"] is True
        assert rec["verify"] == "pending"
        assert rec["leg"] == "full"
        assert rec["snapshot"].keys() == {
            "state", "memory_pct", "cpu_pct", "restart_count", "log_errors", "node_pressure",
        }

    def test_execution_record_non_idempotent_action(self):
        manager = self._fresh_manager()
        manager.register_action(self._action(name="test_action", idempotent=False))
        out = self._apply(manager)
        assert manager._dedup_history[out.idempotency_key]["idempotent"] is False

    # ── execution timeout ───────────────────────────────────────────────────

    def test_timeout_abandons_with_failed_result(self):
        def slow(target, context):
            time.sleep(0.5)
        manager = RemediationManager(risk_threshold=0.0, action_timeout_s=0.15)
        manager.register_action(self._action(name="test_action", on_execute=slow))
        result = self._apply(manager)
        assert result.success is False
        assert result.outcome == "failed"
        assert result.action == "test_action"
        assert result.target == "pod-1"
        assert result.dry_run == manager.context.dry_run
        assert "timeout" in result.message

    def test_no_timeout_runs_inline(self):
        manager = RemediationManager(risk_threshold=0.0, action_timeout_s=0)
        manager.register_action(self._action(name="test_action"))
        assert self._apply(manager).success is True

    # ── planner path: gates, wiring, reduction bookkeeping ──────────────────

    def test_confidence_gate_returns_empty_without_fallback(self):
        manager, _, patcher = self._planner_manager(_FakePlan([]))
        manager.evaluate = MagicMock()
        with patcher, patch.object(_config, "MIN_ACTION_CONFIDENCE", 0.9):
            out = manager.plan_and_execute(
                "DEGRADED", "pod-1", _NS(), _NS(confidence=0.1), 0.9)
        assert out == []
        manager.evaluate.assert_not_called()

    def test_model_support_fallback_passes_args_to_evaluate(self):
        model = MagicMock()
        model.is_confident.return_value = False
        model.state_support.return_value = 0.2
        manager = RemediationManager(risk_threshold=0.0, decision_model=model)
        sentinel = [_NS(success=False)]
        manager.evaluate = MagicMock(return_value=sentinel)
        with patch.object(_config, "MIN_MODEL_SUPPORT", 0.5):
            pod, pred = _NS(), _NS()
            out = manager.plan_and_execute("DEGRADED", "pod-1", pod, pred, 0.9)
        assert out is sentinel
        manager.evaluate.assert_called_once_with(pod, pred, 0.9)
        model.state_support.assert_called_with("DEGRADED")

    def test_planning_disabled_and_degraded_fallbacks(self):
        manager, _, patcher = self._planner_manager(_FakePlan([]))
        sentinel = [_NS(success=False)]
        manager.evaluate = MagicMock(return_value=sentinel)
        pod, pred = _NS(), _NS()
        with patcher, patch.object(_config, "PLANNING_ENABLED", False):
            out = manager.plan_and_execute("DEGRADED", "p", pod, pred, 0.9)
        assert out is sentinel
        manager.evaluate.assert_called_once_with(pod, pred, 0.9)
        manager2, _, patcher2 = self._planner_manager(_FakePlan([]))
        manager2.evaluate = MagicMock(return_value=sentinel)
        manager2.degraded = True
        out = manager2.plan_and_execute("DEGRADED", "p", pod, pred, 0.9)
        assert out is sentinel
        manager2.evaluate.assert_called_once_with(pod, pred, 0.9)

    def test_planner_constructed_with_model_and_costs(self):
        model = ActionAwareTransitionModel(base_chain=MarkovChain())
        rec = []
        manager = RemediationManager(
            risk_threshold=0.0, decision_model=model,
            action_costs={"act": 2.0}, failure_cost=0.6)
        manager.evaluate = MagicMock(return_value=[])
        with patch("predictive_agent.pomdp.Planner", _fake_planner_cls(rec, _FakePlan([]))):
            manager.plan_and_execute("CRITICAL", "p", _NS(), _NS(), 0.9, horizon=4)
        assert rec[0] == {"model": model, "action_costs": {"act": 2.0}, "failure_cost": 0.6}
        assert rec[1] == ("CRITICAL", 4)

    def test_plan_steps_skips_none_and_applies_rest(self):
        plan = _FakePlan([_plan_step("none"), _plan_step("act")])
        manager, _, patcher = self._planner_manager(plan)
        ok = RemediationResult(action="act", target="pod-1", success=True,
                               dry_run=True, message="ok", outcome="acted")
        apply_spy = MagicMock(return_value=ok)
        manager._apply_named_action = apply_spy
        manager.interventions_executed = 5
        pod, pred = _NS(state="DEGRADED"), _NS()
        with patcher:
            results = manager.plan_and_execute(
                "DEGRADED", "pod-1", pod, pred, 0.9, namespace="ns", node_name="n1")
        assert results == [ok]
        assert manager.interventions_executed == 6
        args = apply_spy.call_args.args
        assert args[0] == "act"
        assert args[1] == "pod-1"
        assert args[2] is pod
        assert args[3] is pred
        assert args[4] == 0.9
        assert args[5] == "ns"
        assert args[6] == "n1"
        assert apply_spy.call_args.kwargs.get("from_state") == "DEGRADED"

    def test_from_state_falls_back_to_current_state(self):
        plan = _FakePlan([_plan_step("none"), _plan_step("act")])
        manager, _, patcher = self._planner_manager(plan)
        apply_spy = MagicMock(return_value=None)
        manager._apply_named_action = apply_spy
        with patcher:
            manager.plan_and_execute("STRESSED", "p", _NS(), _NS(), 0.9)
        assert apply_spy.call_args.kwargs.get("from_state") == "STRESSED"

    def test_failed_only_plan_skips_reduction_bookkeeping(self):
        plan = _FakePlan([_plan_step("act", pf=0.5)])
        model = ActionAwareTransitionModel(base_chain=MarkovChain())
        manager, _, patcher = self._planner_manager(plan, model=model)
        manager.register_action(self._action(name="act", result=RemediationResult(
            action="act", target="p", success=False, dry_run=True,
            message="failed", outcome="failed")))
        manager.cumulative_p_failed_reduction = 5.0
        with patcher:
            results = manager.plan_and_execute("CRITICAL", "p", _NS(), _NS(), 0.9)
        assert len(results) == 1 and results[0].success is False
        assert manager.cumulative_p_failed_reduction == 5.0

    def test_reduction_bookkeeping_exact_delta(self):
        model = ActionAwareTransitionModel(base_chain=MarkovChain())
        none_calls = []
        orig_step = model.step_distribution

        def counting_step(d, a):
            none_calls.append(a)
            return orig_step(d, a)
        model.step_distribution = counting_step
        none_dist = {s: 0.0 for s in MarkovChain.STATES}
        none_dist["CRITICAL"] = 1.0
        for _ in range(3):
            none_dist = model.step_distribution(none_dist, "none")
        none_pf = model.failed_probability(none_dist)
        assert 0.0 < none_pf < 1.0  # distinguishes max(1.0, ...) / max(0.0, ...) mutants
        expected_delta = none_pf - 0.04  # plan pf = none_pf - 0.04 -> delta 0.04
        assert 0.0 < expected_delta < 1.0  # distinguishes max(1.0, ...) / max(0.0, ...) mutants
        plan = _FakePlan([_plan_step("none"), _plan_step("act", pf=none_pf - 0.04)])
        manager, _, patcher = self._planner_manager(plan, model=model)
        manager.register_action(self._action(name="act"))
        manager.cumulative_p_failed_reduction = 5.0
        none_calls.clear()  # drop the expectation-computation calls
        with patcher:
            results = manager.plan_and_execute("CRITICAL", "p", _NS(), _NS(), 0.9, horizon=3)
        assert any(r.success for r in results)
        assert none_calls == ["none", "none", "none"]
        assert manager.cumulative_p_failed_reduction == pytest.approx(5.04)

    def test_visibility_none_decision_entry(self):
        manager, _, patcher = self._planner_manager(_FakePlan([_plan_step("none")]))
        manager.risk_threshold = 0.7
        with patcher:
            results = manager.plan_and_execute("CRITICAL", "pod-9", _NS(), _NS(), 0.7)
        assert len(results) == 1
        entry = results[0]
        assert entry.action == "none"
        assert entry.success is False
        assert entry.target == "pod-9"
        assert entry.dry_run == manager.dry_run
        assert entry.message == (
            "planner: no beneficial action \u2014 restart yields no durable "
            "P(FAILED) reduction"
        )
        assert manager.interventions_executed == 0

    def test_circuit_breaker_one_still_reports_none_decision(self):
        plan = _FakePlan([_plan_step("none"), _plan_step("none")])
        manager, _, patcher = self._planner_manager(plan)
        manager.risk_threshold = 0.7
        with patcher:
            results = manager.plan_and_execute(
                "CRITICAL", "p", _NS(), _NS(), 0.9, max_actions=1)
        assert len(results) == 1 and results[0].action == "none"

    def test_circuit_breaker_zero_suppresses_visibility_entry(self):
        plan = _FakePlan([_plan_step("none"), _plan_step("none")])
        manager, _, patcher = self._planner_manager(plan)
        manager.risk_threshold = 0.7
        with patcher:
            results = manager.plan_and_execute(
                "CRITICAL", "p", _NS(), _NS(), 0.9, max_actions=0)
        assert results == []

    def test_review_gate_defers_and_queues(self):
        manager = RemediationManager(risk_threshold=0.0, review_tier="moderate")
        manager.register_action(self._action(name="test_action", risk_tier="moderate"))
        result = self._apply(manager)
        assert result.success is False
        assert result.outcome == "review_required"
        assert "review_required" in result.message
        assert manager.reviews_deferred == 1
        rid = list(manager.review_queue)[0]
        item = manager.review_queue[rid]
        assert set(item.keys()) == {
            "id", "signature", "action", "target", "namespace",
            "tier", "risk_score", "status", "created_at"}
        assert item["status"] == "pending"
        assert item["action"] == "test_action"

    def test_plan_remediation_wiring_and_horizon(self):
        rec = []
        manager = RemediationManager(
            risk_threshold=0.0, action_costs={"a": 1.0}, failure_cost=0.25)
        with patch("predictive_agent.remediator.Planner",
                   _fake_planner_cls(rec, _FakePlan([]))):
            out = manager.plan_remediation("STRESSED", horizon=5)
            assert out == {"steps": []}
            manager.plan_remediation("STRESSED", horizon=None)
        assert rec[0] == {
            "model": manager.decision_model, "action_costs": {"a": 1.0},
            "failure_cost": 0.25}
        assert rec[1] == ("STRESSED", 5)
        assert rec[3] == ("STRESSED", _config.PLANNING_HORIZON)

    # ── learning loop ────────────────────────────────────────────────────────

    def test_record_observations_args_and_calibration_gates(self):
        manager = RemediationManager(risk_threshold=0.0)
        manager.feed_transition = MagicMock()
        manager.calibration_monitor = MagicMock()
        manager.rl_policy = MagicMock()
        manager._pending_feedback["pod"] = [("DEGRADED", "restart_pod", 0.7)]
        manager.record_observations({"pod": "CRITICAL"})
        manager.feed_transition.assert_called_once_with("DEGRADED", "restart_pod", "CRITICAL")
        manager.calibration_monitor.record.assert_called_once_with(0.7, True)
        manager.rl_policy.update.assert_called_once_with("DEGRADED", "restart_pod", "CRITICAL")
        assert "pod" not in manager._pending_feedback

    def test_record_observations_failed_states_membership(self):
        for observed, expected in [("FAILED", True), ("CRITICAL", True), ("DEGRADED", False)]:
            manager = RemediationManager(risk_threshold=0.0)
            manager.calibration_monitor = MagicMock()
            manager.rl_policy = MagicMock()
            manager._pending_feedback["pod"] = [("DEGRADED", "restart_pod", 0.7)]
            manager.record_observations({"pod": observed})
            manager.calibration_monitor.record.assert_called_once_with(0.7, expected)

    def test_record_observations_skips_calibration_without_prediction(self):
        manager = RemediationManager(risk_threshold=0.0)
        manager.calibration_monitor = MagicMock()
        manager.rl_policy = MagicMock()
        manager._pending_feedback["pod"] = [("DEGRADED", "restart_pod", None)]
        manager.record_observations({"pod": "FAILED"})
        manager.calibration_monitor.record.assert_not_called()
        manager.rl_policy.update.assert_called_once()

    def test_record_observations_without_monitor_or_rl(self):
        manager = RemediationManager(risk_threshold=0.0)
        manager._pending_feedback["pod"] = [("DEGRADED", "restart_pod", 0.7)]
        manager.record_observations({"pod": "FAILED"})  # must not raise
        assert "pod" not in manager._pending_feedback

    def test_record_observations_continues_past_empty_pending(self):
        manager = RemediationManager(risk_threshold=0.0)
        manager.feed_transition = MagicMock()
        manager._pending_feedback["ghost"] = []
        manager._pending_feedback["real"] = [("FAILED", "scale", 0.5)]
        manager.record_observations({"ghost": "RECOVERED", "real": "RECOVERED"})
        manager.feed_transition.assert_called_once_with("FAILED", "scale", "RECOVERED")

    def test_feed_transition_passthrough(self):
        model = MagicMock()
        manager = RemediationManager(risk_threshold=0.0, decision_model=model)
        manager.feed_transition("DEGRADED", "restart_pod", "FAILED")
        model.record_transition.assert_called_once_with("DEGRADED", "FAILED", "restart_pod")

    def test_record_pending_gates_and_payload(self):
        manager = RemediationManager(risk_threshold=0.0)
        manager._record_pending("p", "DEGRADED", "restart_pod", 0.7)
        assert manager._pending_feedback["p"] == [("DEGRADED", "restart_pod", 0.7)]
        manager._record_pending("p2", "", "restart_pod", 0.7)
        manager._record_pending("p3", "DEGRADED", "none", 0.7)
        assert "p2" not in manager._pending_feedback
        assert "p3" not in manager._pending_feedback

    def test_note_remediation_outcome_clears_and_escalates(self):
        manager = RemediationManager(risk_threshold=0.0)
        manager._futile_cycles["p"] = 2
        manager.escalations = 5
        escalated, n = manager.note_remediation_outcome("p", at_risk=False, actions_taken=["none"])
        assert (escalated, n) == (False, 0)
        assert "p" not in manager._futile_cycles
        escalated = False
        for _ in range(_config.CRASHLOOP_ESCALATE_AFTER):
            escalated, n = manager.note_remediation_outcome("q", at_risk=True, actions_taken=["none"])
        assert escalated is True and n == _config.CRASHLOOP_ESCALATE_AFTER
        assert manager.escalations == 6

    def test_approve_review_by_id_shape(self):
        manager = RemediationManager(risk_threshold=0.0)
        manager.review_queue["rev-1"] = {"signature": "ns/p/act", "status": "pending"}
        out = manager.approve_review(review_id="rev-1")
        assert out == {"signature": "ns/p/act", "released": 1}
        assert manager.review_queue["rev-1"]["status"] == "approved"

    def test_clear_pending_drops_feedback(self):
        manager = RemediationManager(risk_threshold=0.0)
        manager._pending_feedback["p"] = [("A", "act", 0.5)]
        manager.clear_pending("p")
        assert "p" not in manager._pending_feedback
        manager.clear_pending("missing")  # must not raise

    def test_refine_from_backtester_passthrough(self):
        model = MagicMock()
        model.refine_from_backtester.return_value = 7
        manager = RemediationManager(risk_threshold=0.0, decision_model=model)
        assert manager.refine_from_backtester("bt") == 7
        model.refine_from_backtester.assert_called_once_with("bt")

    # ── reactive-path field pins (evaluate) ──────────────────────────────────

    def test_evaluate_result_fields_and_context_wiring(self):
        manager = RemediationManager(risk_threshold=0.7)
        manager.register_action(self._action(name="test_action"))
        pod = _NS(name="pod-1", namespace="default", state="DEGRADED",
                  cpu_trend=1.0, memory_trend=1.25, node_name="n")
        results = manager.evaluate(pod, None, 0.8)
        assert results[0].action == "test_action"
        assert results[0].outcome == "acted"
        assert manager.context.mem_trend == 1.25
        assert manager.context.cpu_trend == 1.0
        manager.evaluate(_NS(name="p2", namespace=""), None, 0.8)
        assert manager.context.mem_trend == 0.0
        assert manager.context.pod_phase == "Running"

    def test_safety_policy_rate_windows_prune_stale_entries(self):
        policy = SafetyPolicy(cooldown_seconds=0, max_per_minute=100, max_per_hour=1000)
        context = ActionContext(dry_run=True)
        context.rate_limiter["minute"] = [time.time() - 60]
        context.rate_limiter["hour"] = [time.time() - 3600]
        policy.record_action("p", "test_action", context)
        assert len(context.rate_limiter["minute"]) == 1
        assert len(context.rate_limiter["hour"]) == 1
        assert context.cooldowns["test_action:p"] > 0
