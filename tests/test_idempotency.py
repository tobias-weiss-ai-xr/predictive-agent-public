"""Tests for P3: idempotent remediation — dedup window + idempotency keys.

Verifies that re-running the same remediation action for the same
namespace/target within REMEDIATION_DEDUP_WINDOW produces ONE execution, not
two (crash-mid-remediation safety): the second run is skipped with a
'[op] dedup: action already executed' outcome recorded in the audit/remediation
log. Also verifies the idempotency key format (namespace/pod/action/window),
that scale / tune_resources record their desired state (converge, not stack),
and that failed executions can be retried immediately.

Runs WITHOUT network / Docker / kubectl — actions that shell out are mocked.
"""

import os
from unittest.mock import MagicMock, patch

from predictive_agent import config
from predictive_agent.actions.container_restart import ContainerRestartAction
from predictive_agent.actions.node_cordon import NodeCordonAction
from predictive_agent.actions.pod_restart import PodRestartAction
from predictive_agent.actions.right_size import RightSizeAction
from predictive_agent.actions.rollout_restart import RolloutRestartAction
from predictive_agent.actions.scale import DeploymentScaleAction
from predictive_agent.actions.tune_resources import ResourceTunerAction
from predictive_agent.remediator import (
    ActionContext,
    RemediationAction,
    RemediationManager,
    RemediationOutcome,
    RemediationResult,
    SafetyPolicy,
    create_remediation_manager_from_config,
    get_remediation_metrics,
    reset_remediation_metrics,
)


def _no_cooldown_policy():
    """Safety policy that never rate-limits/cooldowns, isolating the dedup gate."""
    return SafetyPolicy(cooldown_seconds=0, max_per_minute=1000, max_per_hour=100000)


def _counting_action(counter: dict, action_name: str = "test_action", is_idempotent: bool = False):
    """An action that always executes and counts how often it really runs."""
    class CountingAction(RemediationAction):
        name = action_name
        idempotent = is_idempotent

        def should_execute(self, pod_state, prediction, risk_score):
            return True

        def execute(self, target, context):
            counter["executions"] += 1
            return RemediationResult(
                action=self.name,
                target=target,
                success=True,
                dry_run=context.dry_run,
                message=f"executed {target}",
                desired_state={"replicas": 2},
            )

    return CountingAction()


def _pod(name="pod-1", namespace="default"):
    p = MagicMock()
    p.name = name
    p.namespace = namespace
    p.node_name = "node-1"
    p.state = "CRITICAL"
    return p


class TestDedupWindowConfig:
    """REMEDIATION_DEDUP_WINDOW wiring (config + manager + factory)."""

    def test_config_defines_dedup_window(self):
        assert isinstance(config.REMEDIATION_DEDUP_WINDOW, int)
        assert config.REMEDIATION_DEDUP_WINDOW > 0
        assert int(os.environ.get("REMEDIATION_DEDUP_WINDOW", "300")) == config.REMEDIATION_DEDUP_WINDOW

    def test_manager_default_window_is_300(self, monkeypatch):
        monkeypatch.delenv("REMEDIATION_DEDUP_WINDOW", raising=False)
        mgr = create_remediation_manager_from_config()
        assert mgr.dedup_window == 300

    def test_manager_env_override(self):
        with patch.dict(os.environ, {"REMEDIATION_DEDUP_WINDOW": "120"}):
            mgr = create_remediation_manager_from_config()
            assert mgr.dedup_window == 120

    def test_explicit_constructor_window_wins(self):
        mgr = RemediationManager(dedup_window=7)
        assert mgr.dedup_window == 7


class TestIdempotencyKeys:
    """Idempotency key = namespace/pod/action/timestamp-window."""

    def test_key_format(self):
        mgr = RemediationManager(dedup_window=300)
        key = mgr.idempotency_key("default", "pod-1", "tune_resources")
        assert key == "default/pod-1/tune_resources/w300"

    def test_key_distinguishes_targets(self):
        mgr = RemediationManager(dedup_window=300)
        assert mgr.idempotency_key("ns", "a", "scale") != mgr.idempotency_key("ns", "b", "scale")

    def test_key_distinguishes_namespaces(self):
        mgr = RemediationManager(dedup_window=300)
        assert mgr.idempotency_key("ns1", "a", "scale") != mgr.idempotency_key("ns2", "a", "scale")

    def test_key_distinguishes_actions(self):
        mgr = RemediationManager(dedup_window=300)
        assert mgr.idempotency_key("ns", "a", "scale") != mgr.idempotency_key("ns", "a", "pod_restart")

    def test_key_embeds_window(self):
        short = RemediationManager(dedup_window=300).idempotency_key("ns", "a", "s")
        long = RemediationManager(dedup_window=600).idempotency_key("ns", "a", "s")
        assert short != long

    def test_executed_result_carries_idempotency_key(self):
        counter = {"executions": 0}
        mgr = RemediationManager(
            dry_run=True, risk_threshold=0.0,
            safety_policy=_no_cooldown_policy(), dedup_window=300,
        )
        mgr.register_action(_counting_action(counter))
        results = mgr.evaluate(_pod(), None, 0.9)
        assert results[0].idempotency_key == "default/pod-1/test_action/w300"


class TestDedupExecution:
    """Re-running the same remediation within the window => one execution."""

    def setup_method(self):
        reset_remediation_metrics()

    def test_second_evaluation_within_window_is_deduplicated(self):
        counter = {"executions": 0}
        mgr = RemediationManager(
            dry_run=True, risk_threshold=0.0,
            safety_policy=_no_cooldown_policy(), dedup_window=300,
        )
        mgr.register_action(_counting_action(counter))
        pod = _pod()

        first = mgr.evaluate(pod, None, 0.9)
        second = mgr.evaluate(pod, None, 0.9)

        assert counter["executions"] == 1
        assert first[0].get_outcome() == RemediationOutcome.ACTED.value
        assert second[0].get_outcome() == RemediationOutcome.SKIPPED.value
        assert "[op] dedup" in second[0].message
        assert "action already executed" in second[0].message
        # Binary-observability metrics: one acted, one skipped.
        assert get_remediation_metrics().get(("test_action", "acted"), 0) == 1
        assert get_remediation_metrics().get(("test_action", "skipped"), 0) == 1

    def test_dedup_skip_recorded_in_remediation_log(self):
        counter = {"executions": 0}
        mgr = RemediationManager(
            dry_run=True, risk_threshold=0.0,
            safety_policy=_no_cooldown_policy(), dedup_window=300,
        )
        mgr.register_action(_counting_action(counter))
        pod = _pod()

        mgr.evaluate(pod, None, 0.9)
        mgr.evaluate(pod, None, 0.9)

        log = mgr.get_remediation_log(limit=10)
        assert len(log) == 2
        assert log[0]["outcome"] == "acted"
        assert log[1]["outcome"] == "skipped"
        assert "dedup" in log[1]["message"]

    def test_dedup_window_expiry_allows_reexecution(self, monkeypatch):
        import predictive_agent.remediator as rem_mod
        from predictive_agent import config as _cfg

        # Pure P3 dedup mechanics: verification off so the settling window
        # doesn't fire a hold mid-test (the closed loop has its own suite).
        monkeypatch.setattr(_cfg, "VERIFY_ENABLED", False)
        clock = {"now": 1000.0}
        monkeypatch.setattr(rem_mod, "_now", lambda: clock["now"])

        counter = {"executions": 0}
        mgr = RemediationManager(
            dry_run=True, risk_threshold=0.0,
            safety_policy=_no_cooldown_policy(), dedup_window=300,
        )
        mgr.register_action(_counting_action(counter))
        pod = _pod()

        mgr.evaluate(pod, None, 0.9)
        assert counter["executions"] == 1

        # Still inside the 300s window -> deduplicated.
        clock["now"] = 1200.0
        second = mgr.evaluate(pod, None, 0.9)
        assert counter["executions"] == 1
        assert second[0].get_outcome() == RemediationOutcome.SKIPPED.value

        # Window expired (t >= 1300) -> action runs again.
        clock["now"] = 1400.0
        third = mgr.evaluate(pod, None, 0.9)
        assert counter["executions"] == 2
        assert third[0].get_outcome() == RemediationOutcome.ACTED.value

    def test_different_targets_not_deduplicated(self):
        counter = {"executions": 0}
        mgr = RemediationManager(
            dry_run=True, risk_threshold=0.0,
            safety_policy=_no_cooldown_policy(), dedup_window=300,
        )
        mgr.register_action(_counting_action(counter))

        mgr.evaluate(_pod("pod-1"), None, 0.9)
        mgr.evaluate(_pod("pod-2"), None, 0.9)
        assert counter["executions"] == 2

    def test_different_namespaces_not_deduplicated(self):
        counter = {"executions": 0}
        mgr = RemediationManager(
            dry_run=True, risk_threshold=0.0,
            safety_policy=_no_cooldown_policy(), dedup_window=300,
        )
        mgr.register_action(_counting_action(counter))

        mgr.evaluate(_pod(name="pod-1", namespace="a"), None, 0.9)
        mgr.evaluate(_pod(name="pod-1", namespace="b"), None, 0.9)
        assert counter["executions"] == 2

    def test_different_actions_not_deduplicated(self):
        counter = {"executions": 0}
        mgr = RemediationManager(
            dry_run=True, risk_threshold=0.0,
            safety_policy=_no_cooldown_policy(), dedup_window=300,
        )
        mgr.register_action(_counting_action(counter, action_name="scale"))
        mgr.register_action(_counting_action(counter, action_name="tune_resources"))

        pod = _pod()
        mgr.evaluate(pod, None, 0.9)
        mgr.evaluate(pod, None, 0.9)
        # Each action has its own idempotency key: on the first pass both run
        # (1 execution each); on the second pass both are deduplicated against
        # their own history — total executions stays one per action.
        assert counter["executions"] == 2
        acted = [r for r in mgr.get_remediation_log(limit=10) if r["outcome"] == "acted"]
        skipped = [r for r in mgr.get_remediation_log(limit=10) if r["outcome"] == "skipped"]
        assert {r["action"] for r in acted} == {"scale", "tune_resources"}
        assert {r["action"] for r in skipped} == {"scale", "tune_resources"}

    def test_failed_execution_does_not_block_retry(self):
        state = {"fail": True, "executions": 0}

        class FlakyAction(RemediationAction):
            name = "flaky"
            idempotent = False

            def should_execute(self, pod_state, prediction, risk_score):
                return True

            def execute(self, target, context):
                state["executions"] += 1
                if state["fail"]:
                    state["fail"] = False
                    return RemediationResult(
                        action=self.name, target=target, success=False,
                        dry_run=context.dry_run, message="error: boom",
                    )
                return RemediationResult(
                    action=self.name, target=target, success=True,
                    dry_run=context.dry_run, message="ok",
                )

        mgr = RemediationManager(
            dry_run=True, risk_threshold=0.0,
            safety_policy=_no_cooldown_policy(), dedup_window=300,
        )
        mgr.register_action(FlakyAction())
        pod = _pod()

        first = mgr.evaluate(pod, None, 0.9)
        second = mgr.evaluate(pod, None, 0.9)
        assert state["executions"] == 2  # failure did NOT populate the dedup history
        assert first[0].get_outcome() == RemediationOutcome.FAILED.value
        assert second[0].get_outcome() == RemediationOutcome.ACTED.value

    def test_planner_path_deduplicates(self):
        counter = {"executions": 0}
        mgr = RemediationManager(
            dry_run=True, risk_threshold=0.0,
            safety_policy=_no_cooldown_policy(), dedup_window=300,
        )
        mgr.register_action(_counting_action(counter))
        pod = _pod()

        first = mgr._apply_named_action("test_action", "pod-1", pod, None, 0.9, "default", "")
        second = mgr._apply_named_action("test_action", "pod-1", pod, None, 0.9, "default", "")

        assert counter["executions"] == 1
        assert first.success is True
        assert second.success is False
        assert "[op] dedup" in second.message

    def test_dedup_counts_and_stats(self):
        counter = {"executions": 0}
        mgr = RemediationManager(
            dry_run=True, risk_threshold=0.0,
            safety_policy=_no_cooldown_policy(), dedup_window=300,
        )
        mgr.register_action(_counting_action(counter))
        pod = _pod()

        mgr.evaluate(pod, None, 0.9)
        mgr.evaluate(pod, None, 0.9)

        stats = mgr.get_stats()
        assert stats["dedup_window"] == 300
        assert stats["deduped_actions"] == 1
        assert stats["outcomes"]["skipped"] == 1
        assert stats["outcomes"]["acted"] == 1


class TestDesiredState:
    """scale / tune_resources record desired state => re-runs converge."""

    def test_scale_records_desired_state(self):
        ctx = ActionContext(dry_run=True, namespace="default")
        ctx.scale_direction = "up"
        ctx.current_replicas = 2

        with patch("predictive_agent.actions.scale.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="scaled", stderr="")
            result = DeploymentScaleAction().execute("backend", ctx)

        # Closed loop: desired_state also carries prev_replicas so a failed
        # verification can revert to the exact prior replica count.
        assert result.desired_state == {"replicas": 3, "prev_replicas": 2}

    def test_tune_resources_records_desired_state(self):
        ctx = ActionContext(dry_run=True, namespace="default")
        ctx.cpu_request = "100m"
        ctx.mem_request = "128Mi"
        ctx.cpu_limit = "200m"
        ctx.mem_limit = "256Mi"

        with patch("predictive_agent.actions.tune_resources.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="patched", stderr="")
            result = ResourceTunerAction().execute("backend", ctx)

        assert result.desired_state == {
            "cpu_request": "100m",
            "mem_request": "128Mi",
            "cpu_limit": "200m",
            "mem_limit": "256Mi",
        }

    def test_dedup_history_keeps_desired_state_and_flags_it(self):
        counter = {"executions": 0}
        mgr = RemediationManager(
            dry_run=True, risk_threshold=0.0,
            safety_policy=_no_cooldown_policy(), dedup_window=300,
        )
        mgr.register_action(_counting_action(counter))
        pod = _pod()

        mgr.evaluate(pod, None, 0.9)
        key = mgr.idempotency_key("default", "pod-1", "test_action")
        rec = mgr._dedup_history[key]
        assert rec["desired_state"] == {"replicas": 2}
        assert rec["idempotent"] is False

        # The dedup skip surfaces the already-applied desired state.
        second = mgr.evaluate(pod, None, 0.9)
        assert "desired_state={'replicas': 2}" in second[0].message


class TestIdempotentFlags:
    """Each action declares whether re-running it is safe (P3)."""

    def test_idempotent_actions(self):
        assert PodRestartAction().idempotent is True
        assert NodeCordonAction().idempotent is True
        assert ContainerRestartAction().idempotent is True

    def test_non_idempotent_actions(self):
        assert RolloutRestartAction().idempotent is False
        assert DeploymentScaleAction().idempotent is False
        assert ResourceTunerAction().idempotent is False
        assert RightSizeAction().idempotent is False

    def test_execution_records_idempotent_flag(self):
        counter = {"executions": 0}
        mgr = RemediationManager(
            dry_run=True, risk_threshold=0.0,
            safety_policy=_no_cooldown_policy(), dedup_window=300,
        )
        mgr.register_action(_counting_action(counter, is_idempotent=True))
        mgr.evaluate(_pod(), None, 0.9)
        key = mgr.idempotency_key("default", "pod-1", "test_action")
        assert mgr._dedup_history[key]["idempotent"] is True
