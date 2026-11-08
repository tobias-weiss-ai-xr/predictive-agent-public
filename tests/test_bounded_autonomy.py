"""P6: bounded remediation autonomy — risk tiers, review gate, per-action timeout.

Covers the ROADMAP P6 acceptance gates:
- With REMEDIATION_REVIEW_TIER=moderate, a `scale` action is logged but NOT
  executed until approved; `node_cordon` (destructive) always requires review.
- Safe actions (container_restart / pod_restart / rollout_restart) bypass the
  moderate gate.
- Approvals are signature-scoped and expire after REMEDIATION_REVIEW_TTL_S.
- An action that exceeds REMEDIATION_TIMEOUT_S is abandoned and logged.
- Every deferral is observable: WARN log, review_required outcome in the
  remediation audit log, and the predictive_remediation_total metric.
"""
import json
import time
import urllib.error
import urllib.request

import pytest

from predictive_agent import config
from predictive_agent.remediator import (
    RemediationAction,
    RemediationManager,
    RemediationOutcome,
    RemediationResult,
    get_remediation_metrics,
    reset_remediation_metrics,
)

# ─── Test doubles ───────────────────────────────────────────────────────────

class FakeAction(RemediationAction):
    """Configurable action double with an execution counter."""

    def __init__(self, name="scale", risk_tier="moderate", eligible=True, sleep_s=0.0):
        self.name = name
        self.risk_tier = risk_tier
        self._eligible = eligible
        self._sleep_s = sleep_s
        self.executed = 0

    def should_execute(self, pod_state, prediction, risk_score):
        return self._eligible

    def execute(self, target, context):
        if self._sleep_s:
            time.sleep(self._sleep_s)
        self.executed += 1
        return RemediationResult(
            action=self.name,
            target=target,
            success=True,
            dry_run=context.dry_run,
            message="ok",
        )


class FakePod:
    def __init__(self, name="pod-1", namespace="default", node="node-1"):
        self.name = name
        self.namespace = namespace
        self.node_name = node
        self.state = "CRITICAL"
        self.cpu_trend = 0.5
        self.memory_trend = 0.5


def _manager(actions, review_tier=None, **kw):
    rem = RemediationManager(dry_run=True, risk_threshold=0.1, **kw)
    if review_tier is not None:
        rem.review_tier = review_tier
    for a in actions:
        rem.register_action(a)
    return rem


# ─── Risk tiers on the real actions ────────────────────────────────────────

class TestRiskTiers:
    def test_real_actions_declare_tiers(self):
        from predictive_agent.actions.container_restart import ContainerRestartAction
        from predictive_agent.actions.node_cordon import NodeCordonAction
        from predictive_agent.actions.pod_restart import PodRestartAction
        from predictive_agent.actions.right_size import RightSizeAction
        from predictive_agent.actions.rollout_restart import RolloutRestartAction
        from predictive_agent.actions.scale import DeploymentScaleAction
        from predictive_agent.actions.tune_resources import ResourceTunerAction

        assert ContainerRestartAction.risk_tier == "safe"
        assert PodRestartAction.risk_tier == "safe"
        assert RolloutRestartAction.risk_tier == "safe"
        assert DeploymentScaleAction.risk_tier == "moderate"
        assert RightSizeAction.risk_tier == "moderate"
        assert ResourceTunerAction.risk_tier == "moderate"
        assert NodeCordonAction.risk_tier == "destructive"

    def test_base_class_default_tier_is_moderate(self):
        assert RemediationAction.risk_tier == "moderate"

    def test_risk_tier_ordering(self):
        assert config.RISK_TIERS["safe"] < config.RISK_TIERS["moderate"]
        assert config.RISK_TIERS["moderate"] < config.RISK_TIERS["destructive"]


# ─── Review gate (reactive evaluate path) ─────────────────────────────────

class TestReviewGate:
    def setup_method(self):
        reset_remediation_metrics()

    def _rem(self, review_tier, action):
        return _manager([action], review_tier=review_tier)

    def test_moderate_gate_defers_scale_until_approved(self):
        """ROADMAP acceptance gate #1: tier=moderate → scale deferred."""
        action = FakeAction("scale", "moderate")
        rem = self._rem("moderate", action)

        results = rem.evaluate(FakePod(), None, risk_score=0.9)
        assert action.executed == 0, "scale must not execute without approval"
        assert len(results) == 1
        assert results[0].outcome == RemediationOutcome.REVIEW_REQUIRED.value
        assert "review_required" in results[0].message

        # observable: queued for review
        pending = rem.list_reviews()
        assert len(pending) == 1
        assert pending[0]["action"] == "scale"
        assert pending[0]["tier"] == "moderate"
        assert pending[0]["status"] == "pending"

        # observable: metric recorded
        assert get_remediation_metrics().get(("scale", "review_required"), 0) >= 1

        # approve → next evaluate executes
        approved = rem.approve_review(signature="default/pod-1/scale")
        assert approved["released"] == 1
        results = rem.evaluate(FakePod(), None, risk_score=0.9)
        assert action.executed == 1
        assert results[0].success is True
        assert rem.list_reviews() == []

    def test_moderate_gate_always_defers_node_cordon(self):
        """ROADMAP acceptance gate #2: destructive node_cordon always gated
        when REMEDIATION_REVIEW_TIER=moderate."""
        action = FakeAction("node_cordon", "destructive")
        rem = self._rem("moderate", action)
        results = rem.evaluate(FakePod(), None, risk_score=0.95)
        assert action.executed == 0
        assert results[0].outcome == RemediationOutcome.REVIEW_REQUIRED.value

    def test_destructive_tier_gates_only_destructive(self):
        action_scale = FakeAction("scale", "moderate")
        action_cordon = FakeAction("node_cordon", "destructive")
        rem = _manager([action_scale, action_cordon], review_tier="destructive")
        rem.evaluate(FakePod(), None, risk_score=0.9)
        assert action_scale.executed == 1, "moderate action passes destructive gate"
        assert action_cordon.executed == 0, "destructive action still gated"

    def test_safe_actions_bypass_moderate_gate(self):
        action = FakeAction("container_restart", "safe")
        rem = self._rem("moderate", action)
        results = rem.evaluate(FakePod(), None, risk_score=0.9)
        assert action.executed == 1
        assert all(
            r.outcome != RemediationOutcome.REVIEW_REQUIRED.value for r in results
        )

    def test_gate_disabled_by_default(self):
        action = FakeAction("node_cordon", "destructive")
        rem = _manager([action], review_tier=None)
        assert rem.review_tier in (None, "none")
        rem.evaluate(FakePod(), None, risk_score=0.9)
        assert action.executed == 1

    def test_approval_expires_after_ttl(self):
        action = FakeAction("scale", "moderate")
        rem = _manager([action], review_tier="moderate", review_ttl_s=0.05)
        rem.evaluate(FakePod(), None, risk_score=0.9)
        rem.approve_review(signature="default/pod-1/scale")
        time.sleep(0.1)  # approval now older than the TTL
        results = rem.evaluate(FakePod(), None, risk_score=0.9)
        assert action.executed == 0
        assert results[0].outcome == RemediationOutcome.REVIEW_REQUIRED.value

    def test_approval_via_review_id(self):
        action = FakeAction("scale", "moderate")
        rem = self._rem("moderate", action)
        rem.evaluate(FakePod(), None, risk_score=0.9)
        rid = rem.list_reviews()[0]["id"]
        assert rem.approve_review(review_id=rid)["released"] == 1
        rem.evaluate(FakePod(), None, risk_score=0.9)
        assert action.executed == 1

    def test_unknown_review_id_returns_none(self):
        rem = _manager([], review_tier="moderate")
        assert rem.approve_review(review_id="rev-999") is None

    def test_deferral_logged_at_warn(self, caplog):
        action = FakeAction("scale", "moderate")
        rem = self._rem("moderate", action)
        with caplog.at_level("WARNING"):
            rem.evaluate(FakePod(), None, risk_score=0.9)
        assert any("review_required" in r.message for r in caplog.records)

    def test_deferral_in_remediation_audit_log(self):
        """Binary observability: the deferral appears in the P2 audit log."""
        action = FakeAction("scale", "moderate")
        rem = self._rem("moderate", action)
        rem.evaluate(FakePod(), None, risk_score=0.9)
        entries = [e for e in rem.context.remediation_log]
        assert any(
            e.get("outcome") == "review_required" and e.get("action") == "scale"
            for e in entries
        )


# ─── Review gate (planner path) ────────────────────────────────────────────

class TestReviewGatePlannerPath:
    def test_plan_and_execute_honours_gate(self):
        action = FakeAction("node_cordon", "destructive")
        rem = _manager([action], review_tier="moderate")
        from predictive_agent.state_model import StateModel

        sm = StateModel()
        sm.update_pod("default", "pod-1", memory_mib=900, memory_limit_mib=1024,
                      cpu_m=900, restart_count=9, log_errors=5, node_pressure=True)
        rem.plan_remediation("CRITICAL", horizon=2)
        # The planner may or may not pick node_cordon; force the named path by
        # executing a plan step that targets it directly.
        execution = rem._apply_named_action(
            "node_cordon", "pod-1", sm.pods["default/pod-1"], None, 0.95,
            namespace="default", node_name="node-1",
        )
        assert action.executed == 0
        assert execution is not None
        assert execution.outcome == RemediationOutcome.REVIEW_REQUIRED.value

    def test_plan_and_execute_after_approval(self):
        action = FakeAction("node_cordon", "destructive")
        rem = _manager([action], review_tier="moderate")
        from predictive_agent.state_model import StateModel

        sm = StateModel()
        sm.update_pod("default", "pod-1", memory_mib=900, memory_limit_mib=1024,
                      cpu_m=900, restart_count=9, log_errors=5, node_pressure=True)
        rem._apply_named_action(
            "node_cordon", "pod-1", sm.pods["default/pod-1"], None, 0.95,
            namespace="default", node_name="node-1",
        )
        rem.approve_review(signature="default/pod-1/node_cordon")
        execution = rem._apply_named_action(
            "node_cordon", "pod-1", sm.pods["default/pod-1"], None, 0.95,
            namespace="default", node_name="node-1",
        )
        assert action.executed == 1
        assert execution.success is True


# ─── Per-action timeout ─────────────────────────────────────────────────────

class TestActionTimeout:
    def test_slow_action_times_out(self):
        slow = FakeAction("tune_resources", "moderate", sleep_s=5.0)
        rem = _manager([slow], review_tier="none", action_timeout_s=0.2)
        results = rem.evaluate(FakePod(), None, risk_score=0.9)
        assert slow.executed == 0
        assert len(results) == 1
        assert results[0].success is False
        assert "timeout" in results[0].message
        assert results[0].outcome == RemediationOutcome.FAILED.value

    def test_fast_action_within_timeout(self):
        fast = FakeAction("pod_restart", "safe")
        rem = _manager([fast], review_tier="none", action_timeout_s=30)
        results = rem.evaluate(FakePod(), None, risk_score=0.9)
        assert fast.executed == 1
        assert results[0].success is True

    def test_timeout_zero_disables_wrapper(self):
        fast = FakeAction("pod_restart", "safe")
        rem = _manager([fast], action_timeout_s=0)
        rem.evaluate(FakePod(), None, risk_score=0.9)
        assert fast.executed == 1

    def test_action_exception_surfaces_as_failed(self):
        class Exploding(FakeAction):
            def execute(self, target, context):
                raise RuntimeError("boom")

        rem = _manager([Exploding("scale", "moderate")], review_tier="none")
        results = rem.evaluate(FakePod(), None, risk_score=0.9)
        assert results[0].success is False
        assert "boom" in results[0].message


# ─── Destructive actions always warn ───────────────────────────────────────

class TestDestructiveLogging:
    def test_destructive_execution_warns_even_without_gate(self, caplog):
        action = FakeAction("node_cordon", "destructive")
        rem = _manager([action], review_tier="none")
        with caplog.at_level("WARNING"):
            rem.evaluate(FakePod(), None, risk_score=0.9)
        assert action.executed == 1
        assert any(
            "Destructive remediation" in r.message for r in caplog.records
        )


# ─── HTTP endpoints ─────────────────────────────────────────────────────────

PORT = 18097
HEALTH = 18098


# function-scoped: start_server writes the module-global _context, and
# test_get_review_without_manager below starts a bare server that nulls
# it — a module-scoped server would keep serving a stale/nulled context
# for later tests (only visible under randomized test order).
@pytest.fixture(scope="function")
def review_server():
    from predictive_agent.server import start_server

    action = FakeAction("scale", "moderate")
    rem = _manager([action], review_tier="moderate")
    # Prime a pending review so /review has something to show.
    rem.evaluate(FakePod(name="web-1"), None, risk_score=0.9)
    server = start_server(PORT, HEALTH, remediation_manager=rem)
    yield server, rem
    server.shutdown()


class TestReviewEndpoints:
    def test_get_review_lists_pending(self, review_server):
        server, rem = review_server
        with urllib.request.urlopen(f"http://localhost:{HEALTH}/review") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
        assert data["enabled"] is True
        assert data["review_tier"] == "moderate"
        assert len(data["pending"]) >= 1
        item = data["pending"][0]
        assert item["action"] == "scale"
        assert item["signature"] == "default/web-1/scale"

    def test_post_review_approves_by_id(self, review_server):
        server, rem = review_server
        rid = rem.list_reviews()[0]["id"]
        req = urllib.request.Request(
            f"http://localhost:{HEALTH}/review",
            data=json.dumps({"id": rid}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
        assert data["status"] == "ok"
        assert data["released"] == 1
        assert data["pending"] == []

    def test_post_review_unknown_id_404(self, review_server):
        server, rem = review_server
        req = urllib.request.Request(
            f"http://localhost:{HEALTH}/review",
            data=json.dumps({"id": "rev-4040"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req)
            pytest.fail("expected 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404

    def test_post_review_missing_body_fields_400(self, review_server):
        server, rem = review_server
        req = urllib.request.Request(
            f"http://localhost:{HEALTH}/review",
            data=json.dumps({}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req)
            pytest.fail("expected 400")
        except urllib.error.HTTPError as e:
            assert e.code == 400

    def test_get_review_without_manager(self):
        from predictive_agent.server import start_server

        server = start_server(18099, 18090)
        try:
            with urllib.request.urlopen("http://localhost:18090/review") as resp:
                data = json.loads(resp.read().decode())
            assert data["enabled"] is False
        finally:
            server.shutdown()
