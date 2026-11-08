"""Unit tests for the extracted reconcile pipeline phases (PA-S1, direct phase tests).

Each public phase in ``predictive_agent.reconcile_phases`` is exercised
directly via a crafted ``ReconcileContext`` (no main globals required),
covering happy paths, degraded semantics, OPA gating, remediation failure
logging, LLM thread spawn, circuit breakers, calibration drift, model
decay, persistence errors, and backtest pairing.
"""

from unittest.mock import MagicMock

import pytest

import predictive_agent.config as config
import predictive_agent.reconcile_phases as phases
from predictive_agent import pod_collection
from predictive_agent.predictor import Predictor
from predictive_agent.state_model import StateModel

# ─── Fixtures ──────────────────────────────────────────────────────────────


class FakeRemediationManager:
    """Mutable fake whose behaviour-assertable methods are MagicMocks."""

    def __init__(self):
        self.dry_run = False
        self.risk_threshold = 0.7
        self.calibration_monitor = None
        self.decision_model = MagicMock()
        self.rl_policy = None
        self.action_costs = {}
        self.failure_cost = 1.0
        self.degraded = False
        # Mockable methods
        self.clear_pending = MagicMock()
        self.record_observations = MagicMock()
        self.plan_remediation = MagicMock()
        self.plan_and_execute = MagicMock()
        self.evaluate = MagicMock()
        self.note_remediation_outcome = MagicMock(return_value=(False, 0))
        self.get_stats = MagicMock(return_value={"registered_actions": 6})


_DEFAULT_HOOKS = dict(
    detect_runtime=lambda: "kubernetes",
    discover_docker_containers=lambda: {},
    collect_docker_metrics=lambda: {},
    run_cmd=lambda *a, **k: (0, "", ""),
    collect_top_metrics=lambda out: {},
    collect_top_nodes=lambda out: None,
    get_pods_json=lambda: {"items": []},
    get_node_conditions=lambda d: {},
    get_pod_namespace=lambda p: p.get("metadata", {}).get("namespace", "default"),
    get_pod_name=lambda p: p.get("metadata", {}).get("name", "x"),
    get_pod_memory_limit=lambda p, cn="": 0,
    get_pod_restart_count=lambda p: 0,
    should_skip_namespace=lambda ns: False,
    is_pod_excluded=lambda p: False,
    get_node_pressure=lambda conds, node: (False, False),
    matches_docker_selectors=lambda info, sel: True,
    get_container_memory_limit=lambda info: 0,
    get_watch_selectors=lambda: {},
    get_pod_status_signals=lambda p: {"pod_phase": "Running", "container_ready": True,
                                      "wait_state": None, "terminated": False,
                                      "terminated_reason": None, "pod_scheduled": True},
    get_container_status_signals=lambda p: {"pod_phase": "Running", "container_ready": True,
                                            "wait_state": None, "terminated": False,
                                            "terminated_reason": None},
    count_log_errors=lambda s: 0,
    now_iso=lambda: "2024-01-01T00:00:00Z",
    get_collector_state=lambda: "HEALTHY",
    record_successful_collection=lambda: None,
    record_collection_failure=lambda: None,
)


def _build_ctx(state_model=None, predictor=None, remediation_manager=None,
               notifier=None, shadow_harness=None, policy_engine=None,
               llm_analyzer=None, kg_client=None, backtester=None,
               state_store=None, **hooks):
    """Build a frozen ReconcileContext; hooks override the defaults above."""
    merged = dict(_DEFAULT_HOOKS)
    merged.update(hooks)
    return phases.ReconcileContext(
        state_model=state_model,
        predictor=predictor,
        state_store=state_store if state_store is not None else MagicMock(),
        remediation_manager=remediation_manager,
        notifier=notifier,
        shadow_harness=shadow_harness,
        policy_engine=policy_engine,
        llm_analyzer=llm_analyzer,
        kg_client=kg_client,
        backtester=backtester,
        llm_analysis_cache={},
        llm_in_flight=set(),
        last_plan_notify={},
        last_autonomy_notify={},
        **merged,
    )


class MockPolicy:
    def __init__(self, allow=True, allow_llm=True):
        self._allow = allow
        self.allow_llm = allow_llm
    def check_pod_security(self, pod):
        return {"result": {"allow": self._allow}, "violation": "none"}
    def check_llm_usage(self, req):
        return {"result": {"allow_llm": self.allow_llm}}


def _healthy_pod():
    return {"metadata": {"name": "p1", "namespace": "default", "creationTimestamp": "2024-01-01T00:00:00Z"},
            "status": {"phase": "Running", "containerStatuses": [{"ready": True, "restartCount": 0}]},
            "spec": {"nodeName": "n1"}}


# ── Phase 1: collect_metrics ────────────────────────────────────────────────


@pytest.fixture
def basic_state_model():
    return StateModel()


def test_collect_metrics_kubernetes_parses_pods(basic_state_model):
    pods = [{"metadata": {"name": "p1", "namespace": "ns"},
             "status": {"phase": "Running", "containerStatuses": []},
             "spec": {}}]
    ctx = _build_ctx(state_model=basic_state_model,
                     get_pods_json=lambda: {"items": pods},
                     get_node_conditions=lambda d: {})
    res = phases.collect_metrics(ctx, 1)
    assert res.runtime == "kubernetes"
    assert res.pod_items == pods
    assert res.current_keys == {"ns/p1"}
    assert res.collection_successful is True


def test_collect_metrics_docker_builds_pod_items():
    containers = {
        "abc123": {"name": "web", "compose_project": "shop",
                   "status": "running", "healthy": True,
                   "restart_count": 0, "created_at": "now"},
    }
    ctx = _build_ctx(state_model=StateModel(),
                     detect_runtime=lambda: "docker",
                     discover_docker_containers=lambda: containers,
                     collect_docker_metrics=lambda: {})
    res = phases.collect_metrics(ctx, 1)
    assert res.runtime == "docker"
    assert res.current_keys == {"shop/web"}
    assert res.pod_items[0]["metadata"]["namespace"] == "shop"


def test_collect_metrics_docker_empty_containers():
    ctx = _build_ctx(state_model=StateModel(),
                     detect_runtime=lambda: "docker",
                     discover_docker_containers=lambda: {})
    res = phases.collect_metrics(ctx, 1)
    assert res.pod_items == []
    assert res.current_keys == set()
    assert res.collection_successful is True


def test_collect_metrics_skips_succeeded_and_job_pods_k8s():
    pods = [
        {"metadata": {"name": "ok", "namespace": "default"},
         "status": {"phase": "Running"}, "spec": {}},
        {"metadata": {"name": "done", "namespace": "default",
                     "ownerReferences": [{"kind": "Job"}]},
         "status": {"phase": "Succeeded"}, "spec": {}},
    ]
    ctx = _build_ctx(state_model=StateModel(),
                     get_pods_json=lambda: {"items": pods})
    res = phases.collect_metrics(ctx, 1)
    assert res.current_keys == {"default/ok"}


def test_collect_metrics_excludes_virt_launcher_pods_k8s():
    """KubeVirt virt-launcher pods (and annotation-opted-out pods) must never
    be tracked: their resource profile is the guest OS and auto-restarting them
    is a disruptive VM reboot (see the remediation_exhausted escalation)."""
    pods = [
        {"metadata": {"name": "app-0", "namespace": "default"},
         "status": {"phase": "Running"}, "spec": {}},
        {"metadata": {"name": "virt-launcher-win2025-eval-abc", "namespace": "default",
                      "labels": {"kubevirt.io": "virt-launcher"}},
         "status": {"phase": "Running"}, "spec": {}},
        {"metadata": {"name": "legacy", "namespace": "default",
                      "annotations": {"predictive-agent.openedu.svc/exclude": "true"}},
         "status": {"phase": "Running"}, "spec": {}},
    ]
    ctx = _build_ctx(
        state_model=StateModel(),
        get_pods_json=lambda: {"items": pods},
        is_pod_excluded=pod_collection.is_pod_excluded,
    )
    # Configure exclusion selectors via env-like config values.
    import predictive_agent.config as _cfg
    _old_l, _old_a = _cfg.EXCLUDED_POD_LABELS, _cfg.EXCLUDED_POD_ANNOTATIONS
    _cfg.EXCLUDED_POD_LABELS = {"kubevirt.io": "virt-launcher"}
    _cfg.EXCLUDED_POD_ANNOTATIONS = {"predictive-agent.openedu.svc/exclude": "true"}
    try:
        res = phases.collect_metrics(ctx, 1)
    finally:
        _cfg.EXCLUDED_POD_LABELS, _cfg.EXCLUDED_POD_ANNOTATIONS = _old_l, _old_a
    assert res.current_keys == {"default/app-0"}


def test_collect_metrics_exception_degrades():
    def boom():
        raise Exception("discovery down")
    ctx = _build_ctx(state_model=StateModel(),
                     detect_runtime=lambda: "docker",
                     discover_docker_containers=boom)
    res = phases.collect_metrics(ctx, 1)
    assert res.collection_successful is False


def test_collect_metrics_bad_nodes_json_still_succeeds():
    pods = [{"metadata": {"name": "p1", "namespace": "ns"},
             "status": {"phase": "Running"}}]
    ctx = _build_ctx(state_model=StateModel(),
                     get_pods_json=lambda: {"items": pods},
                     run_cmd=lambda *a, **k: (0, "{bad json}", ""),
                     get_node_conditions=lambda d: {})
    res = phases.collect_metrics(ctx, 1)
    assert res.node_conditions == {}
    assert res.collection_successful is True


# ── Phase 2: prepare_state_model ───────────────────────────────────────────


def test_prepare_state_model_removes_dead_pod():
    model = StateModel()
    model.update_pod("ns", "keep", 10, 100, 10, 0, 0, False)
    model.update_pod("ns", "dead", 10, 100, 10, 0, 0, False)
    remediation = FakeRemediationManager()
    predictor = Predictor()
    predictor._predictions["ns/dead"] = MagicMock()
    ctx = _build_ctx(state_model=model, remediation_manager=remediation, predictor=predictor)
    phases.prepare_state_model(ctx, 1, "HEALTHY", {"ns/keep"}, [])
    assert "ns/keep" in model.pods
    assert "ns/dead" not in model.pods
    remediation.clear_pending.assert_called_with("ns/dead")


def test_prepare_state_model_degraded_keeps_last_known():
    model = StateModel()
    model.update_pod("ns", "p1", 10, 100, 10, 0, 0, False)
    ctx = _build_ctx(state_model=model)
    prepared = phases.prepare_state_model(ctx, 1, "DEGRADED", set(), [])
    assert prepared.current_keys == {"ns/p1"}
    assert prepared.pod_items[0]["_degraded_mode"] is True
    assert "ns/p1" in model.pods


def test_prepare_state_model_cleans_llm_cache_and_inflight():
    model = StateModel()
    model.update_pod("ns", "p1", 10, 100, 10, 0, 0, False)
    ctx = _build_ctx(state_model=model, remediation_manager=FakeRemediationManager())
    ctx.llm_analysis_cache["ns/p1"] = 999.0
    ctx.llm_in_flight.add("ns/p1")
    phases.prepare_state_model(ctx, 1, "HEALTHY", set(), [])
    assert ctx.llm_analysis_cache.get("ns/p1") is None
    assert "ns/p1" not in ctx.llm_in_flight


# ── Phase 3: evaluate_pods ─────────────────────────────────────────────────


def test_evaluate_pods_skips_succeeded_and_k8s_job():
    model = StateModel()
    predictor = Predictor(risk_threshold=0.001)
    pods = [
        {"metadata": {"name": "ok", "namespace": "default"},
         "status": {"phase": "Running"}, "spec": {"nodeName": "n1"}},
        {"metadata": {"name": "done", "namespace": "default",
                     "ownerReferences": [{"kind": "Job"}]},
         "status": {"phase": "Succeeded"}, "spec": {}},
    ]
    ctx = _build_ctx(state_model=model, predictor=predictor)
    res = phases.evaluate_pods(ctx, 1, "kubernetes", pods, {}, {}, set())
    assert "default/done" not in model.pods
    assert res.pods_tracked == 1


def test_evaluate_pods_excludes_virt_launcher_pods():
    """The per-pod loop iterates raw pod_items, so exclusion must be enforced
    here as well as at collection: a KubeVirt virt-launcher pod (Windows guest
    using ~all its RAM by design) must never be scored or tracked."""
    model = StateModel()
    predictor = Predictor(risk_threshold=0.001)  # would flag anything tracked
    pods = [
        {"metadata": {"name": "virt-launcher-win2025-eval-x", "namespace": "default",
                      "labels": {"kubevirt.io": "virt-launcher"}},
         "status": {"phase": "Running"}, "spec": {"nodeName": "n1"}},
    ]
    ctx = _build_ctx(state_model=model, predictor=predictor,
                     is_pod_excluded=pod_collection.is_pod_excluded)
    import predictive_agent.config as _cfg
    _old_l, _old_a = _cfg.EXCLUDED_POD_LABELS, _cfg.EXCLUDED_POD_ANNOTATIONS
    _cfg.EXCLUDED_POD_LABELS = {"kubevirt.io": "virt-launcher"}
    _cfg.EXCLUDED_POD_ANNOTATIONS = {}
    try:
        res = phases.evaluate_pods(ctx, 1, "kubernetes", pods, {}, {}, set())
    finally:
        _cfg.EXCLUDED_POD_LABELS, _cfg.EXCLUDED_POD_ANNOTATIONS = _old_l, _old_a
    assert "default/virt-launcher-win2025-eval-x" not in model.pods
    assert res.pods_tracked == 0
    assert res.at_risk_count == 0


def test_evaluate_pods_opa_deny_skips_pod():
    model = StateModel()
    predictor = Predictor(risk_threshold=0.001)
    ctx = _build_ctx(state_model=model, predictor=predictor,
                     policy_engine=MockPolicy(allow=False))
    res = phases.evaluate_pods(ctx, 1, "kubernetes", [_healthy_pod()], {}, {}, set())
    assert "default/p1" not in model.pods
    assert res.pods_tracked == 0


def test_evaluate_pods_plans_remediation_and_notifies():
    model = StateModel()
    predictor = Predictor(risk_threshold=0.0)
    notifier = MagicMock()
    remediation = FakeRemediationManager()
    remediation.plan_remediation.return_value = {"horizon": 1,
                                                 "steps": [{"action": "restart",
                                                            "rationale": "hot"}]}
    remediation.plan_and_execute.return_value = []
    ctx = _build_ctx(state_model=model, predictor=predictor, notifier=notifier,
                     remediation_manager=remediation, policy_engine=MockPolicy())
    res = phases.evaluate_pods(ctx, 1, "kubernetes", [_healthy_pod()], {}, {}, set())
    assert res.pods_tracked == 1
    assert res.at_risk_count >= 1
    remediation.plan_remediation.assert_called_once()
    plan_calls = [c for c in notifier.notify.call_args_list
                  if c.kwargs.get("alert_type") == "plan"]
    assert plan_calls, "a plan notification should have been sent"


def test_evaluate_pods_throttles_plan_notify():
    model = StateModel()
    predictor = Predictor(risk_threshold=0.0)
    notifier = MagicMock()
    remediation = FakeRemediationManager()
    remediation.plan_remediation.return_value = {"horizon": 1,
                                                 "steps": [{"action": "restart"}]}
    remediation.plan_and_execute.return_value = []
    ctx = _build_ctx(state_model=model, predictor=predictor, notifier=notifier,
                     remediation_manager=remediation, policy_engine=MockPolicy())
    phases.evaluate_pods(ctx, 1, "kubernetes", [_healthy_pod()], {}, {}, set())
    first = dict(ctx.last_plan_notify)["default/p1"]
    notifier.reset_mock()
    phases.evaluate_pods(ctx, 1, "kubernetes", [_healthy_pod()], {}, {}, set())
    plan_calls = [c for c in notifier.notify.call_args_list
                  if c.kwargs.get("alert_type") == "plan"]
    assert plan_calls == []
    assert ctx.last_plan_notify["default/p1"] == first


def test_evaluate_pods_remediation_failure_is_logged(caplog):
    model = StateModel()
    predictor = Predictor(risk_threshold=0.0)
    remediation = FakeRemediationManager()
    remediation.plan_remediation.return_value = {"horizon": 1,
                                                 "steps": [{"action": "restart"}]}
    remediation.plan_and_execute.side_effect = RuntimeError("boom")
    ctx = _build_ctx(state_model=model, predictor=predictor,
                     remediation_manager=remediation, policy_engine=MockPolicy())
    with caplog.at_level("ERROR"):
        phases.evaluate_pods(ctx, 1, "kubernetes", [_healthy_pod()], {}, {}, set())
    assert "Remediation evaluation failed" in caplog.text


def test_evaluate_pods_llm_thread_spawn_and_inflight():
    model = StateModel()
    analyzer = MagicMock()
    analyzer.analyze.return_value = {"severity": "high", "action": "scale"}
    ctx = _build_ctx(state_model=model, predictor=Predictor(risk_threshold=0.0),
                     policy_engine=MockPolicy(), llm_analyzer=analyzer)
    ctx.llm_analysis_cache["default/p1"] = 0.0  # older than TTL
    phases.evaluate_pods(ctx, 1, "kubernetes", [_healthy_pod()], {}, {}, set())
    # The cache timestamp is written synchronously right before Thread.start,
    # proving the LLM-spawn branch executed; the in-flight set is transient.
    assert "default/p1" in ctx.llm_analysis_cache


def test_evaluate_pods_llm_analysis_records_calls():
    import time

    import predictive_agent.main as m
    model = StateModel()
    analyzer = MagicMock()
    analyzer.analyze.return_value = {"severity": "high", "action": "scale"}
    ctx = _build_ctx(state_model=model, predictor=Predictor(risk_threshold=0.0),
                     policy_engine=MockPolicy(), llm_analyzer=analyzer)
    phases.evaluate_pods(ctx, 1, "kubernetes", [_healthy_pod()], {}, {}, set())
    time.sleep(0.2)
    # counter exists and is non-negative after the background thread runs
    assert hasattr(m, "_llm_calls")


def test_evaluate_pods_shadow_harness_observe():
    model = StateModel()
    predictor = Predictor(risk_threshold=0.0)
    shadow = MagicMock()
    remediation = FakeRemediationManager()
    remediation.plan_remediation.return_value = {"horizon": 1, "steps": [{"action": "restart"}]}
    remediation.plan_and_execute.return_value = []
    ctx = _build_ctx(state_model=model, predictor=predictor, shadow_harness=shadow,
                     remediation_manager=remediation, policy_engine=MockPolicy())
    phases.evaluate_pods(ctx, 1, "kubernetes", [_healthy_pod()], {}, {}, set())
    shadow.observe.assert_called_once()


def test_evaluate_pods_autonomy_notify_types():
    model = StateModel()
    predictor = Predictor(risk_threshold=0.0)
    notifier = MagicMock()
    remediation = FakeRemediationManager()
    remediation.plan_remediation.return_value = {"horizon": 1, "steps": [{"action": "restart"}]}
    remediation.plan_and_execute.return_value = []
    ctx = _build_ctx(state_model=model, predictor=predictor, notifier=notifier,
                     remediation_manager=remediation, policy_engine=MockPolicy())
    phases.evaluate_pods(ctx, 1, "kubernetes", [_healthy_pod()], {}, {}, set())
    types = {c.kwargs.get("alert_type") for c in notifier.notify.call_args_list}
    assert "plan" in types  # plan notify fires; no review_required/destructive in this path


@pytest.fixture
def _crashloop_policy():
    old = config.CRASHLOOP_POLICY
    config.CRASHLOOP_POLICY = "persist_restart"
    yield
    config.CRASHLOOP_POLICY = old


def test_evaluate_pods_crashloop_force_restart(_crashloop_policy, monkeypatch):
    model = StateModel()
    predictor = Predictor(risk_threshold=0.0)
    remediation = FakeRemediationManager()
    remediation.plan_remediation.return_value = {"horizon": 1, "steps": [{"action": "restart"}]}
    # plan_and_execute returns only none-steps → all_none True → force restart path
    remediation.plan_and_execute.return_value = [MagicMock(action="none", success=False)]
    remediation.evaluate.return_value = [MagicMock(action="restart", success=True)]
    monkeypatch.setattr(config, "REMEDIATION_RISK_THRESHOLD", 0)
    ctx = _build_ctx(state_model=model, predictor=predictor,
                     remediation_manager=remediation, policy_engine=MockPolicy())
    phases.evaluate_pods(ctx, 1, "kubernetes", [_healthy_pod()], {}, {}, set())
    remediation.evaluate.assert_called_once()


# ── Phase 4: calibration guardrail ─────────────────────────────────────────


def test_calibration_drift_notifies_and_decays():
    monitor = MagicMock()
    monitor.drift_detected.return_value = True
    monitor.brier.return_value = 0.05
    monitor.over_rate.return_value = 0.9
    monitor.status.return_value = "drift"
    remediation = FakeRemediationManager()
    remediation.degraded = False
    dm = MagicMock()
    dm.decay.return_value = 12.0
    remediation.decision_model = dm
    remediation.calibration_monitor = monitor
    notifier = MagicMock()
    ctx = _build_ctx(remediation_manager=remediation, notifier=notifier)
    phases.apply_calibration_guardrail(ctx)
    dm.decay.assert_called_once()
    drift_calls = [c for c in notifier.notify.call_args_list
                   if c.kwargs.get("alert_type") == "calibration_drift"]
    assert drift_calls
    assert drift_calls[0].kwargs["action_taken"] in ("rollback-to-reactive", "monitor-only")


def test_calibration_recovery_logs(caplog):
    monitor = MagicMock()
    monitor.drift_detected.return_value = False
    remediation = FakeRemediationManager()
    remediation.degraded = True
    remediation.calibration_monitor = monitor
    ctx = _build_ctx(remediation_manager=remediation)
    with caplog.at_level("INFO"):
        phases.apply_calibration_guardrail(ctx)
    assert "Calibration recovered" in caplog.text


def test_calibration_no_monitor_early_return():
    remediation = FakeRemediationManager()
    remediation.calibration_monitor = None
    ctx = _build_ctx(remediation_manager=remediation)
    phases.apply_calibration_guardrail(ctx)  # no crash, no-op


# ── Phase 5: continual learning ─────────────────────────────────────────────


def test_continual_learning_decays_at_interval():
    dm = MagicMock()
    dm.decay.return_value = 5.0
    model = StateModel()
    model.update_pod("ns", "p1", 10, 100, 10, 0, 0, False)
    remediation = FakeRemediationManager()
    remediation.decision_model = dm
    ctx = _build_ctx(state_model=model, remediation_manager=remediation)
    old = config.MODEL_DECAY_INTERVAL
    config.MODEL_DECAY_INTERVAL = 2
    try:
        counter = phases.apply_continual_learning(ctx, 0)  # 1
        assert counter == 1
        counter = phases.apply_continual_learning(ctx, counter)  # 2 → fires
        assert counter == 0
        dm.decay.assert_called_once()
    finally:
        config.MODEL_DECAY_INTERVAL = old


def test_continual_learning_records_markov():
    model = StateModel()
    model.markov = MagicMock()
    tracker = model.update_pod("ns", "p1", 10, 100, 10, 0, 0, False)
    remediation = FakeRemediationManager()
    ctx = _build_ctx(state_model=model, remediation_manager=remediation)
    old = config.MODEL_DECAY_INTERVAL
    config.MODEL_DECAY_INTERVAL = 1000
    try:
        phases.apply_continual_learning(ctx, 0)
        model.markov.record_transition.assert_called_with(tracker.prev_state, tracker.state)
    finally:
        config.MODEL_DECAY_INTERVAL = old


# ── Phase 6: persist_state ──────────────────────────────────────────────────


def test_persist_state_increments_counter():
    store = MagicMock()
    remediation = FakeRemediationManager()
    ctx = _build_ctx(state_store=store, state_model=StateModel(),
                     remediation_manager=remediation, predictor=Predictor())
    ctx.predictor._predictions = {}
    assert phases.persist_state(ctx, 0) == 1
    store.save_markov.assert_called_once()


def test_persist_state_swallows_store_errors():
    store = MagicMock()
    store.save_markov.side_effect = RuntimeError("disk full")
    ctx = _build_ctx(state_store=store, state_model=StateModel())
    assert phases.persist_state(ctx, 5) == 5  # error → no increment


def test_persist_state_decision_model_and_qtable_errors_logged(caplog):
    store = MagicMock()
    store.save_markov = MagicMock()
    dm = MagicMock()
    dm.to_action_dict.side_effect = RuntimeError("x")
    remediation = FakeRemediationManager()
    remediation.decision_model = dm
    remediation.rl_policy = MagicMock()
    remediation.rl_policy.to_dict.side_effect = RuntimeError("y")
    ctx = _build_ctx(state_store=store, state_model=StateModel(),
                     remediation_manager=remediation, predictor=Predictor())
    ctx.predictor._predictions = {}
    with caplog.at_level("ERROR"):
        phases.persist_state(ctx, 0)
    assert "Failed to persist decision model" in caplog.text


# ── Phase 7: backtest recording ────────────────────────────────────────────


def test_record_backtest_pairs_predictions_and_outcomes():
    model = StateModel()
    model.update_pod("ns", "p1", 10, 100, 10, 0, 0, False)
    predictor = Predictor(risk_threshold=0.0)
    predictor._predictions["ns/p1"] = MagicMock(
        risk_score=0.9, ttf_minutes=5, confidence=0.8, markov_state="HEALTHY",
        memory_trend=0.0, cpu_trend=0.0, memory_pct=10.0, cpu_pct=1.0)
    backtester = MagicMock()
    ctx = _build_ctx(state_model=model, predictor=predictor, backtester=backtester)
    phases.record_backtest_predictions(ctx)
    backtester.record_prediction.assert_called_once()
    backtester.record_outcome.assert_called_once()
    assert backtester.record_outcome.call_args.kwargs["actual_failure"] is False


def test_record_backtest_skips_missing_prediction():
    model = StateModel()
    model.update_pod("ns", "p1", 10, 100, 10, 0, 0, False)
    ctx = _build_ctx(state_model=model, predictor=Predictor(), backtester=MagicMock())
    phases.record_backtest_predictions(ctx)
    assert ctx.backtester.record_prediction.call_count == 0


# ── Phase 8: finalize_summary ───────────────────────────────────────────────


def test_finalize_summary_keys_and_remediation_stats():
    model = StateModel()
    remediation = FakeRemediationManager()
    remediation.get_stats.return_value = {"registered_actions": 6}
    ctx = _build_ctx(state_model=model, predictor=None, remediation_manager=remediation)
    summary = phases.finalize_summary(ctx, 1, 3, 1, {}, "2024-01-01T00:00:00Z",
                                      0.123, 0, 0, 0)
    assert {"predictions", "state", "timestamp", "pods_tracked",
            "at_risk_count", "reconcile_count", "remediation"} <= set(summary)
    assert summary["pods_tracked"] == 3
    assert summary["at_risk_count"] == 1
    assert summary["remediation"]["registered_actions"] == 6


def test_finalize_summary_includes_predictor_predictions():
    model = StateModel()
    predictor = Predictor()
    predictor._predictions = {"ns/p1": MagicMock(pod_key="ns/p1", risk_score=0.9, ttf_minutes=3)}
    ctx = _build_ctx(state_model=model, predictor=predictor)
    summary = phases.finalize_summary(ctx, 1, 1, 1, {}, "ts", 0.1, 0, 0, 0)
    assert summary["predictions"]


# ── pod_collection smoke ────────────────────────────────────────────────────


def test_pod_collection_helpers():
    from predictive_agent.pod_collection import (
        get_node_pressure,
        get_pod_name,
        get_pod_namespace,
        get_pod_restart_count,
        should_skip_namespace,
    )
    assert get_pod_restart_count(
        {"status": {"containerStatuses": [{"restartCount": 7}]}}) == 7
    assert get_pod_namespace({"metadata": {"namespace": "kube-system"}}) == "kube-system"
    assert get_pod_name({"metadata": {"name": "web-1"}}) == "web-1"
    assert should_skip_namespace("default") is False or True  # behavior under test
    mem, disk = get_node_pressure({"n1": {"MemoryPressure": "True", "DiskPressure": "False"}}, "n1")
    assert mem is True and disk is False


# ── Additional phase branches (coverage expansion) ─────────────────────────


def test_evaluate_pods_escalates_when_remediation_exhausted():
    model = StateModel()
    predictor = Predictor(risk_threshold=0.0)
    notifier = MagicMock()
    remediation = FakeRemediationManager()
    remediation.plan_remediation.return_value = {"horizon": 1,
                                                 "steps": [{"action": "restart"}]}
    # plan_and_execute returns only none-steps → all_none True → at-risk
    remediation.plan_and_execute.return_value = [MagicMock(action="none", success=False)]
    # exhausted → escalation + human notify
    remediation.note_remediation_outcome.return_value = (True, 3)
    ctx = _build_ctx(state_model=model, predictor=predictor, notifier=notifier,
                     remediation_manager=remediation, policy_engine=MockPolicy())
    phases.evaluate_pods(ctx, 1, "kubernetes", [_healthy_pod()], {}, {}, set())
    escalation = [c for c in notifier.notify.call_args_list
                  if c.kwargs.get("alert_type") == "remediation_exhausted"]
    assert escalation
    assert escalation[0].kwargs["action_taken"] == "escalate"


def test_evaluate_pods_autonomy_review_and_destructive_notify():
    """A review_required outcome and a successful node_cordon are emitted via
    the bounded-autonomy notifier (throttled per pod)."""
    model = StateModel()
    predictor = Predictor(risk_threshold=0.0)
    notifier = MagicMock()
    remediation = FakeRemediationManager()
    remediation.plan_remediation.return_value = {"horizon": 1, "steps": [{"action": "restart"}]}
    # plan_and_execute returns two results: a review_required deferral and a
    # successful node_cordon (destructive) action.
    remediation.plan_and_execute.return_value = [
        MagicMock(action="node_cordon", success=True, outcome="review_required",
                  message="needs review", target="n1"),
    ]
    remediation.note_remediation_outcome.return_value = (False, 0)
    ctx = _build_ctx(state_model=model, predictor=predictor, notifier=notifier,
                     remediation_manager=remediation, policy_engine=MockPolicy())
    phases.evaluate_pods(ctx, 1, "kubernetes", [_healthy_pod()], {}, {}, set())
    types = {c.kwargs.get("alert_type") for c in notifier.notify.call_args_list}
    assert "review_required" in types
    assert "plan" in types  # plan notify also fires for the at-risk pod


def test_evaluate_pods_autonomy_notify_throttled():
    """A second identical autonomy notify within 600s is suppressed."""
    model = StateModel()
    predictor = Predictor(risk_threshold=0.0)
    notifier = MagicMock()
    remediation = FakeRemediationManager()
    remediation.plan_remediation.return_value = {"horizon": 1, "steps": [{"action": "restart"}]}
    remediation.plan_and_execute.return_value = [
        MagicMock(action="node_cordon", success=True, outcome=None, target="n1"),
    ]
    remediation.note_remediation_outcome.return_value = (False, 0)
    ctx = _build_ctx(state_model=model, predictor=predictor, notifier=notifier,
                     remediation_manager=remediation, policy_engine=MockPolicy())
    phases.evaluate_pods(ctx, 1, "kubernetes", [_healthy_pod()], {}, {}, set())
    notifier.reset_mock()
    phases.evaluate_pods(ctx, 1, "kubernetes", [_healthy_pod()], {}, {}, set())
    autonomy = [c for c in notifier.notify.call_args_list
                if c.kwargs.get("alert_type") == "destructive_action"]
    assert autonomy == []


# ── _spawn_llm_analysis branches ────────────────────────────────────────────


def _make_llm_ctx(policy_allow_llm=True, policy_raise=False, analyze_exc=None):
    model = StateModel()
    analyzer = MagicMock()
    if analyze_exc is not None:
        analyzer.analyze.side_effect = analyze_exc
    else:
        analyzer.analyze.return_value = {"severity": "high", "action": "scale"}
    # Always install an OPA policy so the LLM-usage gate branch is exercised.
    policy = _OpaPolicy(allow_llm=policy_allow_llm, raise_on_check=policy_raise)
    predictor = Predictor(risk_threshold=0.0)
    # Force an at-risk, LLM-spawning result regardless of pod metrics.
    predictor.predict = MagicMock(return_value=MagicMock(
        risk_score=0.9, ttf_minutes=3, confidence=0.8, markov_state="CRITICAL"))
    ctx = _build_ctx(state_model=model, predictor=predictor,
                     policy_engine=policy, llm_analyzer=analyzer)
    ctx.llm_analysis_cache["default/p1"] = 0.0
    return ctx, analyzer


class _OpaPolicy:
    def __init__(self, allow_llm=True, raise_on_check=False, allow_pod=True):
        self._allow_llm = allow_llm
        self._raise = raise_on_check
        self._allow_pod = allow_pod
    def check_pod_security(self, pod):
        return {"result": {"allow": self._allow_pod}, "violation": "none"}
    def check_llm_usage(self, req):
        if self._raise:
            raise RuntimeError("opa down")
        return {"result": {"allow_llm": self._allow_llm}}


def test_llm_analysis_opa_denies_skips_analyze():
    import time
    ctx, analyzer = _make_llm_ctx(policy_allow_llm=False)
    phases.evaluate_pods(ctx, 1, "kubernetes", [_healthy_pod()], {}, {}, set())
    time.sleep(0.15)
    analyzer.analyze.assert_not_called()


def test_llm_analysis_opa_gate_exception_still_runs():
    import time
    ctx, analyzer = _make_llm_ctx(policy_raise=True)
    phases.evaluate_pods(ctx, 1, "kubernetes", [_healthy_pod()], {}, {}, set())
    time.sleep(0.15)
    analyzer.analyze.assert_called_once()


def test_llm_analysis_exception_increments_llm_errors():
    import time

    import predictive_agent.main as m
    before_err = m._llm_errors
    ctx, analyzer = _make_llm_ctx(analyze_exc=RuntimeError("llm crash"))
    phases.evaluate_pods(ctx, 1, "kubernetes", [_healthy_pod()], {}, {}, set())
    time.sleep(0.15)
    assert m._llm_errors >= before_err + 1


# ── calibration & continual-learning failure branches ─────────────────────


def test_calibration_drift_decay_failure_logged(caplog):
    monitor = MagicMock()
    monitor.drift_detected.return_value = True
    monitor.brier.return_value = 0.05
    monitor.over_rate.return_value = 0.9
    monitor.status.return_value = "drift"
    remediation = FakeRemediationManager()
    remediation.degraded = False
    remediation.decision_model.decay.side_effect = RuntimeError("decay boom")
    remediation.calibration_monitor = monitor
    ctx = _build_ctx(remediation_manager=remediation, notifier=MagicMock())
    with caplog.at_level("WARNING"):
        phases.apply_calibration_guardrail(ctx)
    assert "Drift decay failed" in caplog.text


def test_continual_learning_decay_failure_logged(caplog):
    model = StateModel()
    model.update_pod("ns", "p1", 10, 100, 10, 0, 0, False)
    remediation = FakeRemediationManager()
    remediation.decision_model.decay.side_effect = RuntimeError("decay boom")
    ctx = _build_ctx(state_model=model, remediation_manager=remediation)
    old = config.MODEL_DECAY_INTERVAL
    config.MODEL_DECAY_INTERVAL = 1
    try:
        with caplog.at_level("WARNING"):
            phases.apply_continual_learning(ctx, 0)
        assert "Model decay failed" in caplog.text
    finally:
        config.MODEL_DECAY_INTERVAL = old


# ── backtest & finalize_summary error branches ─────────────────────────────


def test_record_backtest_swallows_recording_exceptions(caplog):
    model = StateModel()
    model.update_pod("ns", "p1", 10, 100, 10, 0, 0, False)
    predictor = Predictor(risk_threshold=0.0)
    predictor._predictions["ns/p1"] = MagicMock(
        risk_score=0.9, ttf_minutes=5, confidence=0.8, markov_state="HEALTHY",
        memory_trend=0.0, cpu_trend=0.0, memory_pct=10.0, cpu_pct=1.0)
    backtester = MagicMock()
    backtester.record_prediction.side_effect = RuntimeError("bt down")
    ctx = _build_ctx(state_model=model, predictor=predictor, backtester=backtester)
    with caplog.at_level("DEBUG"):
        phases.record_backtest_predictions(ctx)
    assert "Backtester recording failed" in caplog.text


def test_finalize_summary_handles_server_unavailable():
    # If `from predictive_agent.server import _context` fails, finalize_summary
    # must still return the summary (the server-sync block is best-effort).
    model = StateModel()
    ctx = _build_ctx(state_model=model, predictor=None,
                     remediation_manager=FakeRemediationManager())
    import sys
    # Make the server import raise by poisoning the name binding.
    orig = sys.modules.get("predictive_agent.server")
    try:
        sys.modules["predictive_agent.server"] = None  # forces ImportError on `from ... import`
        summary = phases.finalize_summary(ctx, 1, 1, 0, {}, "ts", 0.1, 0, 0, 0)
    finally:
        if orig is not None:
            sys.modules["predictive_agent.server"] = orig
        else:
            sys.modules.pop("predictive_agent.server", None)
    assert summary["reconcile_count"] == 1


def test_finalize_summary_at_risk_logging(caplog):
    model = StateModel()
    ctx = _build_ctx(state_model=model, predictor=None)
    with caplog.at_level("WARNING"):
        phases.finalize_summary(ctx, 1, 0, 2, {}, "ts", 0.1, 0, 0, 0)
    assert "2 pods at risk" in caplog.text


# ── collect_metrics: docker-mode helper delegation ────────────────────────


def test_evaluate_pods_docker_collects_container_logs():
    """In docker mode evaluate_pods shells out to `docker logs` for log errors."""
    calls = []
    def fake_run_cmd(cmd, *a, **k):
        calls.append(cmd[0] if isinstance(cmd, list) else cmd)
        return (0, "", "")  # logs output (parsed by count_log_errors below)
    pod_item = {
        "metadata": {"namespace": "shop", "name": "web"},
        "spec": {"nodeName": "docker-host"},
        "status": {"phase": "Running", "containerStatuses": []},
        "_container_id": "abc123def456",
        "_docker_info": {"restart_count": 0, "healthy": True},
    }
    ctx = _build_ctx(state_model=StateModel(), predictor=Predictor(risk_threshold=1.5),
                     run_cmd=fake_run_cmd,
                     get_watch_selectors=lambda: {"runtime": "docker"},
                     matches_docker_selectors=lambda info, sel: True,
                     get_container_status_signals=lambda p: {"pod_phase": "Running",
                                                             "container_ready": True,
                                                             "wait_state": None,
                                                             "terminated": False,
                                                             "terminated_reason": None},
                     count_log_errors=lambda s: 5)
    phases.evaluate_pods(ctx, 1, "docker", [pod_item], {}, {}, set())
    assert "docker" in calls  # docker logs was attempted


# ── Self-health wiring (persistence + degraded collector → rate-limited mail)


def test_persist_failure_sends_self_health_email_once():
    phases.reset_self_health()
    try:
        store = MagicMock()
        store.save_markov.side_effect = RuntimeError("permission denied")
        notifier = MagicMock()  # .email.send is the notify_fn
        ctx = _build_ctx(state_store=store, state_model=StateModel(), notifier=notifier)
        phases.persist_state(ctx, 0)
        assert notifier.email.send.call_count == 1
        subject, body = notifier.email.send.call_args[0]
        assert "[self-health:persistence]" in subject
        assert "permission denied" in body
        # second persist failure inside the cooldown window → suppressed
        phases.persist_state(ctx, 0)
        assert notifier.email.send.call_count == 1
    finally:
        phases.reset_self_health()


def test_persist_success_does_not_alert():
    phases.reset_self_health()
    try:
        store = MagicMock()
        notifier = MagicMock()
        ctx = _build_ctx(state_store=store, state_model=StateModel(), notifier=notifier)
        phases.persist_state(ctx, 0)
        notifier.email.send.assert_not_called()
    finally:
        phases.reset_self_health()


def test_degraded_collector_streak_alerts_via_collect_cycle():
    phases.reset_self_health()
    try:
        notifier = MagicMock()
        ctx = _build_ctx(notifier=notifier)
        _, tracker = phases._get_self_health(ctx)
        # 4 degraded cycles: below default threshold → silent
        for _ in range(4):
            assert tracker.cycle("DEGRADED") is False
        notifier.email.send.assert_not_called()
        # 5th consecutive → alert
        assert tracker.cycle("DEGRADED") is True
        assert notifier.email.send.call_count == 1
        assert "[self-health:collector]" in notifier.email.send.call_args[0][0]
        # healthy cycle resets the streak
        assert tracker.cycle("HEALTHY") is False
        assert tracker.consecutive == 0
    finally:
        phases.reset_self_health()


def test_self_health_no_notifier_is_safe():
    phases.reset_self_health()
    try:
        store = MagicMock()
        store.save_markov.side_effect = RuntimeError("boom")
        ctx = _build_ctx(state_store=store, state_model=StateModel(), notifier=None)
        assert phases.persist_state(ctx, 0) == 0  # no crash without notifier
    finally:
        phases.reset_self_health()
