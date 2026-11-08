"""Mutation-testing harvest batch (tier-2-mutation-testing, task 2.3).

Each test kills a *family* of surviving mutants by pinning an exact
behavioral contract the original suite left loose:

- kubectl/docker argv + timeout contracts (kills string-mutation,
  flag-case, and timeout±1 mutant families)
- owner-reference Job/CronJob filtering (kills string-casing/key mutants)
- log-collection gates: SKIP_LOGS_STATUSES membership, ready/restart
  condition polarity (kills `not in`→`in`, `is False`→`is True`,
  `and`→`or` families)
- inclusive risk>=0.7 WARNING threshold (kills `>=`→`>`)
- persist_state payload key contract (kills dict-key casing/value mutants)
- autonomy-notify getattr defaults (kills default→None mutants)
- removed-pod finalization state casing ("FAILED" vs "failed")
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import predictive_agent.reconcile_phases as phases
from predictive_agent.predictor import Predictor
from predictive_agent.state_model import StateModel
from tests.test_reconcile_phases import (  # reuse campaign fixtures
    FakeRemediationManager,
    MockPolicy,
    _build_ctx,
    _healthy_pod,
)


class RecordingRunCmd:
    """Records every argv/timeout, returns queued (rc, stdout) per call."""

    def __init__(self, responses=None):
        self.calls = []
        self.responses = list(responses or [])

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append((cmd, kwargs))
        if self.responses:
            return self.responses.pop(0)
        return (0, "", "")


def _recording_ctx(**overrides):
    run = RecordingRunCmd()
    ctx = _build_ctx(state_model=StateModel(),
                     predictor=Predictor(risk_threshold=1.5),
                     run_cmd=run, **overrides)
    return ctx, run


# ── argv/timeout contracts ──────────────────────────────────────────────────


def test_collect_metrics_k8s_exact_argv_and_timeouts():
    """The k8s collection path shells out with exact argv and timeouts."""
    ctx, run = _recording_ctx(
        detect_runtime=lambda: "kubernetes",
        get_pods_json=lambda: {"items": []},
        collect_top_metrics=lambda out: {},
        collect_top_nodes=lambda out: None,
        get_node_conditions=lambda raw: {},
    )
    run.responses = [(0, "ns/p 1 10m", ""), (0, "node 1 10%", "")]
    phases.collect_metrics(ctx, 1)
    argvs = [c[0] for c in run.calls]
    assert ["kubectl", "top", "pods", "-A", "--no-headers"] in argvs
    assert ["kubectl", "top", "nodes", "--no-headers"] in argvs
    assert ["kubectl", "get", "nodes", "-o", "json"] in argvs
    for cmd, kwargs in run.calls:
        assert kwargs.get("timeout") == 10, (cmd, kwargs)


def test_collect_metrics_ownerref_job_pods_not_tracked():
    """Completed-job filtering: Job/CronJob-owned pods are not tracked."""
    def pod(kind):
        return {"metadata": {"namespace": "ns", "name": f"p-{kind}",
                             "ownerReferences": [{"kind": kind}]},
                "status": {"phase": "Running"}}
    ctx, run = _recording_ctx(
        detect_runtime=lambda: "kubernetes",
        get_pods_json=lambda: {"items": [pod("Job"), pod("CronJob"),
                                         pod("ReplicaSet")]},
        collect_top_metrics=lambda out: {},
        collect_top_nodes=lambda out: None,
        get_node_conditions=lambda raw: {},
        get_pod_namespace=lambda p: p["metadata"]["namespace"],
        get_pod_name=lambda p: p["metadata"]["name"],
    )
    run.responses = [(0, "", ""), (0, "", "")]
    result = phases.collect_metrics(ctx, 1)
    # CollectResult.current_keys is threaded into prepare_state_model —
    # Job/CronJob-owned pods must NOT be tracked as current.
    assert result.current_keys == {"ns/p-ReplicaSet"}


# ── log-collection gates ────────────────────────────────────────────────────


def _docker_pod_item(phase="Running", ready=False, restarts=1):
    return {
        "metadata": {"namespace": "shop", "name": "web"},
        "spec": {"nodeName": "docker-host"},
        "status": {"phase": phase,
                   "containerStatuses": [{"ready": ready,
                                          "restartCount": restarts}]},
        "_container_id": "abc123def456",
        "_docker_info": {"restart_count": restarts, "healthy": True},
    }


def _docker_ctx(run):
    def signals(p):
        return {"pod_phase": p["status"]["phase"],
                "container_ready": True, "wait_state": None,
                "terminated": False, "terminated_reason": None,
                "pod_scheduled": True}
    return _build_ctx(
        state_model=StateModel(), predictor=Predictor(risk_threshold=1.5),
        run_cmd=run,
        get_watch_selectors=lambda: {"runtime": "docker"},
        matches_docker_selectors=lambda info, sel: True,
        get_pod_status_signals=signals,
        count_log_errors=lambda s: 3,
    )


def test_evaluate_pods_no_logs_when_containers_ready():
    """ready=True + restartCount=0 ⇒ no kubectl logs call (polarity pin)."""
    run = RecordingRunCmd()
    ctx = _docker_ctx(run)
    phases.evaluate_pods(ctx, 1, "kubernetes",
                         [_docker_pod_item(ready=True, restarts=0)],
                         {}, {}, set())
    assert not any("logs" in c[0] for c in run.calls)


def test_evaluate_pods_logs_unready_container_exact_argv():
    """ready=False ⇒ exact `kubectl logs` argv with --tail=50, timeout=5."""
    run = RecordingRunCmd([(0, "ERR a\nERR b", "")])
    ctx = _docker_ctx(run)
    phases.evaluate_pods(ctx, 1, "kubernetes",
                         [_docker_pod_item(ready=False, restarts=0)],
                         {}, {}, set())
    log_calls = [(cmd, kw) for cmd, kw in run.calls if "logs" in cmd]
    assert log_calls, run.calls
    cmd, kwargs = log_calls[0]
    assert cmd == ["kubectl", "logs", "-n", "shop", "web", "--tail=50"]
    assert kwargs.get("timeout") == 5


def test_evaluate_pods_skip_logs_for_pending_phase():
    """Phase in SKIP_LOGS_STATUSES ⇒ no logs even for unready containers."""
    run = RecordingRunCmd()
    ctx = _docker_ctx(run)
    phases.evaluate_pods(ctx, 1, "kubernetes",
                         [_docker_pod_item(phase="Pending", ready=False)],
                         {}, {}, set())
    assert not any("logs" in c[0] for c in run.calls)


# ── risk threshold polarity ─────────────────────────────────────────────────


def test_evaluate_pods_warning_threshold_inclusive_at_070(caplog):
    """risk_score == 0.7 logs the at-risk WARNING (>= is inclusive)."""
    model = StateModel()
    predictor = Predictor(risk_threshold=0.0)
    predictor.predict = lambda *a, **k: SimpleNamespace(
        risk_score=0.7, ttf_minutes=3, confidence=0.8,
        markov_state="STRESSED", pod_key="ns/p1")
    ctx = _build_ctx(state_model=model, predictor=predictor,
                     remediation_manager=FakeRemediationManager(),
                     policy_engine=MockPolicy(), notifier=None)
    import logging
    with caplog.at_level(logging.WARNING):
        phases.evaluate_pods(ctx, 1, "kubernetes", [_healthy_pod()],
                             {}, {}, set())
    assert any("at risk" in r.message for r in caplog.records)


# ── persist_state payload key contract ──────────────────────────────────────


def test_persist_state_predictions_payload_exact_keys():
    """save_predictions receives the exact documented key set per record."""
    saved = {}

    class Store:
        def save_markov(self, m):
            pass
        def save_decision_model(self, d):
            pass
        def save_predictions(self, data):
            saved["data"] = data

    class Pred:
        pod_key = "ns/p"
        risk_score = 0.5
        ttf_minutes = 3
        confidence = 0.9
        markov_state = "HEALTHY"
        memory_trend = 1.0
        cpu_trend = 0.0
        memory_pct = 10.0
        cpu_pct = 1.0

    predictor = SimpleNamespace(_predictions={"ns/p": Pred()})
    remediation = FakeRemediationManager()
    ctx = _build_ctx(state_model=StateModel(), state_store=Store(),
                     predictor=predictor, remediation_manager=remediation)
    saves = phases.persist_state(ctx, 0)
    assert saves == 1
    (record,) = saved["data"]
    assert set(record.keys()) == {
        "pod_key", "risk_score", "ttf_minutes", "confidence",
        "markov_state", "memory_trend", "cpu_trend", "memory_pct",
        "cpu_pct"}
    assert record["pod_key"] == "ns/p"
    assert record["risk_score"] == 0.5


# ── autonomy-notify getattr defaults ────────────────────────────────────────


def test_evaluate_pods_autonomy_notify_defaults_when_attrs_missing():
    """Result objects lacking .message/.target still notify with kind fallback."""
    model = StateModel()
    predictor = Predictor(risk_threshold=0.0)
    notifier = MagicMock()
    remediation = FakeRemediationManager()
    remediation.plan_remediation.return_value = {"horizon": 1,
                                                 "steps": [{"action": "x"}]}
    # Plain namespace: NO message/outcome attrs -> getattr defaults engage.
    remediation.plan_and_execute.return_value = [
        SimpleNamespace(action="node_cordon", success=True)]
    remediation.note_remediation_outcome.return_value = (False, 0)
    ctx = _build_ctx(state_model=model, predictor=predictor,
                     notifier=notifier, remediation_manager=remediation,
                     policy_engine=MockPolicy())
    phases.evaluate_pods(ctx, 1, "kubernetes", [_healthy_pod()], {}, {}, set())
    autonomy = [c for c in notifier.notify.call_args_list
                if c.kwargs.get("alert_type") == "destructive_action"]
    assert autonomy, notifier.notify.call_args_list
    assert autonomy[0].kwargs["action_taken"] == "node_cordon"
    assert autonomy[0].kwargs["details"] == "destructive_action"


# ── removed-pod finalization state casing ───────────────────────────────────


def test_removed_pod_finalize_transitions_lowercase_failed(caplog):
    """state 'failed' (lowercase) is NOT the terminal 'FAILED': the finalize
    path must still record the transition for it."""
    model = StateModel()
    tracker = model.track_pod("shop", "gone")
    tracker.state = "failed"  # non-canonical casing -> must be finalized
    ctx = _build_ctx(state_model=model,
                     remediation_manager=FakeRemediationManager())
    calls = []
    model.markov = SimpleNamespace(
        record_transition=lambda a, b: calls.append((a, b)))
    phases.prepare_state_model(ctx, 1, "OK", current_keys=set(),
                               pod_items=[])  # no pods -> all removed
    assert calls == [("failed", "FAILED")]
    # canonical FAILED trackers are NOT re-transitioned
    tracker2 = model.track_pod("shop", "done")
    tracker2.state = "FAILED"
    model.markov = SimpleNamespace(
        record_transition=lambda a, b: calls.append((a, b)))
    phases.prepare_state_model(ctx, 2, "OK", current_keys=set(), pod_items=[])
    assert calls == [("failed", "FAILED")]  # unchanged


# ── node pressure key contract (pod_collection) ─────────────────────────────


def test_get_node_pressure_exact_condition_keys():
    """MemoryPressure/DiskPressure are read as exact condition keys."""
    from predictive_agent import pod_collection as pc
    mem, disk = pc.get_node_pressure(
        {"n1": {"MemoryPressure": "True", "DiskPressure": "True"}}, "n1")
    assert (mem, disk) == (True, True)
    mem, disk = pc.get_node_pressure(
        {"n1": {"MemoryPressure": "False", "DiskPressure": "False"}}, "n1")
    assert (mem, disk) == (False, False)


def test_get_pods_json_exact_argv_and_timeout():
    """get_pods_json shells out with the exact argv and timeout=10."""
    import json

    from predictive_agent import main as main_mod
    from predictive_agent import pod_collection as pc
    calls = []

    def fake_run_cmd(cmd, *a, **k):
        calls.append((cmd, k))
        return (0, json.dumps({"items": []}), "")

    # Hermetic: pretend the binary exists regardless of the CI host.
    with patch.object(main_mod, "run_cmd", side_effect=fake_run_cmd), \
         patch.object(pc, "kubectl_binary_missing", return_value=False):
        pc.get_pods_json()
    (cmd, kwargs), = calls
    assert cmd == ["kubectl", "get", "pods", "-A", "-o", "json"]
    assert kwargs.get("timeout") == 10


def test_collect_docker_container_metrics_two_line_parse():
    """Two stats lines parse into two exact metric dicts (line-split pin)."""
    from predictive_agent import main as main_mod
    from predictive_agent import pod_collection as pc
    stdout = ("abc111,5.50%,50MiB / 1GiB\n"
              "def222,120.00%,2GiB / 4GiB\n")
    with patch.object(main_mod, "run_cmd",
                      return_value=(0, stdout, "")):
        metrics = pc.collect_docker_container_metrics()
    assert set(metrics.keys()) == {"abc111", "def222"}
    assert metrics["abc111"]["cpu_m"] == 55      # 5.50% * 10
    assert metrics["def222"]["cpu_m"] == 1200    # 120.00% * 10
    assert set(metrics["abc111"].keys()) == {"cpu_m", "memory_mib"}
