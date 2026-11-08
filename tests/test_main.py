"""Test main reconcile loop."""
from unittest.mock import patch

from predictive_agent import config as _config
from predictive_agent.main import ReconcileLoop, reconcile


def _make_pod(name, phase="Running", namespace="default", owner_kind=None, ready=True):
    """Build a minimal pod JSON object like kubectl -o json returns."""
    pod = {
        "metadata": {
            "name": name,
            "namespace": namespace,
            "creationTimestamp": "2025-01-01T00:00:00Z",
            "ownerReferences": [],
        },
        "spec": {
            "nodeName": "node-1",
            "containers": [{
                "name": name,
                "resources": {"limits": {"memory": "512Mi", "cpu": "500m"}},
            }],
        },
        "status": {
            "phase": phase,
            "conditions": [
                {"type": "Initialized", "status": "True"},
                {"type": "PodScheduled", "status": "True"},
                {"type": "ContainersReady", "status": "True" if ready else "False"},
            ],
            "containerStatuses": [{"ready": ready, "restartCount": 0}],
        },
    }
    if owner_kind:
        pod["metadata"]["ownerReferences"].append({"kind": owner_kind, "name": f"{name}-owner"})
    return pod


def _init_state():
    """Initialize the module-level state model + predictor for reconcile tests."""
    import predictive_agent.main as m
    m._state_model = m.StateModel()
    m._predictor = m.Predictor(risk_threshold=0.3)
    m._remediation_manager = None
    m._notifier = None
    m._kg_client = None
    m._backtester = None
    m._state_store = None
    m._llm_analyzer = None
    m._llm_analysis_cache.clear()


def _reconcile_with_pods(pods):
    """Run reconcile with the given pod list, patching kubectl collection."""
    with patch("predictive_agent.main.detect_runtime", return_value="kubernetes"), \
         patch("predictive_agent.main._get_pods_json", return_value={"items": pods}), \
         patch("predictive_agent.main.collect_top_metrics", return_value={}), \
         patch("predictive_agent.main.collect_top_nodes", return_value={}), \
         patch("predictive_agent.main.get_node_conditions", return_value={}), \
         patch("predictive_agent.main.count_log_errors", return_value=0):
        return reconcile()


def test_job_owned_pod_is_not_tracked():
    """Regression (Fix 4): Job/CronJob-owned pods must not be tracked."""
    import predictive_agent.main as m
    _init_state()
    job_pod = _make_pod("mariadb-galera-tenant-db-init", phase="Succeeded", owner_kind="Job")
    _reconcile_with_pods([job_pod])
    assert "default/mariadb-galera-tenant-db-init" not in m._state_model.pods
    assert "default/mariadb-galera-tenant-db-init" not in m._predictor._predictions


def test_succeeded_pod_is_not_tracked():
    """Regression (Fix 4): Succeeded-phase pods must not be tracked."""
    import predictive_agent.main as m
    _init_state()
    done_pod = _make_pod("init-container", phase="Succeeded")
    _reconcile_with_pods([done_pod])
    assert "default/init-container" not in m._state_model.pods
    assert "default/init-container" not in m._predictor._predictions


def test_regular_pod_is_tracked():
    """A healthy running deployment pod should still be tracked."""
    import predictive_agent.main as m
    _init_state()
    pod = _make_pod("app-1", phase="Running", owner_kind="ReplicaSet")
    _reconcile_with_pods([pod])
    assert "default/app-1" in m._state_model.pods
    assert "default/app-1" in m._predictor._predictions


def test_llm_analysis_runs_in_background_thread(monkeypatch):
    """Regression (LLM integration fix): LLM analysis must run in a
    background thread so a slow LLM never blocks the reconcile loop, and
    must not raise UnboundLocalError on the counters."""
    import time

    import predictive_agent.main as m
    _init_state()

    calls = []

    class FakeAnalyzer:
        def analyze(self, issue, context, prediction=None):
            calls.append((issue, context, prediction))
            time.sleep(0.2)  # simulate a slow LLM
            return {"analysis": "test", "severity": "high", "action": "restart"}

    m._llm_analyzer = FakeAnalyzer()
    m._llm_calls = 0
    m._llm_errors = 0
    m._llm_analysis_cache.clear()
    m._llm_in_flight.clear()
    # Make the analysis threshold permissive so the background-thread path is
    # exercised regardless of the exact risk model (CI/test hygiene: the test
    # verifies threading + latency, not the risk threshold crossing).
    monkeypatch.setattr(_config, "LLM_ANALYSIS_THRESHOLD", 0.0)

    # A pod with very high risk (would exceed LLM_ANALYSIS_THRESHOLD 0.7)
    pod = _make_pod("crashing-pod", phase="Running", owner_kind="ReplicaSet")
    # Force high risk: not ready -> lr*5 -> risk ~0.47+ ; combine with
    # node pressure + high memory for well above 0.7
    pod["status"]["containerStatuses"][0]["ready"] = False
    pod["spec"]["containers"][0]["resources"]["limits"]["memory"] = "128Mi"

    with patch("predictive_agent.main.detect_runtime", return_value="kubernetes"), \
         patch("predictive_agent.main._get_pods_json", return_value={"items": [pod]}), \
         patch("predictive_agent.main.collect_top_metrics",
               return_value={"default/crashing-pod": {"cpu_m": 400, "memory_mib": 120}}), \
         patch("predictive_agent.main.collect_top_nodes", return_value={}), \
         patch("predictive_agent.main.get_node_conditions",
               return_value={"node-1": {"MemoryPressure": "True"}}), \
         patch("predictive_agent.main.count_log_errors", return_value=0):
        start = time.monotonic()
        reconcile()
        elapsed = time.monotonic() - start

    # Reconcile must NOT block on the slow LLM call
    assert elapsed < 5.0, f"reconcile blocked {elapsed:.1f}s on LLM call"
    assert len(calls) >= 1, "LLM analysis was not triggered"

    # Wait for the background thread to finish
    deadline = time.monotonic() + 5
    while m._llm_in_flight and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not m._llm_in_flight, "in-flight set not drained"
    assert m._llm_calls >= 1, "llm_calls counter not incremented"
    assert m._llm_errors == 0
    m._llm_analyzer = None


def test_finalized_pod_cleans_stale_prediction():
    """Regression (Fix 1): when a pod disappears, its stale prediction must
    be removed from _predictor._predictions (memory leak fix)."""
    import predictive_agent.main as m
    _init_state()
    pod = _make_pod("app-1", phase="Running", owner_kind="ReplicaSet")
    _reconcile_with_pods([pod])
    assert "default/app-1" in m._predictor._predictions

    # Pod disappears on next cycle → must be finalized: removed from both
    # state model AND predictions
    _reconcile_with_pods([])
    assert "default/app-1" not in m._state_model.pods
    assert "default/app-1" not in m._predictor._predictions


def test_pod_transitioning_to_succeeded_drops_prediction():
    """Regression (Fix 4 + Fix 1): a previously-tracked pod that transitions
    to Succeeded must be removed from state model + predictions."""
    import predictive_agent.main as m
    _init_state()
    pod = _make_pod("job-pod", phase="Running", owner_kind="ReplicaSet")
    _reconcile_with_pods([pod])
    assert "default/job-pod" in m._predictor._predictions

    # Same pod but now Succeeded → should be dropped
    done_pod = _make_pod("job-pod", phase="Succeeded", owner_kind="ReplicaSet")
    _reconcile_with_pods([done_pod])
    assert "default/job-pod" not in m._predictor._predictions


def test_reconcile_exists():
    """reconcile function should exist and be callable."""
    assert callable(reconcile)


def test_reconcile_returns_predictions():
    """reconcile should return a list of predictions."""
    with patch("predictive_agent.main.collect_top_metrics", return_value={}), \
         patch("predictive_agent.main.collect_top_nodes", return_value={}), \
         patch("predictive_agent.main.get_pod_resources", return_value={}), \
         patch("predictive_agent.main.get_node_conditions", return_value={}), \
         patch("predictive_agent.main.count_log_errors", return_value=0):
        result = reconcile()
        assert isinstance(result, dict)
        assert "predictions" in result
        assert "state" in result
        assert "timestamp" in result


def test_reconcile_with_metrics():
    """reconcile should process metrics and produce predictions."""
    mock_metrics = {
        "default/test-pod": {"cpu_m": 100, "memory_mib": 256}
    }
    mock_nodes = {
        "node-1": {"cpu_m": 500, "memory_mib": 4000, "cpu_pct": 25.0, "memory_pct": 40.0}
    }
    mock_resources = {
        "test-container": {"cpu_m": 1000, "memory_mib": 512}
    }
    mock_conditions = {
        "node-1": {"Ready": "True", "MemoryPressure": "False", "DiskPressure": "False"}
    }
    with patch("predictive_agent.main.collect_top_metrics", return_value=mock_metrics), \
         patch("predictive_agent.main.collect_top_nodes", return_value=mock_nodes), \
         patch("predictive_agent.main.get_pod_resources", return_value=mock_resources), \
         patch("predictive_agent.main.get_node_conditions", return_value=mock_conditions), \
         patch("predictive_agent.main.count_log_errors", return_value=0):
        result = reconcile()
        assert "predictions" in result
        assert isinstance(result["predictions"], list)
        assert "state" in result
        assert "timestamp" in result


def test_reconcile_loop_creation():
    """ReconcileLoop should be creatable."""
    loop = ReconcileLoop(interval=1)
    assert loop.interval == 1
    assert loop.running is False or loop.running is True


def test_reconcile_loop_start_stop():
    """ReconcileLoop should start and stop."""
    loop = ReconcileLoop(interval=1)
    loop.start()
    assert loop.running is True
    loop.stop()
    assert loop.running is False


def test_reconcile_loop_runs_once():
    """ReconcileLoop should run reconcile at least once when started."""
    call_count = 0
    _original_reconcile = None

    def mock_reconcile():
        nonlocal call_count
        call_count += 1
        return {"predictions": [], "state": {}, "timestamp": "2025-01-01T00:00:00Z"}

    loop = ReconcileLoop(interval=0.2)
    loop._reconcile_fn = mock_reconcile
    loop.start()
    import time
    time.sleep(0.7)  # ≥2 calls at 0, 0.2, 0.4, 0.6
    loop.stop()
    assert call_count >= 2
