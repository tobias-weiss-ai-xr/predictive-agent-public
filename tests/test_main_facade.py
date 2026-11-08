"""Tests for the refactored main.py reconcile facade (PA-S2).

Verifies that (1) the public names tests historically patch are still
present as module attributes on ``predictive_agent.main`` (late-bound
aliases to pod_collection), (2) patching ``main._get_pods_json`` /
``main._collect_docker_container_metrics`` intercepts reconcile(), and
(3) reconcile() still returns the documented summary contract.
"""

from unittest.mock import patch

import pytest

import predictive_agent.main as main_mod
from predictive_agent.predictor import Predictor
from predictive_agent.state_model import StateModel

# ── Preserved public names (aliases must still exist) ───────────────────────


@pytest.mark.parametrize("name", [
    "_get_pods_json", "_collect_docker_container_metrics", "_get_pod_restart_count",
    "_get_pod_namespace", "_get_pod_name", "_get_pod_memory_limit",
    "_should_skip_namespace", "_is_pod_excluded", "_get_node_pressure", "_matches_docker_selectors",
    "_get_container_memory_limit", "_now_iso", "_kubectl_available",
    "reconcile", "ReconcileLoop", "main",
])
def test_facade_aliases_present(name):
    assert hasattr(main_mod, name), f"main.{name} must remain a public alias"


def test_facade_reconcile_signature_unchanged():
    import inspect
    sig = inspect.signature(main_mod.reconcile)
    assert list(sig.parameters) == []  # no-arg reconcile


def test_facade_state_model_none_returns_minimal_summary():
    main_mod._state_model = None
    summary = main_mod.reconcile()
    assert isinstance(summary, dict)
    for k in ("predictions", "state", "timestamp", "pods_tracked",
              "at_risk_count", "reconcile_count"):
        assert k in summary
    assert summary["predictions"] == []
    assert summary["pods_tracked"] == 0


@pytest.fixture(autouse=True)
def _reset_main_globals():
    """Snapshot and restore the module globals reconcile() touches."""
    keys = ("_state_model", "_predictor", "_state_store", "_remediation_manager",
            "_notifier", "_reconcile_count", "_last_reconcile_time",
            "_last_reconcile_duration", "_state_saves", "_model_decay_counter",
            "_policy_engine", "_llm_analyzer", "_kg_client", "_backtester",
            "_shadow_harness", "_llm_calls", "_llm_errors")
    saved = {k: getattr(main_mod, k) for k in keys}
    main_mod._reconcile_count = 0
    main_mod._state_saves = 0
    main_mod._model_decay_counter = 0
    yield
    for k, v in saved.items():
        setattr(main_mod, k, v)


def test_patch_get_pods_json_intercepts_reconcile():
    main_mod._state_model = StateModel()
    main_mod._predictor = Predictor(risk_threshold=1.5)  # nothing is "at risk"
    pods = [{"metadata": {"name": "p1", "namespace": "default"},
             "status": {"phase": "Running", "containerStatuses": []},
             "spec": {"nodeName": "n1"}}]
    with patch.object(main_mod, "detect_runtime", return_value="kubernetes"), \
         patch.object(main_mod, "run_cmd", return_value=(0, "", "")), \
         patch.object(main_mod, "collect_top_metrics", return_value={}), \
         patch.object(main_mod, "collect_top_nodes", return_value=None), \
         patch.object(main_mod, "get_node_conditions", return_value={}), \
         patch.object(main_mod, "count_log_errors", return_value=0), \
         patch.object(main_mod, "_get_pods_json", return_value={"items": pods}) as mock_pods, \
         patch.object(main_mod, "get_collector_state", return_value="HEALTHY"):
        summary = main_mod.reconcile()
    assert mock_pods.called
    assert "default/p1" in summary["state"]["pods"]
    assert summary["reconcile_count"] == 1


def test_patch_collect_docker_metrics_intercepts_reconcile():
    main_mod._state_model = StateModel()
    main_mod._predictor = Predictor(risk_threshold=1.5)
    metrics = {"abcd12345678": {"cpu_m": 250, "memory_mib": 512}}
    with patch.object(main_mod, "detect_runtime", return_value="docker"), \
         patch.object(main_mod, "discover_docker_containers", return_value={}), \
         patch.object(main_mod, "_collect_docker_container_metrics",
                      return_value=metrics) as mock_metrics, \
         patch.object(main_mod, "get_collector_state", return_value="DEGRADED"):
        summary = main_mod.reconcile()
    assert mock_metrics.called
    # empty containers + DEGRADED → no pods tracked, but no crash
    assert summary["pods_tracked"] == 0


def test_reconcile_summary_contract_keys():
    main_mod._state_model = StateModel()
    main_mod._predictor = Predictor(risk_threshold=1.5)
    with patch.object(main_mod, "detect_runtime", return_value="kubernetes"), \
         patch.object(main_mod, "run_cmd", return_value=(0, "", "")), \
         patch.object(main_mod, "collect_top_metrics", return_value={}), \
         patch.object(main_mod, "collect_top_nodes", return_value=None), \
         patch.object(main_mod, "get_node_conditions", return_value={}), \
         patch.object(main_mod, "count_log_errors", return_value=0), \
         patch.object(main_mod, "_get_pods_json", return_value={"items": []}), \
         patch.object(main_mod, "get_collector_state", return_value="HEALTHY"):
        summary = main_mod.reconcile()
    assert set(summary) >= {"predictions", "state", "timestamp", "pods_tracked",
                            "at_risk_count", "reconcile_count"}


def test_reconcile_counter_advances():
    main_mod._state_model = StateModel()
    main_mod._predictor = Predictor(risk_threshold=1.5)
    patches = [
        patch.object(main_mod, "detect_runtime", return_value="kubernetes"),
        patch.object(main_mod, "run_cmd", return_value=(0, "", "")),
        patch.object(main_mod, "collect_top_metrics", return_value={}),
        patch.object(main_mod, "collect_top_nodes", return_value=None),
        patch.object(main_mod, "get_node_conditions", return_value={}),
        patch.object(main_mod, "count_log_errors", return_value=0),
        patch.object(main_mod, "_get_pods_json", return_value={"items": []}),
        patch.object(main_mod, "get_collector_state", return_value="HEALTHY"),
    ]
    for p in patches:
        p.start()
    try:
        main_mod.reconcile()
        assert main_mod._reconcile_count == 1
        main_mod.reconcile()
        assert main_mod._reconcile_count == 2
    finally:
        for p in patches:
            p.stop()
