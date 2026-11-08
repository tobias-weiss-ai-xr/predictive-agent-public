"""Test main reconcile loop — degraded-mode and docker-mode error paths.

PA-T1: reconcile() error paths.

These tests exercise the resilience of ``predictive_agent.main.reconcile``
against collection failures and docker-shaped inputs:

1. Collection failure (kubectl unavailable) degrades gracefully: reconcile
   still returns a predictions dict and keeps the last-known state model /
   prediction data instead of crashing.
2. Docker-style pod records (namespace ``docker`` or a compose project used
   as the namespace) get tracked with ``<namespace>/<name>`` pod keys.
3. An empty container list is a healthy cycle (0 pods tracked), not an error.
4. Pod items with missing or ``None`` ``creationTimestamp`` do not crash.
5. ``reconcile`` always returns a dict containing a ``predictions`` key.

Patching follows the style of ``tests/test_main.py``: collector/helper
functions are mocked on the ``predictive_agent.main`` module object, and every
test resets the module globals it mutates via a fixture.
"""

import json
import logging
from unittest.mock import patch

import pytest

import predictive_agent.config as config
import predictive_agent.main as main_mod

# Module globals mutated by reconcile() that each test must reset.
_MUTATED_GLOBALS = (
    "_state_model",
    "_predictor",
    "_state_store",
    "_remediation_manager",
    "_notifier",
    "_reconcile_count",
    "_last_reconcile_time",
    "_last_reconcile_duration",
    "_cache",
    "_history",
    "_state_saves",
)


# ─── Helpers (docker-style fixtures) ────────────────────────────────────────

def _discover_docker_containers():
    """Return fake Docker containers, mimicing a docker-mode discovery call.

    Each record carries a ``compose_project`` field that a docker-mode
    collector would use as the pod namespace.
    """
    return [
        {
            "name": "web",
            "compose_project": "shop",
            "cpu_m": 120,
            "memory_mib": 256,
            "memory_limit_mib": 512,
            "restart_count": 0,
            "node_pressure": False,
            "log_errors": 0,
        },
        {
            "name": "db",
            "compose_project": "shop",
            "cpu_m": 40,
            "memory_mib": 1024,
            "memory_limit_mib": 2048,
            "restart_count": 2,
            "node_pressure": True,
            "log_errors": 3,
        },
    ]


def _docker_containers_to_pod_items(containers):
    """Normalize docker container records into pod-like JSON items.

    The compose project is used as the namespace, so pod keys end up as
    ``<compose_project>/<name>`` inside reconcile().
    """
    items = []
    for container in containers:
        items.append({
            "metadata": {
                "namespace": container.get("compose_project", "default"),
                "name": container.get("name", ""),
                "creationTimestamp": None,
            },
            "spec": {
                "nodeName": "docker-host",
                "containers": [
                    {"name": container.get("name", ""), "resources": {}}
                ],
            },
            "status": {
                "phase": "Running",
                "containerStatuses": [
                    {
                        "name": container.get("name", ""),
                        "ready": True,
                        "restartCount": container.get("restart_count", 0),
                    }
                ],
            },
            "docker_container": container,
        })
    return items


def _fake_run_cmd(code, output):
    """Build a run_cmd replacement that always returns (code, output, '')."""
    def _run(cmd, timeout=None):
        return (code, output, "")
    return _run


def _make_pod(namespace="default", name="web-0", creation_timestamp="2025-01-01T00:00:00Z"):
    """Build a minimal pod JSON item with a valid creationTimestamp."""
    metadata = {"namespace": namespace, "name": name}
    if creation_timestamp is not _MISSING:
        metadata["creationTimestamp"] = creation_timestamp
    return {
        "metadata": metadata,
        "spec": {
            "nodeName": "node-1",
            "containers": [
                {
                    "name": name,
                    "resources": {
                        "limits": {
                            "memory": "512Mi" if name == "web-0" else "1Gi"
                        }
                    },
                }
            ],
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [
                {"name": name, "ready": True, "restartCount": 0}
            ],
        },
    }


class _Missing:
    """Sentinel for "attribute absent" in helper kwargs."""


_MISSING = _Missing()


def _initialized_defaults():
    """Set the module globals to a clean, initialized tracker setup."""
    from predictive_agent.predictor import Predictor
    from predictive_agent.state_model import StateModel
    main_mod._state_model = StateModel()
    main_mod._predictor = Predictor()
    main_mod._state_store = None
    main_mod._remediation_manager = None
    main_mod._notifier = None


# ─── Fixtures ───────────────────────────────────────────────────────────────

# detect_runtime is pinned to "kubernetes" suite-wide in tests/conftest.py;
# docker-path tests override it explicitly per-test.

@pytest.fixture(autouse=True)
def reset_main_globals():
    """Reset every module global that reconcile() mutates.

    Snapshots the values before each test and restores them afterwards so a
    failed/aborted test can never leak state into later tests.
    """
    import predictive_agent.main as m

    snapshot = {name: getattr(m, name) for name in _MUTATED_GLOBALS}
    yield
    for name, value in snapshot.items():
        setattr(m, name, value)


@pytest.fixture
def initialized():
    """Set up a clean StateModel + Predictor on the main module."""
    _initialized_defaults()
    yield


# ─── (5) reconcile returns a dict containing predictions ────────────────────

def test_reconcile_returns_predictions_dict(initialized):
    """reconcile returns a dict that always contains a `predictions` key."""
    pod_items = {"items": [_make_pod()]}
    with patch.object(
        main_mod, "run_cmd",
        side_effect=_fake_run_cmd(0, "default/web-0  50m   256Mi\n"),
    ), patch.object(main_mod, "_get_pods_json", return_value=pod_items), patch.object(
        main_mod, "get_node_conditions",
        return_value={"node-1": {"Ready": "True", "MemoryPressure": "False"}},
    ):
        result = main_mod.reconcile()

    assert isinstance(result, dict)
    assert "predictions" in result
    assert isinstance(result["predictions"], list)
    assert "state" in result
    assert "timestamp" in result
    assert result["pods_tracked"] == 1


def test_reconcile_uninitialized_returns_prediction_dict():
    """Even without a state model reconcile returns a predictions dict."""
    with patch.object(
        main_mod, "run_cmd", side_effect=_fake_run_cmd(1, "")
    ), patch.object(main_mod, "_get_pods_json", return_value={"items": []}):
        result = main_mod.reconcile()
    assert isinstance(result, dict)
    assert "predictions" in result
    assert result["predictions"] == []
    assert result["pods_tracked"] == 0


# ─── (1) Collection failure → degraded mode keeps last-known state ──────────

def test_collection_failure_still_returns_predictions_dict(initialized):
    """kubectl unavailable (rc != 0) must not crash; still returns predictions."""
    with patch.object(
        main_mod, "run_cmd", side_effect=_fake_run_cmd(1, "")
    ), patch.object(main_mod, "_get_pods_json", return_value={"items": []}):
        result = main_mod.reconcile()

    assert isinstance(result, dict)
    assert "predictions" in result
    assert result["predictions"] == []
    assert result["pods_tracked"] == 0
    assert result["at_risk_count"] == 0


def test_collection_failure_keeps_last_known_state(initialized):
    """A degraded cycle keeps the last-known trackers and predictions.

    First run a healthy cycle so the state model holds a tracker and the
    predictor holds a prediction; then run a cycle where every collection
    fails. The tracker and the last-known prediction must survive.
    """
    pod_items = {"items": [_make_pod()]}
    with patch.object(
        main_mod, "run_cmd",
        side_effect=_fake_run_cmd(0, "default/web-0  50m   256Mi\n"),
    ), patch.object(main_mod, "_get_pods_json", return_value=pod_items), patch.object(
        main_mod, "get_node_conditions",
        return_value={"node-1": {"Ready": "True", "MemoryPressure": "False"}},
    ):
        healthy = main_mod.reconcile()

    assert healthy["pods_tracked"] == 1
    assert "default/web-0" in main_mod._state_model.pods
    last_predictions = [p["pod_key"] for p in healthy["predictions"]]
    assert "default/web-0" in last_predictions

    # Degraded cycle — collection fails and the collector state machine is
    # in DEGRADED (prior success + failure), so reconcile must keep the
    # last-known tracker and prediction instead of pruning them.
    with patch.object(
        main_mod, "run_cmd", side_effect=_fake_run_cmd(1, "")
    ), patch.object(main_mod, "_get_pods_json", return_value={"items": []}), patch.object(
        main_mod, "get_collector_state", return_value="DEGRADED"
    ):
        degraded = main_mod.reconcile()

    assert isinstance(degraded, dict)
    assert "predictions" in degraded
    # DEGRADED keep-last-known: the previously seen pod is still tracked.
    assert degraded["pods_tracked"] == 1
    # Last-known state preserved across the degraded cycle.
    assert "default/web-0" in main_mod._state_model.pods
    degraded_predictions = [p["pod_key"] for p in degraded["predictions"]]
    assert "default/web-0" in degraded_predictions


def test_collector_raise_returns_predictions(initialized):
    """A collector exception inside the guarded collection block is contained.

    get_node_conditions raising a JSONDecodeError (the guarded block in
    reconcile) must not crash the cycle — reconcile still returns the
    predictions dict and keeps processing pods.
    """
    pod_items = {"items": [_make_pod()]}
    with patch.object(
        main_mod, "run_cmd",
        side_effect=_fake_run_cmd(0, "default/web-0  50m   256Mi\n"),
    ), patch.object(main_mod, "_get_pods_json", return_value=pod_items), patch.object(
        main_mod, "get_node_conditions",
        side_effect=json.JSONDecodeError("boom", "doc", 0),
    ):
        result = main_mod.reconcile()

    assert isinstance(result, dict)
    assert "predictions" in result
    assert result["pods_tracked"] == 1
    assert "default/web-0" in main_mod._state_model.pods


def test_degraded_uninitialized_logs_warning(caplog):
    """Collection failure with no state model logs a degraded warning."""
    with patch.object(main_mod, "_state_model", None), patch.object(
        main_mod, "run_cmd", side_effect=_fake_run_cmd(1, "")
    ), patch.object(main_mod, "_get_pods_json", return_value={"items": []}), caplog.at_level(
        logging.WARNING, logger="predictive-agent"
    ):
        main_mod.reconcile()

    assert any(
        "State model not initialized" in record.message
        for record in caplog.records
        if record.levelno >= logging.WARNING
    )


# ─── (2) Docker runtime path: docker/<name> keys, compose project ns ────────

def test_docker_namespace_pods_tracked_with_docker_name_keys(initialized):
    """Pods from a 'docker' namespace get tracked as 'docker/<name>' keys."""
    pod_items = {
        "items": [
            _make_pod(namespace="docker", name="nginx-1"),
            _make_pod(namespace="docker", name="redis-0"),
        ]
    }
    with patch.object(
        main_mod, "run_cmd",
        side_effect=_fake_run_cmd(
            0, "docker/nginx-1  20m   64Mi\ndocker/redis-0  15m   32Mi\n"
        ),
    ), patch.object(main_mod, "_get_pods_json", return_value=pod_items), patch.object(
        config, "WATCH_NAMESPACES", "docker"
    ):
        result = main_mod.reconcile()

    assert result["pods_tracked"] == 2
    assert "docker/nginx-1" in main_mod._state_model.pods
    assert "docker/redis-0" in main_mod._state_model.pods
    prediction_keys = {p["pod_key"] for p in result["predictions"]}
    assert "docker/nginx-1" in prediction_keys
    assert "docker/redis-0" in prediction_keys


def test_compose_project_used_as_namespace(initialized):
    """discover_docker_containers-style records track with compose project ns.

    Docker containers carry a ``compose_project`` field; when normalized into
    pod JSON, that project becomes the namespace, so pod keys are
    ``<compose_project>/<name>``.
    """
    containers = _discover_docker_containers()
    pod_items = {"items": _docker_containers_to_pod_items(containers)}

    with patch.object(
        main_mod, "run_cmd",
        side_effect=_fake_run_cmd(0, "shop/web  50m   256Mi\nshop/db  40m   1024Mi\n"),
    ), patch.object(main_mod, "_get_pods_json", return_value=pod_items), patch.object(
        config, "WATCH_NAMESPACES", ""
    ):
        result = main_mod.reconcile()

    assert result["pods_tracked"] == 2
    assert "shop/web" in main_mod._state_model.pods
    assert "shop/db" in main_mod._state_model.pods
    prediction_keys = {p["pod_key"] for p in result["predictions"]}
    assert "shop/web" in prediction_keys
    assert "shop/db" in prediction_keys


# ─── (3) Empty container list is healthy, 0 pods, not an error ──────────────

def test_empty_container_list_healthy_zero_pods(initialized):
    """An empty container/pod list is a healthy cycle, not an error."""
    with patch.object(
        main_mod, "run_cmd", side_effect=_fake_run_cmd(0, "")
    ), patch.object(main_mod, "_get_pods_json", return_value={"items": []}):
        result = main_mod.reconcile()

    assert isinstance(result, dict)
    assert "predictions" in result
    assert result["predictions"] == []
    assert result["pods_tracked"] == 0
    assert result["at_risk_count"] == 0
    assert result["state"] is not None


# ─── (4) Missing / None creationTimestamp must not crash ────────────────────

def test_missing_creation_timestamp_does_not_crash(initialized):
    """Pod items without a creationTimestamp key are processed normally."""
    pod_items = {"items": [_make_pod(creation_timestamp=_MISSING)]}
    with patch.object(
        main_mod, "run_cmd",
        side_effect=_fake_run_cmd(0, "default/web-0  50m   256Mi\n"),
    ), patch.object(main_mod, "_get_pods_json", return_value=pod_items), patch.object(
        main_mod, "get_node_conditions",
        return_value={"node-1": {"Ready": "True", "MemoryPressure": "False"}},
    ):
        result = main_mod.reconcile()

    assert result["pods_tracked"] == 1
    assert "default/web-0" in main_mod._state_model.pods


def test_none_creation_timestamp_does_not_crash(initialized):
    """Pod items with creationTimestamp=None are processed normally."""
    pod_items = {"items": [_make_pod(creation_timestamp=None)]}
    with patch.object(
        main_mod, "run_cmd",
        side_effect=_fake_run_cmd(0, "default/web-0  50m   256Mi\n"),
    ), patch.object(main_mod, "_get_pods_json", return_value=pod_items), patch.object(
        main_mod, "get_node_conditions",
        return_value={"node-1": {"Ready": "True", "MemoryPressure": "False"}},
    ):
        result = main_mod.reconcile()

    assert result["pods_tracked"] == 1
    assert "default/web-0" in main_mod._state_model.pods


# ─── Reconcile bookkeeping (bonus coverage) ─────────────────────────────────

def test_reconcile_count_increments_across_cycles(initialized):
    """Each reconcile cycle increments the module-level reconcile counter."""
    before = main_mod._reconcile_count if hasattr(main_mod, "_reconcile_count") else 0
    with patch.object(
        main_mod, "run_cmd", side_effect=_fake_run_cmd(1, "")
    ), patch.object(main_mod, "_get_pods_json", return_value={"items": []}):
        first = main_mod.reconcile()
        second = main_mod.reconcile()

    # Relative: other tests in the suite may have already bumped the counter.
    assert second["reconcile_count"] == first["reconcile_count"] + 1
    assert first["reconcile_count"] >= before


def test_reconcile_skip_namespace_not_tracked(initialized):
    """Pods in the skip list are filtered out and never tracked."""
    pod_items = {"items": [_make_pod(namespace="opendesk-predictive-agent", name="ignored")]}
    with patch.object(
        main_mod, "run_cmd",
        side_effect=_fake_run_cmd(0, "opendesk-predictive-agent/ignored  10m   32Mi\n"),
    ), patch.object(main_mod, "_get_pods_json", return_value=pod_items):
        result = main_mod.reconcile()

    assert result["pods_tracked"] == 0
    assert main_mod._state_model.pods == {}


def test_reconcile_reused_state_across_cycles(initialized):
    """Pods stay tracked across consecutive cycles (per-pod state retained)."""
    pod_items = {"items": [_make_pod()]}
    with patch.object(
        main_mod, "run_cmd",
        side_effect=_fake_run_cmd(0, "default/web-0  50m   256Mi\n"),
    ), patch.object(main_mod, "_get_pods_json", return_value=pod_items), patch.object(
        main_mod, "get_node_conditions",
        return_value={"node-1": {"Ready": "True", "MemoryPressure": "False"}},
    ):
        main_mod.reconcile()

    assert "default/web-0" in main_mod._state_model.pods

    with patch.object(
        main_mod, "run_cmd",
        side_effect=_fake_run_cmd(0, "default/web-0  60m   300Mi\n"),
    ), patch.object(main_mod, "_get_pods_json", return_value=pod_items), patch.object(
        main_mod, "get_node_conditions",
        return_value={"node-1": {"Ready": "True", "MemoryPressure": "False"}},
    ):
        second = main_mod.reconcile()

    assert second["pods_tracked"] == 1
    # One tracker, still at the same key — Kalman/state history is retained;
    # we did not create a duplicate tracker.
    assert list(main_mod._state_model.pods.keys()) == ["default/web-0"]
