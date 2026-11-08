"""Tests for the agent startup path (main.main) — Docker vs Kubernetes
action registration.

Covers the branch that the live dry-run canary exercised against a real
Docker host: on ``runtime == "docker"`` the agent registers a Docker-socket
ContainerRestartAction as ``pod_restart`` plus an extra ``container_restart``
action; on Kubernetes it registers the kubectl PodRestartAction (also named
``pod_restart``) and must NOT register the Docker-only ``container_restart``.
"""
from unittest import mock

import pytest

import predictive_agent.main as m


class _RecordingManager:
    """Fake RemediationManager that records registered action names."""

    def __init__(self):
        self.registered = []
        self.action_costs = {}
        self.failure_cost = 1.0
        self.rl_policy = None
        self.calibration_monitor = None
        self.dry_run = True
        self.risk_threshold = 0.7

    def register_action(self, action):
        self.registered.append((action.name, type(action).__name__))

    def get_stats(self):
        return {"registered_actions": [n for n, _ in self.registered]}


def _run_main(manager, runtime):
    """Run main.main() with side-effectful pieces patched, using real action
    classes so ``name`` attributes are truthful."""
    with mock.patch.object(m, "detect_runtime", return_value=runtime), \
         mock.patch.object(m, "get_watch_selectors", return_value={}), \
         mock.patch.object(m, "StateStore"), \
         mock.patch.object(m, "create_remediation_manager_from_config", return_value=manager), \
         mock.patch.object(m, "create_notifier_from_config", return_value=None), \
         mock.patch.object(m, "get_kg_client", return_value=None), \
         mock.patch.object(m, "_start_http_server", return_value=None), \
         mock.patch.object(m, "ReconcileLoop"), \
         mock.patch.object(m.signal, "signal"), \
         mock.patch.object(m.time, "sleep", side_effect=SystemExit(0)), \
         mock.patch.object(m, "DockerSocketClient"), \
         mock.patch.object(m, "StateModel"), \
         mock.patch.object(m, "Predictor"), \
         mock.patch.object(m, "ActionAwareTransitionModel"), \
         mock.patch.object(m, "Backtester"):
        m.main()


EXPECTED_COMMON = {"node_cordon", "right_size", "rollout_restart", "scale", "tune_resources"}


_MUTATED_GLOBALS = [
    "_server", "_remediation_manager", "_notifier", "_llm_calls", "_llm_errors",
    "_state_saves", "_model_decay_counter", "_last_plan_notify", "_last_autonomy_notify",
    "_kg_client", "_backtester", "_llm_analyzer", "_llm_analysis_cache",
    "_llm_in_flight", "_state_model", "_predictor", "_state_store",
    "_decision_model", "_rl_policy", "_calibration_monitor", "_shadow_harness",
    "_last_reconcile_duration",
]


@pytest.fixture(autouse=True)
def _preserve_module_state():
    """Restore every module global that main() mutates, so these startup tests
    never leak state into other tests running in the same process."""
    saved = {name: getattr(m, name, None) for name in _MUTATED_GLOBALS}
    yield
    for name, value in saved.items():
        if hasattr(m, name):
            setattr(m, name, value)


def test_startup_docker_registers_docker_actions():
    """runtime==docker must register a Docker-socket ContainerRestartAction as
    pod_restart and an extra ContainerRestartAction."""
    manager = _RecordingManager()
    _run_main(manager, runtime="docker")
    names = set(n for n, _ in manager.registered)
    assert "pod_restart" in names
    assert "container_restart" in names
    # All the common runtime-agnostic actions are registered too.
    assert names >= EXPECTED_COMMON
    # The docker pod_restart must be the Docker ContainerRestartAction, not k8s.
    pod_entry = next((t for n, t in manager.registered if n == "pod_restart"), None)
    assert pod_entry == "ContainerRestartAction"


def test_startup_kubernetes_registers_pod_restart():
    """runtime==kubernetes must register the kubectl PodRestartAction as
    pod_restart and NOT the Docker-only ContainerRestartAction."""
    manager = _RecordingManager()
    _run_main(manager, runtime="kubernetes")
    names = set(n for n, _ in manager.registered)
    assert "pod_restart" in names
    assert "container_restart" not in names
    assert names >= EXPECTED_COMMON
    pod_entry = next((t for n, t in manager.registered if n == "pod_restart"), None)
    assert pod_entry == "PodRestartAction"
