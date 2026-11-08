"""Pytest configuration for predictive-agent tests."""
import os
import subprocess
import sys

# Add project root to path so we can import predictive_agent package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

# Binaries tests must never invoke for real — they would make the suite
# host-dependent (green on docker-less CI, red wherever a daemon answers).
# Mock them instead; see tests/test_main_degraded.py (detect_runtime pin)
# and tests/test_container_restart.py (subprocess.run mock) for the patterns.
_FORBIDDEN_BINARIES = {"docker", "kubectl", "docker-compose"}
_real_subprocess_run = subprocess.run


def _guarded_run(cmd, *args, **kwargs):
    head = cmd[0] if isinstance(cmd, (list, tuple)) and cmd else str(cmd).split(maxsplit=1)[0]
    if os.path.basename(str(head)) in _FORBIDDEN_BINARIES:
        # RuntimeError (not pytest.fail): the caller may run us inside a worker
        # thread whose `except Exception` must be able to handle the refusal.
        raise RuntimeError(
            f"test invoked the real {head!r} binary — mock the container-runtime "
            f"boundary instead: patch run_cmd/subprocess.run or pin detect_runtime "
            f"(see tests/conftest.py)"
        )
    return _real_subprocess_run(cmd, *args, **kwargs)


@pytest.fixture(autouse=True)
def pin_kubernetes_runtime(monkeypatch):
    """Give every test a hermetic cluster/runtime baseline.

    - main.detect_runtime() pinned to "kubernetes" (docker-path tests
      override explicitly per-test; an explicit patch wins for the
      duration of its context).
    - main.run_cmd() defaults to rc=1, i.e. "no cluster reachable" — the
      state the suite was written against (degraded collection). Tests
      that need kubectl output patch run_cmd themselves.

    Without this baseline, any host with a Docker daemon or a working
    kubeconfig silently routes tests against real infrastructure
    (regression that hit pcrz1334: reconcile collected live containers
    from the workstation daemon).
    """
    import predictive_agent.main as _main
    monkeypatch.setattr(_main, "detect_runtime", lambda: "kubernetes")
    monkeypatch.setattr(_main, "run_cmd", lambda cmd, timeout=10: (1, "", "hermetic baseline: no cluster"))
    # Never let the startup fail-fast gate (kubectl-missing) blow up tests that
    # exercise main.main() with a mock runtime. Tests that target the gate
    # itself re-instate the real function (tests/test_runtime_prereqs.py).
    monkeypatch.setattr(_main, "_check_runtime_prerequisites", lambda: None)
    # kubectl_binary_missing() checks the REAL PATH via shutil.which (not the
    # mocked run_cmd), so without pinning it the suite outcome depends on
    # whether the host happens to have kubectl installed: on a runner without
    # kubectl, get_pods_json() raises KubectlUnavailableError before the mocked
    # kubectl output is ever read -> 0 pods tracked -> /ready 503, killing the
    # reconcile/rationale/server tests (mac runner before 2026-11-08). Pin it to
    # "present" so mocked run_cmd output drives collection everywhere. Tests that
    # target the missing-binary path re-patch it per test
    # (tests/test_runtime_prereqs.py, tests/test_pod_collection.py), which
    # overrides this autouse fixture.
    import predictive_agent.pod_collection as _pod_collection
    monkeypatch.setattr(_pod_collection, "kubectl_binary_missing", lambda: False)


@pytest.fixture(autouse=True)
def reset_runtime_cache():
    """Clear collector._RUNTIME_CACHE around every test.

    detect_runtime() caches its answer module-globally; a test that runs with
    a patched environ (or on a docker host) would otherwise pin the cache for
    all later tests. Reset before AND after so nothing leaks in or out.
    """
    import predictive_agent.collector as _collector
    _collector._RUNTIME_CACHE = None
    yield
    _collector._RUNTIME_CACHE = None


@pytest.fixture(autouse=True)
def reset_collector_state():
    """Restore the pristine collector state (HEALTHY, no prior) around every test.

    _COLLECTOR_STATE is a module global mutated by reconcile-driven tests
    (record_collection_failure DEGRADEs/DOWNs it). Without a reset, the
    collector-state leaks across tests: a reconcile test earlier in the suite
    leaves it down, so a later server test asserting /ready -> 200 gets 503
    (failed on the no-kubectl mac runner as tests/test_concurrency.py). Tests
    that exercise degraded/healthy states set it explicitly themselves
    (tests/test_degraded_mode.py, tests/test_contract.py). Reset before AND
    after so nothing leaks in or out.
    """
    import predictive_agent.collector as _collector
    _collector.set_collector_state(
        _collector.COLLECTOR_STATE_HEALTHY, has_prior=False, last_collect=None
    )
    yield
    _collector.set_collector_state(
        _collector.COLLECTOR_STATE_HEALTHY, has_prior=False, last_collect=None
    )


@pytest.fixture(autouse=True)
def no_real_container_runtime(monkeypatch):
    """Fail any test that shells out to the real docker/kubectl binaries."""
    monkeypatch.setattr(subprocess, "run", _guarded_run)
