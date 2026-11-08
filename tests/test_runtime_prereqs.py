"""Tests for the startup fail-fast gate: `main._check_runtime_prerequisites`.

Regression guard for the silent-blindness incident: a kubectl-mode image
shipped WITHOUT the kubectl binary ran for weeks tracking 0 pods with no
warnings, no degraded collector state and a green /ready. The gate turns that
misconfiguration into a loud startup failure (SystemExit) instead.

NOTE: conftest.py's hermetic baseline replaces `_check_runtime_prerequisites`
with a no-op so tests that call `main.main()` never hard-exit. These tests
re-instate the REAL function (captured at import time) per test.
"""

import pytest

import predictive_agent.config as config
import predictive_agent.main as main_mod
from predictive_agent import pod_collection

# The real gate, captured BEFORE the autouse conftest fixture swaps it for a
# no-op (module import happens before any fixture runs).
_REAL_GATE = main_mod._check_runtime_prerequisites


@pytest.fixture(autouse=True)
def _restore_real_gate(monkeypatch):
    monkeypatch.setattr(main_mod, "_check_runtime_prerequisites", _REAL_GATE)
    yield


def test_kubectl_mode_without_binary_fails_fast(monkeypatch, caplog):
    """kubectl-mode + missing binary must SystemExit with a CRITICAL log."""
    monkeypatch.setattr(config, "COLLECTOR_MODE", "kubectl")
    monkeypatch.setattr(pod_collection, "kubectl_binary_missing", lambda: True)
    with caplog.at_level("CRITICAL", logger="predictive-agent"):
        with pytest.raises(SystemExit) as exc:
            _REAL_GATE()
    assert exc.value.code == 1
    assert "kubectl" in caplog.text
    assert "COLLECTOR_MODE=kubectl" in caplog.text


def test_kubectl_mode_with_binary_passes(monkeypatch):
    """kubectl-mode + binary present must not raise."""
    monkeypatch.setattr(config, "COLLECTOR_MODE", "kubectl")
    monkeypatch.setattr(pod_collection, "kubectl_binary_missing", lambda: False)
    _REAL_GATE()  # must not raise


def test_docker_mode_without_binary_passes(monkeypatch):
    """docker collector doesn't need kubectl — no fail-fast there."""
    monkeypatch.setattr(config, "COLLECTOR_MODE", "docker")
    monkeypatch.setattr(pod_collection, "kubectl_binary_missing", lambda: True)
    _REAL_GATE()  # must not raise
