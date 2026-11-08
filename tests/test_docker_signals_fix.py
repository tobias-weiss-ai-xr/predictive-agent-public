"""Tests for the v4.0.8 docker signal + LLM-failure-visibility fixes.

* ``get_container_status_signals`` must read the top-level ``RestartCount``
  that modern Docker puts on the inspect JSON (``State.RestartCount`` is null).
* ``docker_status_signals`` converts the rich ``container_info`` dict produced
  by ``discover_docker_containers`` into the pod-style signal dict the reconcile
  loop consumes — docker pods must no longer appear unconditionally Running/ready.
* ``LLMAnalyzer`` logs a WARNING carrying the underlying error whenever it
  returns a fallback analysis (previously every failure was silent at INFO).
"""
import logging
import urllib.error
from unittest.mock import patch

from pytest import LogCaptureFixture

from predictive_agent.collector import (
    docker_status_signals,
    get_container_status_signals,
)
from predictive_agent.llm import LLMAnalyzer, LLMBackend

# ─── get_container_status_signals: top-level RestartCount ──────────────────


def test_container_status_signals_reads_top_level_restart_count():
    """Modern docker inspect stores RestartCount top-level, State has none."""
    inspect_json = {
        "RestartCount": 7,
        "State": {
            "Running": True,
            "Paused": False,
            "Restarting": False,
            "Dead": False,
            "OOMKilled": False,
            "ExitCode": None,
        },
    }
    sig = get_container_status_signals(inspect_json)
    assert sig["restart_count"] == 7


def test_container_status_signals_restarting_keeps_restart_count():
    """A crash-looping container reports its real RestartCount."""
    inspect_json = {
        "RestartCount": 312,
        "State": {
            "Running": True,
            "Paused": False,
            "Restarting": True,
            "Dead": False,
            "OOMKilled": False,
            "ExitCode": None,
        },
    }
    sig = get_container_status_signals(inspect_json)
    assert sig["restart_count"] == 312
    assert sig["wait_state"] == "Restarting"
    assert sig["terminated"] is True
    assert sig["ready"] is False


# ─── docker_status_signals: container_info -> pod-style signals ─────────────


def test_docker_status_signals_running():
    """Running container without healthcheck: ready, not terminated."""
    info = {"status": "running", "healthy": None, "restart_count": 0, "exit_code": None}
    sig = docker_status_signals(info)
    assert sig == {
        "pod_phase": "Running",
        "container_ready": True,
        "wait_state": None,
        "terminated": False,
        "terminated_reason": None,
        "restart_count": 0,
    }


def test_docker_status_signals_running_unhealthy():
    """Running container with a failing healthcheck: not ready."""
    info = {"status": "running", "healthy": False, "restart_count": 0, "exit_code": None}
    sig = docker_status_signals(info)
    assert sig["pod_phase"] == "Running"
    assert sig["container_ready"] is False
    assert sig["terminated"] is False


def test_docker_status_signals_restarting():
    """Crash-looping container: wait_state Restarting, not ready, terminated."""
    info = {"status": "restarting", "healthy": None, "restart_count": 42, "exit_code": 1}
    sig = docker_status_signals(info)
    assert sig["wait_state"] == "Restarting"
    assert sig["container_ready"] is False
    assert sig["terminated"] is True
    assert sig["restart_count"] == 42


def test_docker_status_signals_exited():
    """Exited container: Succeeded phase (skipped, mirrors k8s) + reason."""
    info = {"status": "exited", "healthy": None, "restart_count": 0, "exit_code": 1}
    sig = docker_status_signals(info)
    assert sig["pod_phase"] == "Succeeded"
    assert sig["terminated"] is True
    assert sig["terminated_reason"] == "Exited (1)"


def test_docker_status_signals_dead():
    """Dead container: Failed phase + terminated reason."""
    info = {"status": "dead", "healthy": None, "restart_count": 2, "exit_code": 137}
    sig = docker_status_signals(info)
    assert sig["pod_phase"] == "Failed"
    assert sig["terminated"] is True
    assert sig["terminated_reason"] == "Error"


# ─── LLM fallback visibility at WARNING level ───────────────────────────────


def _analyzer():
    return LLMAnalyzer(
        backend=LLMBackend.OPENAI,
        url="http://localhost:4000/v1",
        model="m",
        api_key="k",
        timeout=5,
    )


def test_llm_http_error_logs_at_warning(caplog: LogCaptureFixture):
    """HTTP rejection must surface the code in a WARNING, not stay silent."""
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.HTTPError("http://x", 400, "Bad Request", {}, None),
    ):
        with caplog.at_level(logging.WARNING, logger="predictive_agent.llm"):
            _analyzer().analyze(issue="Pod x/y at risk", context="cpu=1%", prediction={})
    assert any("400" in r.message for r in caplog.records)


def test_llm_timeout_logs_at_warning(caplog: LogCaptureFixture):
    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        with caplog.at_level(logging.WARNING, logger="predictive_agent.llm"):
            _analyzer().analyze(issue="Pod x/y at risk", context="cpu=1%", prediction={})
    assert any("timed out" in r.message.lower() for r in caplog.records)


def test_llm_connectivity_error_logs_at_warning(caplog: LogCaptureFixture):
    err = urllib.error.URLError("conn refused")
    with patch("urllib.request.urlopen", side_effect=err):
        with caplog.at_level(logging.WARNING, logger="predictive_agent.llm"):
            _analyzer().analyze(issue="Pod x/y at risk", context="cpu=1%", prediction={})
    assert any("conn refused" in r.message.lower() for r in caplog.records)


def test_llm_success_does_not_warn(caplog: LogCaptureFixture):
    """Successful calls stay quiet at WARNING level."""

    class _Resp:
        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    body = (
        b'{"id":"1","choices":[{"message":{"content":'
        b'"{\\"analysis\\":\\"ok\\",\\"severity\\":\\"low\\",'
        b'\\"action\\":\\"Monitor\\",\\"command\\":\\"\\"}"}}]}'
    )
    with patch("urllib.request.urlopen", return_value=_Resp(body)):
        with caplog.at_level(logging.WARNING, logger="predictive_agent.llm"):
            result = _analyzer().analyze(issue="x", context="c", prediction={})
    assert result["action"] == "Monitor"
    assert not caplog.records
