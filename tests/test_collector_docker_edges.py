"""Edge case tests for the Docker discovery parsing paths in collector.py.

Covers parsing-robustness edges of the docker-native monitoring feature:

* ``discover_docker_containers`` survives blank / garbage ``docker ps`` lines
* selector filtering excludes label, compose-project and name-wildcard
  mismatches
* ``_get_container_inspect`` failure falls back to ps-only info
* containers with null ``Labels`` / ``Names`` in the ps JSON are handled
* ``detect_runtime`` caches its result and honors ``OPERATOR_RUNTIME``
* ``get_watch_selectors`` matches fnmatch wildcards (e.g. ``predagent-*``)

Mirrors the patching style of ``tests/test_collector.py``: ``run_cmd`` is
patched with a ``MockCompletedProcess``-style side effect and the
``_RUNTIME_CACHE`` module global is reset between tests.
"""
import os
from unittest.mock import patch

import pytest

import predictive_agent.collector as collector
from predictive_agent.collector import (
    _matches_selectors,
    _parse_docker_ps_line,
    detect_runtime,
    discover_docker_containers,
    get_watch_selectors,
)

# ─── Shared helpers ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_runtime_cache():
    """Reset the module-level _RUNTIME_CACHE global before and after each test."""
    collector._RUNTIME_CACHE = None
    yield
    collector._RUNTIME_CACHE = None


class MockCompletedProcess:
    """Mock for subprocess.CompletedProcess (matches tests/test_collector.py)."""

    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def docker_env(**overrides):
    """Context manager: isolated environment forced into Docker mode.

    Clears the real environment (so stray OPERATOR_WATCH_* vars can never leak
    into a test) and layers the docker-mode + selector overrides on top.
    """
    env = {"OPERATOR_RUNTIME": "docker"}
    env.update(overrides)
    return patch.dict(os.environ, env, clear=True)


def make_run_mock(ps_output, inspect_by_id=None, inspect_fail_ids=()):
    """Build a ``run_cmd`` side effect that dispatches docker ps / docker inspect.

    ``ps_output`` is the full ``docker ps --format '{{json .}}'`` stdout.
    ``inspect_by_id`` maps a container ID to a full ``docker inspect`` stdout
    (a JSON list). IDs in ``inspect_fail_ids`` simulate inspect failures.
    """
    inspect_by_id = inspect_by_id or {}

    def mock_run_cmd(cmd, timeout=10, **kwargs):
        joined = " ".join(cmd)
        if "docker ps" in joined:
            return (0, ps_output, "")
        if "docker inspect" in joined:
            container_id = cmd[-1]
            if container_id in inspect_fail_ids:
                return (1, "", "inspect error")
            if container_id in inspect_by_id:
                return (0, inspect_by_id[container_id], "")
            return (1, "", "inspect: no such container")
        return (1, "", "")

    return mock_run_cmd


def inspect_json(container_id, **state_overrides):
    """Build a minimal valid ``docker inspect`` list for one container."""
    state = {
        "Running": True,
        "Paused": False,
        "Restarting": False,
        "Dead": False,
    }
    state.update(state_overrides)
    entry = (
        '{"Id": "%s", "Name": "/%s", "State": %s, '  # noqa: UP031
        '"Config": {"Labels": {}}, "NetworkSettings": {"Ports": {}, '
        '"Networks": {}}}'
        % (container_id, container_id, _json_dumps(state))
    )
    # docker inspect returns a LIST of objects, not a bare object.
    return "[" + entry + "]"


def _json_dumps(value):
    import json

    return json.dumps(value)


# ─── (1) discover_docker_containers: blank / garbage docker ps lines ────────


def test_discover_skips_blank_and_garbage_lines():
    """Blank, non-JSON and ID-less lines in docker ps output are skipped."""
    ps_output = "\n".join(
        [
            "",  # blank line
            "   ",  # whitespace-only line
            "not-json",  # not valid JSON at all
            "{invalid 'json' -> ",  # broken JSON
            '{"Image": "nginx"}',  # valid JSON but missing "ID"
            '{"ID": "abc123def456", "Image": "nginx", "Names": "web-1", '
            '"Status": "Up 5 minutes", "Labels": "app=web"}',
        ]
    )
    inspect = inspect_json("abc123def456")
    with docker_env():
        with patch("predictive_agent.collector.run_cmd") as mock_run:
            mock_run.side_effect = make_run_mock(ps_output, {"abc123def456": inspect})
            containers = discover_docker_containers()

    # Only the well-formed container survived, without any exception.
    assert len(containers) == 1
    assert "abc123def456" in containers
    assert containers["abc123def456"]["name"] == "web-1"


def test_parse_docker_ps_line_blank_and_garbage():
    """_parse_docker_ps_line returns None for blank / garbage lines."""
    assert _parse_docker_ps_line("") is None
    assert _parse_docker_ps_line("   \n") is None
    assert _parse_docker_ps_line("garbage-not-json") is None
    assert _parse_docker_ps_line("{broken") is None


# ─── (2) selector filtering: excluded containers ─────────────────────────────


def test_discover_excludes_label_mismatch():
    """Containers whose labels do not match the selector are excluded."""
    ps_output = "\n".join(
        [
            '{"ID": "abc123def456", "Image": "nginx", "Names": "web-1", '
            '"Status": "Up 5 minutes", "Labels": "app=web"}',
            '{"ID": "def456abc789", "Image": "redis", "Names": "cache-1", '
            '"Status": "Up 5 minutes", "Labels": "app=cache"}',
        ]
    )
    inspect = {
        "abc123def456": inspect_json("abc123def456"),
        "def456abc789": inspect_json("def456abc789"),
    }
    with docker_env(OPERATOR_WATCH_LABELS="app=web"):
        with patch("predictive_agent.collector.run_cmd") as mock_run:
            mock_run.side_effect = make_run_mock(ps_output, inspect)
            containers = discover_docker_containers()

    assert set(containers) == {"abc123def456"}
    assert containers["abc123def456"]["name"] == "web-1"


def test_discover_excludes_compose_project_mismatch():
    """Containers outside the configured compose projects are excluded."""
    ps_output = "\n".join(
        [
            '{"ID": "abc123def456", "Image": "nginx", "Names": "myapp-web-1", '
            '"Status": "Up 5 minutes", '
            '"Labels": "com.docker.compose.project=myapp,app=web"}',
            '{"ID": "def456abc789", "Image": "redis", "Names": "otherapp-cache-1", '
            '"Status": "Up 5 minutes", '
            '"Labels": "com.docker.compose.project=otherapp,app=cache"}',
        ]
    )
    inspect = {
        "abc123def456": inspect_json("abc123def456"),
        "def456abc789": inspect_json("def456abc789"),
    }
    with docker_env(OPERATOR_WATCH_COMPOSE_PROJECTS="myapp"):
        with patch("predictive_agent.collector.run_cmd") as mock_run:
            mock_run.side_effect = make_run_mock(ps_output, inspect)
            containers = discover_docker_containers()

    assert set(containers) == {"abc123def456"}
    assert containers["abc123def456"]["compose_project"] == "myapp"


def test_discover_excludes_name_wildcard_mismatch():
    """Containers that do not match the name wildcard are excluded."""
    ps_output = "\n".join(
        [
            '{"ID": "abc123def456", "Image": "nginx", "Names": "predagent-api-1", '
            '"Status": "Up 5 minutes", "Labels": "app=api"}',
            '{"ID": "def456abc789", "Image": "redis", "Names": "unrelated-1", '
            '"Status": "Up 5 minutes", "Labels": "app=cache"}',
        ]
    )
    inspect = {
        "abc123def456": inspect_json("abc123def456"),
        "def456abc789": inspect_json("def456abc789"),
    }
    with docker_env(OPERATOR_WATCH_NAMES="predagent-*"):
        with patch("predictive_agent.collector.run_cmd") as mock_run:
            mock_run.side_effect = make_run_mock(ps_output, inspect)
            containers = discover_docker_containers()

    assert set(containers) == {"abc123def456"}
    assert containers["abc123def456"]["name"] == "predagent-api-1"


# ─── (3) inspect failure falls back to ps-only info ──────────────────────────


def test_inspect_failure_falls_back_to_ps_info():
    """When docker inspect fails, ps-only info is used without crashing."""
    ps_output = (
        '{"ID": "abc123def456", "Image": "nginx", "Names": "web-1", '
        '"Status": "Up 5 minutes", "Labels": "app=web"}'
    )
    with docker_env():
        with patch("predictive_agent.collector.run_cmd") as mock_run:
            mock_run.side_effect = make_run_mock(
                ps_output, inspect_fail_ids=("abc123def456",)
            )
            containers = discover_docker_containers()

    assert len(containers) == 1
    container = containers["abc123def456"]
    # Info still derived from the ps line.
    assert container["name"] == "web-1"
    assert container["image"] == "nginx"
    # With no inspect data the state dict is empty -> status stays "unknown".
    assert container["status"] == "unknown"
    # restart_count defaults to 0 instead of crashing on a missing field.
    assert container["restart_count"] == 0


def test_inspect_json_error_falls_back_to_ps_info():
    """Invalid docker inspect output (not a JSON list) also falls back to ps info."""
    ps_output = (
        '{"ID": "abc123def456", "Image": "nginx", "Names": "web-1", '
        '"Status": "Up 5 minutes", "Labels": "app=web"}'
    )
    with docker_env():
        with patch("predictive_agent.collector.run_cmd") as mock_run:
            mock_run.side_effect = make_run_mock(
                ps_output, {"abc123def456": "[broken"}
            )
            containers = discover_docker_containers()

    assert len(containers) == 1
    assert containers["abc123def456"]["restart_count"] == 0


# ─── (4) null Labels and null Names in ps JSON ───────────────────────────────


def test_null_labels_and_names_handled():
    """A container with null Labels/Names falls back to inspect, no crash."""
    ps_output = (
        '{"ID": "abc123def456", "Image": "nginx", "Names": null, '
        '"Status": "Up 5 minutes", "Labels": null}'
    )
    inspect = (
        '[{"Id": "abc123def456", "Name": "/web-1", '
        '"State": {"Running": true, "Paused": false, "Restarting": false, '
        '"Dead": false}, "Config": {"Labels": {}}, '
        '"NetworkSettings": {"Ports": {}, "Networks": {}}}]'
    )
    with docker_env():
        with patch("predictive_agent.collector.run_cmd") as mock_run:
            mock_run.side_effect = make_run_mock(
                ps_output, {"abc123def456": inspect}
            )
            containers = discover_docker_containers()

    assert len(containers) == 1
    container = containers["abc123def456"]
    # Name comes from `docker inspect` once the ps "Names" is null.
    assert container["name"] == "web-1"
    # Null Labels parse to an empty dict.
    assert container["labels"] == {}
    assert container["compose_project"] is None


# ─── (5) detect_runtime caching + OPERATOR_RUNTIME override ──────────────────


def test_detect_runtime_caches_result():
    """detect_runtime caches its value; later calls skip re-checking."""
    collector._RUNTIME_CACHE = None
    with patch.dict(os.environ, {"OPERATOR_RUNTIME": "docker"}, clear=True):
        assert detect_runtime() == "docker"

    # Env now says kubernetes, but the cached "docker" value must win.
    with patch.dict(os.environ, {"OPERATOR_RUNTIME": "kubernetes"}, clear=True):
        assert detect_runtime() == "docker"
        assert detect_runtime() == "docker"  # stable across repeated calls


def test_detect_runtime_override_wins():
    """OPERATOR_RUNTIME override wins even when a Docker socket is available."""
    collector._RUNTIME_CACHE = None
    # DOCKER_SOCKET points at something that (very likely) exists; the
    # explicit override must be honored before the socket check.
    with patch.dict(
        os.environ,
        {"OPERATOR_RUNTIME": "kubernetes", "DOCKER_SOCKET": "/dev/null"},
        clear=True,
    ):
        assert detect_runtime() == "kubernetes"

    collector._RUNTIME_CACHE = None
    with patch.dict(
        os.environ,
        {"OPERATOR_RUNTIME": "k8s", "DOCKER_SOCKET": "/dev/null"},
        clear=True,
    ):
        assert detect_runtime() == "kubernetes"  # 'k8s' alias normalized


# ─── (6) get_watch_selectors names-only wildcard matching ────────────────────


def test_get_watch_selectors_names_only_wildcards():
    """Names-only selectors parse wildcards and match via fnmatch."""
    with docker_env(OPERATOR_WATCH_NAMES="predagent-*"):
        selectors = get_watch_selectors()

    assert selectors["runtime"] == "docker"
    assert selectors["names"] == ["predagent-*"]
    assert selectors["labels"] == {}
    assert selectors["compose_projects"] == []

    # fnmatch wildcard semantics: predagent-<suffix> matches, bare
    # "predagent" and unrelated names do not.
    assert _matches_selectors({"Names": "predagent-api-1"}, selectors) is True
    assert _matches_selectors({"Names": "predagent-1"}, selectors) is True
    assert _matches_selectors({"Names": "predagent"}, selectors) is False
    assert _matches_selectors({"Names": "other-app-1"}, selectors) is False


def test_discover_matches_name_wildcard_prefix_slash():
    """Name matching tolerates the leading '/' some ps lines carry."""
    selectors = {"runtime": "docker", "labels": {}, "compose_projects": [], "names": ["predagent-*"]}
    assert _matches_selectors({"Names": "predagent-worker"}, selectors) is True
    assert _matches_selectors({"Names": "/predagent-worker"}, selectors) is True
