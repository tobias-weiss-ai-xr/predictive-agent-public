"""Truth tables for every remediation action's should_execute guard.

Mutation testing (mutmut remediator campaign) left 250+ survivors in the
eligibility guards — the code that decides WHEN the agent is allowed to
restart/scale/cordon production objects. These tests pin each guard's
decision table: risk gate, protected-namespace veto, and the qualifying
signal per action.
"""

from types import SimpleNamespace

from predictive_agent.actions.container_restart import ContainerRestartAction
from predictive_agent.actions.node_cordon import NodeCordonAction
from predictive_agent.actions.pod_restart import PodRestartAction
from predictive_agent.actions.right_size import RightSizeAction
from predictive_agent.actions.rollout_restart import RolloutRestartAction
from predictive_agent.actions.scale import DeploymentScaleAction
from predictive_agent.actions.tune_resources import ResourceTunerAction


def _pod(ns="default", **kw):
    return SimpleNamespace(namespace=ns, **kw)


def test_pod_restart_truth_table():
    a = PodRestartAction()
    dict(risk_score=0.5)
    assert a.should_execute(_pod(phase="Failed"), None, 0.5) is False  # risk gate
    assert a.should_execute(_pod("kube-system", phase="Failed"), None, 0.9) is False  # protected
    assert a.should_execute(_pod(phase="Failed"), None, 0.9) is True
    assert a.should_execute(_pod(restart_count=5), None, 0.9) is True
    assert a.should_execute(_pod(phase="Running", restart_count=1), None, 0.9) is False


def test_container_restart_truth_table():
    a = ContainerRestartAction()
    assert a.should_execute(_pod(), None, 0.5) is False  # risk gate
    assert a.should_execute(_pod("kube-system", restart_count=99), None, 0.9) is False
    assert a.should_execute(_pod(restart_count=5), None, 0.9) is True
    assert a.should_execute(_pod(running=False), None, 0.9) is True
    assert a.should_execute(_pod(state="CrashLoopBackOff"), None, 0.9) is True
    assert a.should_execute(_pod(state="running", status="healthy"), None, 0.9) is False


def test_rollout_restart_truth_table():
    a = RolloutRestartAction()
    # needs risk + restarts AND 2+ failing pods in the deployment
    assert a.should_execute(_pod(restart_count=3, failing_pods_in_deployment=1), None, 0.9) is False
    assert a.should_execute(_pod(restart_count=3, failing_pods_in_deployment=2), None, 0.9) is True
    assert a.should_execute(_pod(restart_count=1, failing_pods_in_deployment=5), None, 0.9) is False
    assert a.should_execute(_pod("kube-system", restart_count=3, failing_pods_in_deployment=2), None, 0.9) is False


def test_deployment_scale_truth_table():
    a = DeploymentScaleAction()
    # needs sustained data (5+ cycles) AND cpu outside 20-80 band
    assert a.should_execute(_pod(cpu_pct=95, data_points=3), None, 0.9) is False
    assert a.should_execute(_pod(cpu_pct=50, data_points=10), None, 0.9) is False  # mid band
    assert a.should_execute(_pod(cpu_pct=95, data_points=10), None, 0.9) is True
    assert a.should_execute(_pod(cpu_pct=5, data_points=10), None, 0.9) is True
    assert a.should_execute(_pod("kube-system", cpu_pct=95, data_points=10), None, 0.9) is False


def test_resource_tuner_truth_table():
    a = ResourceTunerAction()
    assert a.should_execute(_pod(data_points=9, trend_confidence=0.99), None, 0.9) is False
    assert a.should_execute(_pod(data_points=10, trend_confidence=0.5), None, 0.9) is False
    assert a.should_execute(_pod(data_points=10, trend_confidence=0.9), None, 0.9) is True
    assert a.should_execute(_pod("kube-system", data_points=10, trend_confidence=0.9), None, 0.9) is False


def test_node_cordon_truth_table():
    a = NodeCordonAction()
    assert a.should_execute(_pod(node_pressure=True), None, 0.75) is False  # below 0.8
    assert a.should_execute(_pod(node_pressure=False), None, 0.9) is False
    assert a.should_execute(_pod(node_pressure=True), None, 0.9) is True


def test_right_size_below_advisory_threshold_is_refused():
    a = RightSizeAction()
    assert a.should_execute(_pod(), None, 0.4) is False


# ── command-construction pins: the command IS the action ─────────────────
# Mutation testing showed the kubectl/docker invocation builders survived
# every string mutant (e.g. dropping "-n namespace" or flipping a flag).
# These pin the exact wire format of the three destructive code paths.

def test_pod_restart_execute_builds_exact_kubectl_argv(monkeypatch):
    import predictive_agent.actions.pod_restart as mod
    calls = []
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: calls.append(a[0]) or SimpleNamespace(returncode=0, stdout="", stderr=""))
    a = PodRestartAction()
    ctx = SimpleNamespace(namespace="prod", dry_run=False)
    a.execute("web-0", ctx)
    assert calls[0] == ["kubectl", "delete", "pod", "web-0", "-n", "prod"]
    ctx.dry_run = True
    a.execute("web-0", ctx)
    assert calls[1] == ["kubectl", "delete", "pod", "web-0", "-n", "prod", "--dry-run=server"]


def test_resource_tuner_execute_builds_exact_json_patch(monkeypatch):
    import predictive_agent.actions.tune_resources as mod
    seen = {}
    monkeypatch.setattr(mod.subprocess, "run", lambda cmd, **k: seen.update(argv=cmd) or SimpleNamespace(returncode=0, stdout="", stderr=""))
    a = ResourceTunerAction()
    ctx = SimpleNamespace(namespace="prod", dry_run=False, cpu_request="250m",
                          mem_request="256Mi", cpu_limit="500m", mem_limit="512Mi")
    a.execute("web", ctx)
    import json as _json
    assert "--type=json" in seen["argv"]
    arg = next(a for a in seen["argv"] if a.startswith("-p="))
    patch = _json.loads(arg[3:])
    assert patch == [
        {"op": "replace", "path": "/spec/template/spec/containers/0/resources/requests/cpu", "value": "250m"},
        {"op": "replace", "path": "/spec/template/spec/containers/0/resources/requests/memory", "value": "256Mi"},
        {"op": "replace", "path": "/spec/template/spec/containers/0/resources/limits/cpu", "value": "500m"},
        {"op": "replace", "path": "/spec/template/spec/containers/0/resources/limits/memory", "value": "512Mi"},
    ]


def test_docker_socket_client_sends_method_path_host(monkeypatch):
    """Raw HTTP over AF_UNIX: request line must carry method+path, Host: unix."""
    import socket as _socket

    from predictive_agent.actions.container_restart import DockerSocketClient
    sent = {}

    payload = iter(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")

    class FakeSock:
        def connect(self, addr): sent["addr"] = addr
        def settimeout(self, t): pass
        def sendall(self, data): sent["raw"] = data.decode()
        def recv(self, n):
            b = next(payload, None)
            return b"" if b is None else bytes([b])
        def close(self): pass

    monkeypatch.setattr(_socket, "socket", lambda *a, **k: FakeSock())
    monkeypatch.setattr("predictive_agent.actions.container_restart.os.path.exists", lambda p: True)
    c = DockerSocketClient(socket_path="/fake/docker.sock")
    status, headers, body = c.request("POST", "/containers/x/restart?t=0")
    assert sent["addr"] == "/fake/docker.sock"
    assert sent["raw"].startswith("POST /containers/x/restart?t=0 HTTP/1.1\r\n")
    assert "Host: unix\r\n" in sent["raw"]
    assert status == 200 and body == "{}"
