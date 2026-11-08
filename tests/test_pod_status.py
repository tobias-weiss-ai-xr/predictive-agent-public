"""Test pod status signal collection from collector.get_pod_status_signals and Docker container signals."""
from predictive_agent.collector import get_container_status_signals, get_pod_status_signals


def _make_pod(phase="Running", container_ready=True, wait_state=None,
               terminated=False, terminated_reason=None, restart_count=0,
               scheduled=True, initialized=True):
    """Build a minimal pod JSON object for testing."""
    container_status = {
        "name": "main",
        "ready": container_ready,
        "restartCount": restart_count,
    }
    if wait_state:
        container_status["state"] = {"waiting": {"reason": wait_state}}
    elif terminated:
        container_status["state"] = {"terminated": {"reason": terminated_reason or "Error"}}
    else:
        container_status["state"] = {"running": {}}

    if terminated_reason and not terminated:
        container_status["lastState"] = {"terminated": {"reason": terminated_reason}}

    conditions = []
    if initialized:
        conditions.append({"type": "Initialized", "status": "True"})
    else:
        conditions.append({"type": "Initialized", "status": "False"})
    if scheduled:
        conditions.append({"type": "PodScheduled", "status": "True"})
    else:
        conditions.append({"type": "PodScheduled", "status": "False"})

    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": "test-pod", "namespace": "default"},
        "spec": {"nodeName": "node-1"},
        "status": {
            "phase": phase,
            "containerStatuses": [container_status],
            "conditions": conditions,
        },
    }


class TestGetPodStatusSignals:
    """Test get_pod_status_signals function."""

    def test_running_healthy_pod(self):
        """A healthy Running pod with ready container should return low-risk signals."""
        pod = _make_pod(phase="Running", container_ready=True)
        signals = get_pod_status_signals(pod)
        assert signals["pod_phase"] == "Running"
        assert signals["container_ready"] is True
        assert signals["wait_state"] is None
        assert signals["terminated"] is False
        assert signals["terminated_reason"] is None
        assert signals["restart_count"] == 0
        assert signals["pod_scheduled"] is True
        assert signals["pod_initialized"] is True

    def test_crash_loop_back_off(self):
        """CrashLoopBackOff should be captured as wait_state."""
        pod = _make_pod(phase="Running", container_ready=False, wait_state="CrashLoopBackOff")
        signals = get_pod_status_signals(pod)
        assert signals["wait_state"] == "CrashLoopBackOff"
        assert signals["container_ready"] is False

    def test_create_container_config_error(self):
        """CreateContainerConfigError should be captured."""
        pod = _make_pod(phase="Pending", container_ready=False, wait_state="CreateContainerConfigError")
        signals = get_pod_status_signals(pod)
        assert signals["wait_state"] == "CreateContainerConfigError"
        assert signals["pod_phase"] == "Pending"

    def test_image_pull_back_off(self):
        """ImagePullBackOff should be captured."""
        pod = _make_pod(phase="Pending", container_ready=False, wait_state="ImagePullBackOff")
        signals = get_pod_status_signals(pod)
        assert signals["wait_state"] == "ImagePullBackOff"

    def test_oom_killed(self):
        """OOMKilled should be captured as terminated with reason."""
        pod = _make_pod(phase="Running", terminated=True, terminated_reason="OOMKilled")
        signals = get_pod_status_signals(pod)
        assert signals["terminated"] is True
        assert signals["terminated_reason"] == "OOMKilled"

    def test_terminated_error(self):
        """Terminated with Error reason should be captured."""
        pod = _make_pod(phase="Running", terminated=True, terminated_reason="Error")
        signals = get_pod_status_signals(pod)
        assert signals["terminated"] is True
        assert signals["terminated_reason"] == "Error"

    def test_last_state_terminated(self):
        """Last terminated state should be captured even if currently running."""
        pod = _make_pod(phase="Running", container_ready=True, terminated_reason="OOMKilled")
        signals = get_pod_status_signals(pod)
        # terminated_reason should be set from lastState
        assert signals["terminated_reason"] == "OOMKilled"

    def test_restart_count(self):
        """Restart count should be summed across containers."""
        pod = _make_pod(restart_count=5)
        signals = get_pod_status_signals(pod)
        assert signals["restart_count"] == 5

    def test_restart_count_multiple_containers(self):
        """Restart count should sum across multiple containers."""
        pod = _make_pod(restart_count=3)
        # Add a second container
        pod["status"]["containerStatuses"].append({
            "name": "sidecar",
            "ready": True,
            "restartCount": 7,
            "state": {"running": {}},
        })
        signals = get_pod_status_signals(pod)
        assert signals["restart_count"] == 10

    def test_pending_phase(self):
        """Pending phase should be captured."""
        pod = _make_pod(phase="Pending", container_ready=False, scheduled=False)
        signals = get_pod_status_signals(pod)
        assert signals["pod_phase"] == "Pending"
        assert signals["pod_scheduled"] is False

    def test_failed_phase(self):
        """Failed phase should be captured."""
        pod = _make_pod(phase="Failed", container_ready=False, terminated=True, terminated_reason="Error")
        signals = get_pod_status_signals(pod)
        assert signals["pod_phase"] == "Failed"
        assert signals["terminated"] is True

    def test_succeeded_phase(self):
        """Succeeded phase should be captured."""
        pod = _make_pod(phase="Succeeded", container_ready=False, terminated=True, terminated_reason="Completed")
        signals = get_pod_status_signals(pod)
        assert signals["pod_phase"] == "Succeeded"
        assert signals["terminated"] is True
        assert signals["terminated_reason"] == "Completed"

    def test_not_scheduled(self):
        """Unscheduled pod should have pod_scheduled=False."""
        pod = _make_pod(phase="Pending", scheduled=False)
        signals = get_pod_status_signals(pod)
        assert signals["pod_scheduled"] is False

    def test_unknown_phase(self):
        """Unknown phase should be captured."""
        pod = _make_pod(phase="Unknown")
        signals = get_pod_status_signals(pod)
        assert signals["pod_phase"] == "Unknown"

    def test_empty_pod(self):
        """Empty pod JSON should return safe defaults."""
        pod = {"status": {}}
        signals = get_pod_status_signals(pod)
        assert signals["pod_phase"] == "Unknown"
        assert signals["container_ready"] is True
        assert signals["wait_state"] is None
        assert signals["terminated"] is False
        assert signals["restart_count"] == 0


# =============================================================================
# Docker container status signals tests
# =============================================================================


def _make_container(running=True, restarting=False, dead=False, oom_killed=False,
                     restart_count=0, exit_code=None, health_status=None, paused=False):
    """Build a minimal Docker inspect JSON object for testing."""
    state = {
        "Running": running,
        "Paused": paused,
        "Restarting": restarting,
        "Dead": dead,
        "OOMKilled": oom_killed,
        "RestartCount": restart_count,
    }
    if exit_code is not None:
        state["ExitCode"] = exit_code

    if health_status:
        state["Health"] = {"Status": health_status}

    return {
        "Id": "abc123",
        "Name": "/test-container",
        "State": state,
        "Config": {"Image": "test:latest"},
    }


class TestGetContainerStatusSignals:
    """Test get_container_status_signals function for Docker containers."""

    def test_running_healthy_container(self):
        """A running healthy container should return ready=True, no issues."""
        container = _make_container(running=True, health_status="healthy")
        signals = get_container_status_signals(container)
        assert signals["wait_state"] is None
        assert signals["terminated"] is False
        assert signals["restart_count"] == 0
        assert signals["ready"] is True

    def test_running_no_health_container(self):
        """A running container without health checks should be ready."""
        container = _make_container(running=True, health_status=None)
        signals = get_container_status_signals(container)
        assert signals["ready"] is True
        assert signals["terminated"] is False

    def test_running_unhealthy_container(self):
        """A running but unhealthy container should not be ready."""
        container = _make_container(running=True, health_status="unhealthy")
        signals = get_container_status_signals(container)
        assert signals["ready"] is False
        assert signals["wait_state"] is None
        assert signals["terminated"] is False

    def test_restarting_container(self):
        """A restarting container should have wait_state=Restarting."""
        container = _make_container(running=False, restarting=True, restart_count=5)
        signals = get_container_status_signals(container)
        assert signals["wait_state"] == "Restarting"
        assert signals["terminated"] is True
        assert signals["restart_count"] == 5
        assert signals["ready"] is False

    def test_restart_loop_scenario(self):
        """Container in restart loop should show Restarting wait_state and incremented restart_count."""
        container = _make_container(running=False, restarting=True, restart_count=10)
        signals = get_container_status_signals(container)
        assert signals["wait_state"] == "Restarting"
        assert signals["restart_count"] == 10
        assert signals["terminated"] is True
        assert signals["ready"] is False

    def test_dead_container(self):
        """A dead container should have wait_state=Dead and terminated=True."""
        container = _make_container(running=False, dead=True, restart_count=0)
        signals = get_container_status_signals(container)
        assert signals["wait_state"] == "Dead"
        assert signals["terminated"] is True
        assert signals["ready"] is False

    def test_oom_killed_container(self):
        """An OOM-killed container should have wait_state=OOMKilled and terminated=True."""
        container = _make_container(running=False, oom_killed=True, restart_count=0)
        signals = get_container_status_signals(container)
        assert signals["wait_state"] == "OOMKilled"
        assert signals["terminated"] is True
        assert signals["ready"] is False

    def test_exited_container(self):
        """An exited container should have wait_state=Exited and terminated=True."""
        container = _make_container(running=False, exit_code=0, restart_count=0)
        signals = get_container_status_signals(container)
        assert signals["wait_state"] == "Exited"
        assert signals["terminated"] is True
        assert signals["ready"] is False

    def test_exited_with_error_code(self):
        """Container exited with non-zero code should show exit code in wait_state."""
        container = _make_container(running=False, exit_code=137, restart_count=0)
        signals = get_container_status_signals(container)
        assert signals["wait_state"] == "Exited (137)"
        assert signals["terminated"] is True
        assert signals["ready"] is False

    def test_paused_container(self):
        """A paused container is not ready but not terminated."""
        container = _make_container(running=True, paused=True, restart_count=0)
        signals = get_container_status_signals(container)
        assert signals["wait_state"] is None
        assert signals["terminated"] is False
        assert signals["ready"] is False

    def test_restart_count(self):
        """Restart count should be extracted from container state."""
        container = _make_container(running=True, restart_count=7)
        signals = get_container_status_signals(container)
        assert signals["restart_count"] == 7

    def test_empty_container(self):
        """Empty container JSON should return safe defaults."""
        container = {"State": {}}
        signals = get_container_status_signals(container)
        assert signals["wait_state"] is None
        assert signals["terminated"] is True  # Not running
        assert signals["restart_count"] == 0
        assert signals["ready"] is False

    def test_starting_container(self):
        """A container in Created/starting state (not yet running) should not be ready."""
        container = _make_container(running=False, restarting=False, dead=False,
                                     restart_count=0, exit_code=None)
        signals = get_container_status_signals(container)
        assert signals["wait_state"] is None
        assert signals["terminated"] is True
        assert signals["ready"] is False
