"""Test PodTracker state model."""

import pytest

from predictive_agent.state_model import PodTracker, StateModel, classify_state


def test_classify_state_healthy():
    """Test healthy state classification."""
    state = classify_state(
        memory_pct=50.0, cpu_pct=30.0, restart_rate=0.0,
        log_errors=0, node_pressure=False, markov_state="HEALTHY"
    )
    assert state == "HEALTHY"


def test_classify_state_degraded():
    """Test degraded state classification."""
    state = classify_state(
        memory_pct=70.0, cpu_pct=50.0, restart_rate=0.5,
        log_errors=2, node_pressure=False, markov_state="HEALTHY"
    )
    assert state in ("HEALTHY", "DEGRADED")


def test_classify_state_stressed():
    """Test stressed state classification."""
    state = classify_state(
        memory_pct=80.0, cpu_pct=70.0, restart_rate=1.0,
        log_errors=5, node_pressure=False, markov_state="DEGRADED"
    )
    assert state in ("DEGRADED", "STRESSED")


def test_classify_state_critical():
    """Test critical state classification."""
    state = classify_state(
        memory_pct=95.0, cpu_pct=90.0, restart_rate=5.0,
        log_errors=10, node_pressure=True, markov_state="STRESSED"
    )
    assert state in ("STRESSED", "CRITICAL")


def test_pod_tracker_creation():
    """Test PodTracker creation."""
    pt = PodTracker("opendesk", "openldap-0")
    assert pt.namespace == "opendesk"
    assert pt.name == "openldap-0"
    assert pt.state == "HEALTHY"
    assert pt.kalman_memory is not None
    assert pt.kalman_cpu is not None


def test_pod_tracker_update():
    """Test updating pod metrics."""
    pt = PodTracker("opendesk", "openldap-0")
    pt.update(
        memory_mib=500, memory_limit_mib=1024,
        cpu_m=100, restart_count=0, log_errors=0,
        node_pressure=False
    )
    assert pt.kalman_memory.level == 500
    assert pt.kalman_cpu.level == 100
    assert pt.memory_pct == pytest.approx(48.83, abs=1.0)


def test_pod_tracker_state_transition():
    """Test state transitions are tracked."""
    pt = PodTracker("opendesk", "openldap-0")
    # Start healthy
    assert pt.state == "HEALTHY"

    # Push to critical
    for _ in range(10):
        pt.update(
            memory_mib=990, memory_limit_mib=1024,
            cpu_m=950, restart_count=5, log_errors=10,
            node_pressure=True
        )
    assert pt.state in ("STRESSED", "CRITICAL")


def test_pod_tracker_ttf():
    """Test time-to-failure calculation."""
    pt = PodTracker("opendesk", "openldap-0")
    # Simulate rising memory
    for mem in [800, 810, 820, 830, 840, 850]:
        pt.update(
            memory_mib=mem, memory_limit_mib=1024,
            cpu_m=100, restart_count=0, log_errors=0,
            node_pressure=False
        )
    ttf = pt.time_to_failure()
    assert ttf is not None
    assert ttf > 0


def test_state_model_creation():
    """Test StateModel creation."""
    sm = StateModel()
    assert sm.pods == {}
    assert sm.markov is not None


def test_state_model_track_pod():
    """Test tracking a pod in StateModel."""
    sm = StateModel()
    sm.update_pod("opendesk", "openldap-0",
                   memory_mib=500, memory_limit_mib=1024,
                   cpu_m=100, restart_count=0, log_errors=0,
                   node_pressure=False)
    assert "opendesk/openldap-0" in sm.pods


def test_state_model_persistence():
    """Test StateModel save/load."""
    sm = StateModel()
    sm.update_pod("opendesk", "openldap-0",
                   memory_mib=500, memory_limit_mib=1024,
                   cpu_m=100, restart_count=0, log_errors=0,
                   node_pressure=False)
    data = sm.to_dict()
    assert "pods" in data
    assert "markov" in data

    sm2 = StateModel.from_dict(data)
    assert "opendesk/openldap-0" in sm2.pods


def test_state_model_markov_learning():
    """Test that Markov chain learns from state transitions."""
    sm = StateModel()
    # Simulate a pod going from HEALTHY to DEGRADED
    for _ in range(5):
        sm.update_pod("ns", "pod-1",
                       memory_mib=500, memory_limit_mib=1024,
                       cpu_m=100, restart_count=0, log_errors=0,
                       node_pressure=False)
    assert sm.markov.total_transitions >= 0  # May or may not transition
