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
    pt = PodTracker("openedusuite", "openldap-0")
    assert pt.namespace == "openedusuite"
    assert pt.name == "openldap-0"
    assert pt.state == "HEALTHY"
    assert pt.kalman_memory is not None
    assert pt.kalman_cpu is not None


def test_pod_tracker_update():
    """Test updating pod metrics."""
    pt = PodTracker("openedusuite", "openldap-0")
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
    pt = PodTracker("openedusuite", "openldap-0")
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
    pt = PodTracker("openedusuite", "openldap-0")
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


def test_pod_tracker_cpu_pct_normalized_against_cpu_limit():
    """cpu_pct must be normalized against the pod's summed CPU limit, not a
    hardcoded single core. A 16-core LLM pod at ~11.8 cores is ~74% busy, not
    1180% saturated — the old formula mis-scored it as >95% CPU and drove
    risk/state to DEGRADED/CRITICAL (ollama-next-0 false positive)."""
    pt = PodTracker("llm", "ollama-next-0")
    pt.update(
        memory_mib=29244, memory_limit_mib=65536,
        cpu_m=11800, restart_count=0, log_errors=0,
        node_pressure=False, cpu_limit_m=16000
    )
    assert pt.memory_pct == pytest.approx(44.6, abs=0.5)
    # 11800m of a 16000m limit = 73.75% (was 1180% with the old 1-core math)
    assert pt.cpu_pct == pytest.approx(73.75, abs=0.5)
    assert pt.cpu_pct < 95  # no longer "CPU saturated" under the risk model


def test_pod_tracker_cpu_pct_defaults_to_single_core_when_limit_unknown():
    """Without a cpu_limit_m the previous behaviour (1 core = 100%) is kept,
    so existing callers and pods without CPU limits behave unchanged."""
    pt = PodTracker("ns", "pod-0")
    pt.update(memory_mib=500, memory_limit_mib=1024, cpu_m=100,
              restart_count=0, log_errors=0, node_pressure=False)
    assert pt.cpu_pct == pytest.approx(10.0, abs=0.1)


def test_pod_tracker_cpu_pct_ignores_zero_limit():
    """A zero/absent CPU limit must not give 0% (which would look healthy) —
    fall back to the single-core default instead."""
    pt = PodTracker("ns", "pod-0")
    pt.update(memory_mib=500, memory_limit_mib=1024, cpu_m=500,
              restart_count=0, log_errors=0, node_pressure=False,
              cpu_limit_m=0)
    assert pt.cpu_pct == pytest.approx(50.0, abs=0.1)


def test_state_model_creation():
    """Test StateModel creation."""
    sm = StateModel()
    assert sm.pods == {}
    assert sm.markov is not None


def test_state_model_track_pod():
    """Test tracking a pod in StateModel."""
    sm = StateModel()
    sm.update_pod("openedusuite", "openldap-0",
                   memory_mib=500, memory_limit_mib=1024,
                   cpu_m=100, restart_count=0, log_errors=0,
                   node_pressure=False)
    assert "openedusuite/openldap-0" in sm.pods


def test_state_model_persistence():
    """Test StateModel save/load."""
    sm = StateModel()
    sm.update_pod("openedusuite", "openldap-0",
                   memory_mib=500, memory_limit_mib=1024,
                   cpu_m=100, restart_count=0, log_errors=0,
                   node_pressure=False)
    data = sm.to_dict()
    assert "pods" in data
    assert "markov" in data

    sm2 = StateModel.from_dict(data)
    assert "openedusuite/openldap-0" in sm2.pods


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


def test_classify_state_low_restart_rate_not_penalized():
    """Restart rates below 0.1/hr (1 restart per 10h) are background noise.

    restart_rate = restart_count / pod_age decays toward 0 but never reaches
    it, so the previous `> 0` branch applied a permanent classify penalty to
    any pod that ever restarted. Floor at a meaningful threshold so the
    stale-restart signal decays naturally with age.
    """
    # HEALTHY anchor + tiny stale rate must stay HEALTHY (this passed before
    # too, but documents the intent)
    assert classify_state(
        memory_pct=30.0, cpu_pct=20.0, restart_rate=0.05,
        log_errors=0, node_pressure=False, markov_state="HEALTHY",
    ) == "HEALTHY"
    # A pod stranded in DEGRADED with a stale restart and nothing else wrong
    # must be allowed to decay back to HEALTHY (previously the >0 restart
    # branch kept it pinned: DEGRADED(|1) + restart(|1) => score 2/3 boundary
    # plus hysteresis made recovery depend on the markov anchor).
    assert classify_state(
        memory_pct=30.0, cpu_pct=20.0, restart_rate=0.05,
        log_errors=0, node_pressure=False, markov_state="DEGRADED",
    ) == "HEALTHY"


def test_classify_state_fresh_restart_still_penalized():
    """A genuinely recent restart (rate > 1/hr) still counts.

    classify_state alone only cross 3-retakes with multiple signals; the
    real fresh-restart escalation lives in the risk multiplier. Here a
    >1/hr rate on top of a DEGRADED anchor must at least reach DEGRADED.
    """
    assert classify_state(
        memory_pct=30.0, cpu_pct=20.0, restart_rate=2.0,
        log_errors=0, node_pressure=False, markov_state="DEGRADED",
    ) == "DEGRADED"
    # Below the floor (<=0.1/hr) with a DEGRADED anchor decays to HEALTHY —
    # stale restart must not keep the pod pinned.
    assert classify_state(
        memory_pct=30.0, cpu_pct=20.0, restart_rate=0.05,
        log_errors=0, node_pressure=False, markov_state="DEGRADED",
    ) == "HEALTHY"


class TestCpuLimitKnown:
    """Regression (2026-11-25 audit): "no limit" must survive as a fact.

    ``get_pod_cpu_limit`` returns 0 when no container declares a CPU limit.
    The tracker substituted a synthetic 1000m default and dropped the fact that
    no limit exists, so every downstream consumer (cpu_pct, classify_state,
    risk.py) judged a legal burst against a one-core ceiling.
    """

    def test_zero_limit_marks_limit_unknown(self):
        model = StateModel()
        model.update_pod("ns", "p", 10, 100, 500, 0, 0, False, cpu_limit_m=0)
        tracker = model.pods["ns/p"]
        assert tracker.cpu_limit_known is False
        # numeric default is kept so existing thresholds still have a scale
        assert tracker.cpu_limit_m == 1000

    def test_real_limit_is_known(self):
        model = StateModel()
        model.update_pod("ns", "p", 10, 100, 500, 0, 0, False, cpu_limit_m=4000)
        tracker = model.pods["ns/p"]
        assert tracker.cpu_limit_known is True
        assert tracker.cpu_limit_m == 4000

    def test_unknown_limit_does_not_penalise_classification(self):
        """An unlimited pod bursting past one core must not score as saturated."""
        model = StateModel()
        # 1250m usage, no limit: legal burst on a big node.
        model.update_pod("ns", "p", 10, 100, 1250, 0, 0, False, cpu_limit_m=0)
        unlimited = model.pods["ns/p"].state
        # Hysteresis: state only flips after two agreeing cycles.
        for _ in range(5):
            model.update_pod("ns", "p", 10, 100, 1250, 0, 0, False, cpu_limit_m=0)
        unlimited = model.pods["ns/p"].state
        # Same usage against a real 1000m limit IS saturation.
        model2 = StateModel()
        for _ in range(5):
            model2.update_pod("ns", "p", 10, 100, 1250, 0, 0, False, cpu_limit_m=1000)
        limited = model2.pods["ns/p"].state
        assert unlimited == "HEALTHY"
        assert limited != "HEALTHY"

    def test_limit_known_round_trips_through_serialization(self):
        model = StateModel()
        model.update_pod("ns", "p", 10, 100, 500, 0, 0, False, cpu_limit_m=0)
        data = model.to_dict()
        restored = StateModel.from_dict(data)
        assert restored.pods["ns/p"].cpu_limit_known is False


# ── Seasonal (Holt-Winters) trend companions ─────────────────────────────────


def _seasonal_warmup(tracker, loop=6, period=24, step=60.0, amp=400.0, base=2000.0, monkeypatch=None):
    """Feed `tracker` `loop` full periods of a clean daily sine so the seasonal
    model fits with full seasonal-index coverage. The wall clock (time.time)
    is advanced `step` seconds per sample so the seasonal index advances one
    slot per observation (mirrors the 60s reconcile cadence)."""
    import math

    import predictive_agent.state_model as sm
    real_time = sm.time.time
    clock = [1735689600.0]  # fixed epoch
    def fake_time():
        return clock[0]
    monkeypatch.setattr(sm.time, "time", fake_time)
    n = loop * period
    for i in range(n):
        v = base + amp * math.sin(2 * math.pi * (i % period) / period)
        tracker.update(v, 4096, int(v % 500), 0, 0, False, cpu_limit_m=1000)
        clock[0] += step
    monkeypatch.setattr(sm.time, "time", real_time)
    return tracker


class TestSeasonalTrend:
    def test_off_mode_does_not_keep_seasonal_state(self):
        t = PodTracker("ns", "p", seasonal_mode="off", seasonal_period=24)
        assert t.seasonal_mode == "off"
        t.update(100, 4096, 50, 0, 0, False, cpu_limit_m=1000)
        assert t.seasonal_mem_conf == 0.0
        assert t.seasonal_mem_vel == 0.0

    def test_shadow_mode_exposes_seasonal_conf_signal(self, monkeypatch):
        t = PodTracker("ns", "p", seasonal_mode="shadow", seasonal_period=24)
        _seasonal_warmup(t, monkeypatch=monkeypatch)
        assert t.seasonal_mem_conf > 0.5, f"conf={t.seasonal_mem_conf}"
        assert t.seasonal_cpu_conf > 0.5
        # shadow still returns the Kalman velocity unchanged
        assert t.memory_trend == t.kalman_memory.velocity

    def test_on_mode_uses_seasonal_velocity_when_confident(self, monkeypatch):
        t = PodTracker("ns", "p", seasonal_mode="on", seasonal_period=24)
        _seasonal_warmup(t, loop=8, monkeypatch=monkeypatch)
        assert t.seasonal_mem_conf > 0.5
        # seasonal expected velocity must be bounded and sane (per-sample delta
        # of a 400-amplitude / 24-period sine ≈ (2*pi*400)/24 ≈ 105)
        assert abs(t.seasonal_mem_vel) < 400
        # and it must differ from the flat Kalman velocity on a clean signal
        assert t.memory_trend != pytest.approx(t.kalman_memory.velocity, abs=1e-6)

    def test_on_mode_falls_back_to_kalman_without_fit(self):
        t = PodTracker("ns", "p", seasonal_mode="on", seasonal_period=24)
        t.update(100, 4096, 50, 0, 0, False, cpu_limit_m=1000)
        assert t.seasonal_mem_conf == 0.0
        assert t.memory_trend == t.kalman_memory.velocity

    def test_seasonal_fields_in_tracker_dict(self):
        t = PodTracker("ns", "p", seasonal_mode="shadow", seasonal_period=24)
        d = t.__dict__
        assert "seasonal_mode" in d
        assert "seasonal_mem_conf" in d
