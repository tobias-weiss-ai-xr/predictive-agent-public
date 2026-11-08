"""Tests for per-metric Kalman parameters and state model anomaly detection."""
from predictive_agent.kalman import KalmanTrend
from predictive_agent.state_model import PodTracker, StateModel


class TestPerMetricKalmanParams:
    """Test that different metrics use different Kalman parameters."""

    def test_memory_kalman_default_params(self):
        """Memory Kalman filter should use lower process noise by default."""
        pt = PodTracker("ns", "pod-1")
        # Memory: process_noise=0.5, measurement_noise=50.0
        assert pt.kalman_memory.Q[0][0] == 0.5  # process_noise * dt (dt=1)
        assert pt.kalman_memory.R == 50.0

    def test_cpu_kalman_default_params(self):
        """CPU Kalman filter should use higher process noise by default."""
        pt = PodTracker("ns", "pod-1")
        # CPU: process_noise=5.0, measurement_noise=200.0
        assert pt.kalman_cpu.Q[0][0] == 5.0  # process_noise * dt (dt=1)
        assert pt.kalman_cpu.R == 200.0

    def test_memory_kalman_smoother_than_cpu(self):
        """Memory filter should be smoother (level closer to mean) than CPU filter."""
        pt = PodTracker("ns", "pod-1")
        # Feed same noisy data to both
        values = [100, 90, 110, 95, 105, 98, 102, 92, 108, 97]
        for v in values:
            pt.kalman_memory.update(float(v))
            pt.kalman_cpu.update(float(v))
        # Memory (smoother, lower process noise) should have level closer to mean (~100)
        # CPU (more reactive, higher process noise) should track noise more closely
        mean_val = sum(values) / len(values)  # ~99.7
        mem_error = abs(pt.kalman_memory.level - mean_val)
        cpu_error = abs(pt.kalman_cpu.level - mean_val)
        # Memory should be at least as close to the mean as CPU
        assert mem_error <= cpu_error + 2.0  # Allow small tolerance

    def test_custom_kalman_params(self):
        """Test that custom Kalman parameters can be passed to PodTracker."""
        custom_mem = KalmanTrend(process_noise=0.1, measurement_noise=10.0)
        custom_cpu = KalmanTrend(process_noise=10.0, measurement_noise=500.0)
        pt = PodTracker("ns", "pod-1", kalman_memory=custom_mem, kalman_cpu=custom_cpu)
        assert pt.kalman_memory.Q[0][0] == 0.1
        assert pt.kalman_memory.R == 10.0
        assert pt.kalman_cpu.Q[0][0] == 10.0
        assert pt.kalman_cpu.R == 500.0

    def test_env_var_override(self, monkeypatch):
        """PodTracker defaults follow the module-level _KALMAN_* constants
        (which config wires from KALMAN_* env vars at import).

        Patch the constants directly instead of reload()-ing with a polluted
        environ: the original version reloaded while the env vars were still
        set (monkeypatch teardown runs after the test), permanently leaking
        the override into the module for every later test.
        """
        import predictive_agent.state_model as sm
        monkeypatch.setattr(sm, "_KALMAN_MEM_PROCESS_NOISE", 2.0)
        monkeypatch.setattr(sm, "_KALMAN_MEM_MEASUREMENT_NOISE", 75.0)
        monkeypatch.setattr(sm, "_KALMAN_CPU_PROCESS_NOISE", 15.0)
        monkeypatch.setattr(sm, "_KALMAN_CPU_MEASUREMENT_NOISE", 300.0)
        pt = sm.PodTracker("ns", "pod-1")
        assert pt.kalman_memory.Q[0][0] == 2.0
        assert pt.kalman_memory.R == 75.0
        assert pt.kalman_cpu.Q[0][0] == 15.0
        assert pt.kalman_cpu.R == 300.0


class TestStateModelAnomalyDetection:
    """Test anomaly detection in the state model."""

    def test_pod_tracker_has_anomaly_scores(self):
        """PodTracker should have memory_anomaly_score and cpu_anomaly_score."""
        pt = PodTracker("ns", "pod-1")
        assert hasattr(pt, "memory_anomaly_score")
        assert hasattr(pt, "cpu_anomaly_score")
        assert pt.memory_anomaly_score == 0.0
        assert pt.cpu_anomaly_score == 0.0

    def test_anomaly_score_updates_on_spike(self):
        """Anomaly score should increase on a sudden spike."""
        pt = PodTracker("ns", "pod-1")
        # Feed stable data
        for v in [100, 100, 100, 100]:
            pt.update(memory_mib=100, memory_limit_mib=1024, cpu_m=50,
                      restart_count=0, log_errors=0, node_pressure=False)
        assert pt.memory_anomaly_score < 1.0  # Low anomaly on stable data
        # Sudden spike
        pt.update(memory_mib=500, memory_limit_mib=1024, cpu_m=50,
                  restart_count=0, log_errors=0, node_pressure=False)
        assert pt.memory_anomaly_score > 1.0  # High anomaly on spike

    def test_anomaly_score_stays_low_on_gradual_increase(self):
        """Anomaly score should stay low on gradual, predictable increases."""
        pt = PodTracker("ns", "pod-1")
        for i in range(20):
            pt.update(memory_mib=100 + i * 5, memory_limit_mib=1024, cpu_m=50,
                      restart_count=0, log_errors=0, node_pressure=False)
        # Gradual increase should not trigger high anomaly
        assert pt.memory_anomaly_score < 3.0

    def test_state_model_persistence_includes_anomaly_scores(self):
        """State model serialization should include anomaly scores."""
        sm = StateModel()
        sm.update_pod("ns", "pod-1",
                       memory_mib=500, memory_limit_mib=1024,
                       cpu_m=100, restart_count=0, log_errors=0,
                       node_pressure=False)
        data = sm.to_dict()
        pod_data = data["pods"]["ns/pod-1"]
        assert "memory_anomaly_score" in pod_data
        assert "cpu_anomaly_score" in pod_data

    def test_state_model_load_preserves_anomaly_scores(self):
        """State model deserialization should restore anomaly scores."""
        sm = StateModel()
        sm.update_pod("ns", "pod-1",
                       memory_mib=500, memory_limit_mib=1024,
                       cpu_m=100, restart_count=0, log_errors=0,
                       node_pressure=False)
        # Manually set anomaly scores
        sm.pods["ns/pod-1"].memory_anomaly_score = 3.5
        sm.pods["ns/pod-1"].cpu_anomaly_score = 2.1
        data = sm.to_dict()
        sm2 = StateModel.from_dict(data)
        assert sm2.pods["ns/pod-1"].memory_anomaly_score == 3.5
        assert sm2.pods["ns/pod-1"].cpu_anomaly_score == 2.1
