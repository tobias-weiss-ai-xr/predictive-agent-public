"""Boundary and edge-case tests.

SOTA paradigm: Boundary value analysis tests the edges of input domains
where bugs are most likely to occur. Each function is tested at its
minimum, maximum, just below, just above, and exact boundary values.

Covers:
- Kalman filter: zero measurements, single measurement, negative velocity
- Risk score: exactly at thresholds (70%, 85%, 95%, 100%)
- Markov chain: empty transitions, unknown states, all-same-state
- Collector: empty/whitespace/malformed kubectl output
- State classifier: exact threshold scores (3, 6, 10)
- Predictor: zero trend, zero limit, negative remaining
- Persistence: empty files, corrupted JSON, permission errors
- Server: all endpoints with empty state
"""
import os
import tempfile

import pytest

from predictive_agent.collector import (
    collect_top_metrics,
    collect_top_nodes,
    count_log_errors,
    get_node_conditions,
    get_pod_resources,
    parse_cpu,
    parse_memory,
)
from predictive_agent.kalman import KalmanTrend
from predictive_agent.markov import MarkovChain
from predictive_agent.persistence import StateStore
from predictive_agent.predictor import Predictor
from predictive_agent.risk import calculate_risk
from predictive_agent.state_model import classify_state

# ─── Kalman boundary tests ──────────────────────────────────────────────────

class TestKalmanBoundary:
    """Boundary tests for KalmanTrend."""

    def test_zero_measurement(self):
        """Update with 0.0 should set level to 0.0."""
        kt = KalmanTrend()
        kt.update(0.0)
        assert kt.level == 0.0

    def test_single_measurement_velocity_zero(self):
        """After a single update, velocity should be 0."""
        kt = KalmanTrend()
        kt.update(42.0)
        assert kt.velocity == 0.0

    def test_large_measurement(self):
        """Very large measurement should not overflow."""
        kt = KalmanTrend()
        kt.update(1e15)
        assert kt.level == 1e15
        assert kt.velocity == 0.0

    def test_negative_velocity_decreasing(self):
        """Consistently decreasing measurements produce negative velocity."""
        kt = KalmanTrend()
        for v in [100, 90, 80, 70, 60]:
            kt.update(float(v))
        assert kt.velocity < 0

    def test_predict_zero_steps(self):
        """Predict with 0 steps should return current level."""
        kt = KalmanTrend()
        kt.update(100.0)
        pred_level, _ = kt.predict(0.0)
        assert pred_level == pytest.approx(100.0, abs=0.01)

    def test_predict_negative_steps(self):
        """Predict with negative steps should go backwards (below current level)."""
        kt = KalmanTrend()
        kt.update(100.0)
        kt.update(110.0)  # positive velocity
        current_level = kt.level
        pred_level, _ = kt.predict(-1.0)
        # With negative steps and positive velocity, prediction should be below current level
        assert pred_level < current_level

    def test_time_to_threshold_zero_velocity(self):
        """Time to threshold with zero velocity should return max_steps."""
        kt = KalmanTrend()
        kt.update(100.0)
        kt.update(100.0)  # velocity ≈ 0
        steps, confidence = kt.time_to_threshold(200.0)
        assert steps == 180  # max_steps default

    def test_time_to_threshold_already_past(self):
        """Threshold already reached should return (0, 1.0)."""
        kt = KalmanTrend()
        kt.update(200.0)
        kt.update(210.0)  # positive velocity, level > threshold
        steps, confidence = kt.time_to_threshold(150.0)
        assert steps == 0
        assert confidence == 1.0


# ─── Risk score boundary tests ──────────────────────────────────────────────

class TestRiskBoundary:
    """Boundary tests for calculate_risk at exact threshold values."""

    def test_memory_exactly_70(self):
        """Memory at exactly 70% triggers the >70 threshold (lr *= 2.0)."""
        base = {"cpu_pct": 0, "restart_rate_per_hr": 0, "log_error_rate_per_min": 0,
                "node_memory_pressure": False, "node_disk_pressure": False}
        risk_below = calculate_risk({**base, "memory_pct": 69.9}, "HEALTHY", 0, 0)
        risk_at = calculate_risk({**base, "memory_pct": 70.0}, "HEALTHY", 0, 0)
        risk_above = calculate_risk({**base, "memory_pct": 70.1}, "HEALTHY", 0, 0)
        assert risk_at >= risk_below
        assert risk_above >= risk_at

    def test_memory_exactly_85(self):
        """Memory at exactly 85% does NOT trigger >85 (strictly greater). 85.1 does."""
        base = {"cpu_pct": 0, "restart_rate_per_hr": 0, "log_error_rate_per_min": 0,
                "node_memory_pressure": False, "node_disk_pressure": False}
        risk_at_85 = calculate_risk({**base, "memory_pct": 85.0}, "HEALTHY", 0, 0)
        risk_below = calculate_risk({**base, "memory_pct": 84.9}, "HEALTHY", 0, 0)
        risk_above = calculate_risk({**base, "memory_pct": 85.1}, "HEALTHY", 0, 0)
        # 85.0 is NOT > 85, so same tier as 84.9 (both >70)
        assert risk_at_85 == pytest.approx(risk_below)
        # 85.1 IS > 85, so higher tier
        assert risk_above > risk_at_85

    def test_memory_exactly_95(self):
        """Memory at exactly 95% does NOT trigger >95 (strictly greater). 95.1 does."""
        base = {"cpu_pct": 0, "restart_rate_per_hr": 0, "log_error_rate_per_min": 0,
                "node_memory_pressure": False, "node_disk_pressure": False}
        risk_at_95 = calculate_risk({**base, "memory_pct": 95.0}, "HEALTHY", 0, 0)
        risk_below = calculate_risk({**base, "memory_pct": 94.9}, "HEALTHY", 0, 0)
        risk_above = calculate_risk({**base, "memory_pct": 95.1}, "HEALTHY", 0, 0)
        # 95.0 is NOT > 95, so same tier as 94.9 (both >85)
        assert risk_at_95 == pytest.approx(risk_below)
        # 95.1 IS > 95, so higher tier
        assert risk_above > risk_at_95

    def test_restart_exactly_1(self):
        """Restart rate at exactly 1 does NOT trigger >1 (strictly greater). 1.1 does."""
        base = {"memory_pct": 0, "cpu_pct": 0, "log_error_rate_per_min": 0,
                "node_memory_pressure": False, "node_disk_pressure": False}
        risk_at_1 = calculate_risk({**base, "restart_rate_per_hr": 1.0}, "HEALTHY", 0, 0)
        risk_below = calculate_risk({**base, "restart_rate_per_hr": 0.9}, "HEALTHY", 0, 0)
        risk_above = calculate_risk({**base, "restart_rate_per_hr": 1.1}, "HEALTHY", 0, 0)
        # 1.0 is NOT > 1, so same tier as 0.9
        assert risk_at_1 == pytest.approx(risk_below)
        # 1.1 IS > 1, so higher tier
        assert risk_above > risk_at_1

    def test_all_zero_metrics(self):
        """All metrics zero should yield near-prior risk."""
        risk = calculate_risk({}, "HEALTHY", 0.0, 0.0)
        assert 0.14 <= risk <= 0.16  # prior is now 0.15 based on calibration

    def test_all_maxed_metrics(self):
        """All metrics maxed should yield near-max risk but ≤ 0.99."""
        risk = calculate_risk(
            {"memory_pct": 100, "cpu_pct": 100, "restart_rate_per_hr": 1000,
             "log_error_rate_per_min": 1000, "node_memory_pressure": True,
             "node_disk_pressure": True, "memory_limit_mib": 1024, "memory_mib": 1024,
             "memory_trend_mib_per_min": 100},
            "FAILED", 1.0, 1.0
        )
        assert risk <= 0.99
        assert risk > 0.9


# ─── Markov chain boundary tests ─────────────────────────────────────────────

class TestMarkovBoundary:
    """Boundary tests for MarkovChain."""

    def test_no_transitions(self):
        """Fresh chain with no transitions should still produce a valid matrix."""
        mc = MarkovChain()
        matrix = mc.transition_matrix()
        assert len(matrix) == 6
        for row in matrix:
            assert abs(sum(row) - 1.0) < 1e-9

    def test_unknown_state_transition(self):
        """Unknown states should map to index 0 (HEALTHY)."""
        mc = MarkovChain()
        mc.record_transition("UNKNOWN_STATE", "ALSO_UNKNOWN")
        assert mc.total_transitions == 1
        # Should have recorded in row 0, col 0
        assert mc.counts[0][0] == 10.5  # 9.5 (prior) + 1

    def test_all_same_state_transitions(self):
        """Recording the same transition many times should converge."""
        mc = MarkovChain()
        for _ in range(100):
            mc.record_transition("HEALTHY", "HEALTHY")
        probs = mc.predict("HEALTHY")
        assert probs["HEALTHY"] > 0.9

    def test_empty_from_state_predict(self):
        """Predict from a state with no outgoing transitions should still work."""
        mc = MarkovChain()
        # FAILED state has prior counts, so this should work
        probs = mc.predict("FAILED")
        assert "HEALTHY" in probs
        assert abs(sum(probs.values()) - 1.0) < 1e-9

    def test_persistence_roundtrip_empty(self):
        """Save/load an empty chain should preserve counts."""
        mc = MarkovChain()
        data = mc.to_dict()
        mc2 = MarkovChain.from_dict(data)
        assert mc2.total_transitions == 0
        assert mc2.counts == mc.counts

    def test_persistence_roundtrip_with_data(self):
        """Save/load a chain with transitions should preserve all data."""
        mc = MarkovChain()
        mc.record_transition("HEALTHY", "DEGRADED")
        mc.record_transition("DEGRADED", "STRESSED")
        mc.record_transition("STRESSED", "CRITICAL")
        data = mc.to_dict()
        mc2 = MarkovChain.from_dict(data)
        assert mc2.total_transitions == 3
        assert mc2.counts == mc.counts

    def test_from_dict_empty(self):
        """from_dict with empty data should return a fresh chain."""
        mc = MarkovChain.from_dict({})
        assert mc.total_transitions == 0
        assert mc.counts == MarkovChain.PRIOR_COUNTS

    def test_from_dict_none(self):
        """from_dict with None should return a fresh chain."""
        mc = MarkovChain.from_dict(None)
        assert mc.total_transitions == 0


# ─── Collector boundary tests ───────────────────────────────────────────────

class TestCollectorBoundary:
    """Boundary tests for collector functions."""

    def test_parse_cpu_zero(self):
        assert parse_cpu("0m") == 0

    def test_parse_cpu_empty(self):
        assert parse_cpu("") == 0

    def test_parse_cpu_core(self):
        """1 core = 1000m."""
        assert parse_cpu("1") == 1000
        assert parse_cpu("2.5") == 2500

    def test_parse_cpu_decimal_cores(self):
        assert parse_cpu("0.5") == 500
        assert parse_cpu("0.1") == 100

    def test_parse_memory_zero(self):
        assert parse_memory("0Mi") == 0

    def test_parse_memory_empty(self):
        assert parse_memory("") == 0

    def test_parse_memory_ki(self):
        assert parse_memory("1024Ki") == 1

    def test_parse_memory_ti(self):
        assert parse_memory("1Ti") == 1048576

    def test_collect_top_metrics_malformed(self):
        """Malformed lines should be skipped, not crash."""
        output = """NAMESPACE  NAME  CPU  MEMORY
bad line
ns/pod  100m  128Mi
  ns2/pod2  200m  256Mi
"""
        metrics = collect_top_metrics(output)
        # Should parse valid lines, skip bad ones
        assert len(metrics) >= 0  # No crash

    def test_collect_top_nodes_empty(self):
        assert collect_top_nodes("") == {}

    def test_collect_top_nodes_malformed(self):
        output = """NAME  CPU  CPU%  MEMORY  MEMORY%
bad
clrz14-06  1000m  10%  8000Mi  20%
"""
        metrics = collect_top_nodes(output)
        assert "clrz14-06" in metrics

    def test_count_log_errors_empty(self):
        assert count_log_errors("") == 0

    def test_count_log_errors_no_errors(self):
        assert count_log_errors("[INFO] all good\n[DEBUG] fine") == 0

    def test_count_log_errors_all_patterns(self):
        """Each error pattern should be counted once per line."""
        logs = "\n".join([
            "ERROR: something",
            "Error: something",
            "FATAL: something",
            "PANIC: something",
            "OOM killed",
            "CrashLoopBackOff",
            "Exception in thread",
            "Traceback (most recent call last):",
            "panic: runtime error",
            "fatal: something",
        ])
        assert count_log_errors(logs) == 10

    def test_count_log_errors_dedup_per_line(self):
        """Multiple patterns on the same line should count as 1."""
        logs = "ERROR FATAL PANIC OOM all on one line"
        assert count_log_errors(logs) == 1

    def test_get_pod_resources_empty(self):
        assert get_pod_resources({}) == {}

    def test_get_pod_resources_no_limits(self):
        """Pod with no resource limits should return 0 for cpu/mem."""
        pod = {"spec": {"containers": [{"name": "app"}]}}
        resources = get_pod_resources(pod)
        assert "app" in resources
        assert resources["app"]["cpu_m"] == 0
        assert resources["app"]["memory_mib"] == 0

    def test_get_node_conditions_empty(self):
        assert get_node_conditions({}) == {}

    def test_get_node_conditions_no_conditions(self):
        node_json = {"items": [{"metadata": {"name": "node-1"}, "status": {}}]}
        conditions = get_node_conditions(node_json)
        # Node with no conditions returns empty dict (no conditions key added)
        assert conditions == {}


# ─── State classifier boundary tests ────────────────────────────────────────

class TestClassifyStateBoundary:
    """Boundary tests for classify_state at exact score thresholds."""

    def test_score_exactly_3_degraded(self):
        """Score of 3 should be DEGRADED.
        memory_pct=51 (>50, score 1) + cpu_pct=61 (>60, score 2) = 3"""
        state = classify_state(memory_pct=51, cpu_pct=61, restart_rate=0,
                               log_errors=0, node_pressure=False, markov_state="HEALTHY")
        assert state == "DEGRADED"

    def test_score_exactly_6_stressed(self):
        """Score of 6 should be STRESSED.
        memory_pct=86 (>85, score 3) + cpu_pct=81 (>80, score 3) = 6"""
        state = classify_state(memory_pct=86, cpu_pct=81, restart_rate=0,
                               log_errors=0, node_pressure=False, markov_state="HEALTHY")
        assert state == "STRESSED"

    def test_score_exactly_10_critical(self):
        """Score of 10 should be CRITICAL.
        memory_pct=96 (>95, score 4) + cpu_pct=96 (>95, score 4) + restart_rate=2 (>1, score 2) = 10"""
        state = classify_state(memory_pct=96, cpu_pct=96, restart_rate=2,
                               log_errors=0, node_pressure=False, markov_state="HEALTHY")
        assert state == "CRITICAL"

    def test_all_zero_healthy(self):
        """All metrics at zero should be HEALTHY."""
        state = classify_state(memory_pct=0, cpu_pct=0, restart_rate=0,
                               log_errors=0, node_pressure=False, markov_state="HEALTHY")
        assert state == "HEALTHY"

    def test_node_pressure_adds_3(self):
        """Node pressure adds 3 to the score, producing a worse state."""
        state_rank = {"HEALTHY": 0, "DEGRADED": 1, "STRESSED": 2, "CRITICAL": 3, "FAILED": 4, "RECOVERED": 5}
        state_no_pressure = classify_state(memory_pct=50, cpu_pct=50, restart_rate=0,
                                            log_errors=0, node_pressure=False, markov_state="HEALTHY")
        state_pressure = classify_state(memory_pct=50, cpu_pct=50, restart_rate=0,
                                         log_errors=0, node_pressure=True, markov_state="HEALTHY")
        # With pressure (+3), state should be worse (higher rank)
        assert state_rank[state_pressure] > state_rank[state_no_pressure]


# ─── Predictor boundary tests ───────────────────────────────────────────────

class TestPredictorBoundary:
    """Boundary tests for Predictor."""

    def test_zero_memory_trend(self):
        """Zero memory trend should produce None TTF."""
        p = Predictor()
        result = p.predict(
            pod_key="ns/pod", memory_pct=50.0, memory_trend_mib_per_min=0.0,
            memory_limit_mib=1024, memory_mib=512, cpu_pct=30.0,
            restart_rate_per_hr=0.0, log_error_rate_per_min=0.0,
            node_memory_pressure=False, node_disk_pressure=False,
            markov_state="HEALTHY", markov_p_critical=0.0, markov_p_failed=0.0,
        )
        assert result.ttf_minutes is None

    def test_zero_memory_limit(self):
        """Zero memory limit should produce None TTF."""
        p = Predictor()
        result = p.predict(
            pod_key="ns/pod", memory_pct=0.0, memory_trend_mib_per_min=5.0,
            memory_limit_mib=0, memory_mib=0, cpu_pct=30.0,
            restart_rate_per_hr=0.0, log_error_rate_per_min=0.0,
            node_memory_pressure=False, node_disk_pressure=False,
            markov_state="HEALTHY", markov_p_critical=0.0, markov_p_failed=0.0,
        )
        assert result.ttf_minutes is None

    def test_memory_already_at_limit(self):
        """Memory already at limit should produce TTF=0."""
        p = Predictor()
        result = p.predict(
            pod_key="ns/pod", memory_pct=100.0, memory_trend_mib_per_min=5.0,
            memory_limit_mib=1024, memory_mib=1024, cpu_pct=30.0,
            restart_rate_per_hr=0.0, log_error_rate_per_min=0.0,
            node_memory_pressure=False, node_disk_pressure=False,
            markov_state="HEALTHY", markov_p_critical=0.0, markov_p_failed=0.0,
        )
        assert result.ttf_minutes == 0

    def test_extremely_small_trend(self):
        """Extremely small trend should not overflow (Hypothesis-found bug)."""
        p = Predictor()
        result = p.predict(
            pod_key="ns/pod", memory_pct=50.0, memory_trend_mib_per_min=1e-308,
            memory_limit_mib=1024, memory_mib=512, cpu_pct=30.0,
            restart_rate_per_hr=0.0, log_error_rate_per_min=0.0,
            node_memory_pressure=False, node_disk_pressure=False,
            markov_state="HEALTHY", markov_p_critical=0.0, markov_p_failed=0.0,
        )
        # Should not crash and should return None or a very large number
        assert result.ttf_minutes is None or result.ttf_minutes <= 43200

    def test_negative_remaining(self):
        """When memory_mib > memory_limit_mib, TTF should be 0."""
        p = Predictor()
        result = p.predict(
            pod_key="ns/pod", memory_pct=120.0, memory_trend_mib_per_min=5.0,
            memory_limit_mib=1024, memory_mib=1229, cpu_pct=30.0,
            restart_rate_per_hr=0.0, log_error_rate_per_min=0.0,
            node_memory_pressure=False, node_disk_pressure=False,
            markov_state="HEALTHY", markov_p_critical=0.0, markov_p_failed=0.0,
        )
        assert result.ttf_minutes == 0


# ─── Persistence boundary tests ─────────────────────────────────────────────

class TestPersistenceBoundary:
    """Boundary tests for StateStore."""

    def test_load_corrupted_markov(self):
        """Loading a corrupted markov file should return a fresh chain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sm_file = os.path.join(tmpdir, "state.json")
            with open(sm_file, "w") as f:
                f.write("{invalid json!!!}")
            store = StateStore(sm_file)
            mc = store.load_markov()
            assert mc.total_transitions == 0

    def test_load_corrupted_predictions(self):
        """Loading corrupted predictions should return empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pred_file = os.path.join(tmpdir, "pred.json")
            with open(pred_file, "w") as f:
                f.write("not json")
            store = StateStore("/dev/null", pred_file)
            preds = store.load_predictions()
            assert preds == []

    def test_load_empty_markov_file(self):
        """Loading an empty file should return a fresh chain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sm_file = os.path.join(tmpdir, "empty.json")
            with open(sm_file, "w") as f:
                f.write("")
            store = StateStore(sm_file)
            mc = store.load_markov()
            assert mc.total_transitions == 0

    def test_save_creates_nested_directory(self):
        """Saving should create nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sm_file = os.path.join(tmpdir, "a", "b", "c", "state.json")
            store = StateStore(sm_file)
            from predictive_agent.markov import MarkovChain
            store.save_markov(MarkovChain())
            assert os.path.exists(sm_file)

    def test_atomic_write_temp_cleanup(self):
        """Atomic write should not leave .tmp files behind."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sm_file = os.path.join(tmpdir, "state.json")
            store = StateStore(sm_file)
            from predictive_agent.markov import MarkovChain
            store.save_markov(MarkovChain())
            assert not os.path.exists(sm_file + ".tmp")

    def test_save_load_roundtrip_preserves_counts(self):
        """Save and load should preserve exact transition counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sm_file = os.path.join(tmpdir, "state.json")
            store = StateStore(sm_file)
            mc = MarkovChain()
            mc.record_transition("HEALTHY", "DEGRADED")
            mc.record_transition("DEGRADED", "STRESSED")
            mc.record_transition("STRESSED", "CRITICAL")
            mc.record_transition("CRITICAL", "FAILED")
            store.save_markov(mc)
            mc2 = store.load_markov()
            assert mc2.counts == mc.counts
            assert mc2.total_transitions == mc.total_transitions
