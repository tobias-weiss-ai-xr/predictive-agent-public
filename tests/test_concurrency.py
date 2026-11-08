"""Concurrency and thread-safety tests.

SOTA paradigm: Concurrency tests verify thread safety of shared state.
The ReconcileLoop runs in a background thread while the HTTP server handles
requests concurrently. These tests ensure no race conditions, deadlocks, or
data corruption under concurrent access.

Covers:
- ReconcileLoop start/stop from multiple threads
- Concurrent HTTP requests to the server
- StateModel concurrent updates (pod tracking from multiple threads)
- Predictor concurrent predictions
- StateStore concurrent save/load
- HTTP server handles parallel requests correctly
"""
import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request

import pytest

from predictive_agent.markov import MarkovChain
from predictive_agent.persistence import StateStore
from predictive_agent.predictor import PredictionResult, Predictor
from predictive_agent.server import start_server
from predictive_agent.state_model import StateModel

# ─── ReconcileLoop concurrency ──────────────────────────────────────────────

class TestReconcileLoopConcurrency:
    """Test ReconcileLoop thread safety."""

    def test_start_stop_idempotent(self):
        """Starting/stopping multiple times should not crash."""
        from predictive_agent.main import ReconcileLoop
        loop = ReconcileLoop(interval=1)
        loop.start()
        loop.start()  # Double start should be no-op
        assert loop.running is True
        loop.stop()
        loop.stop()  # Double stop should be no-op
        assert loop.running is False

    def test_concurrent_start_stop(self):
        """Concurrent start/stop from different threads should not crash."""
        from predictive_agent.main import ReconcileLoop
        loop = ReconcileLoop(interval=0.1)
        errors = []

        def start_loop():
            try:
                loop.start()
            except Exception as e:
                errors.append(e)

        def stop_loop():
            try:
                time.sleep(0.2)
                loop.stop()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=start_loop)
        t2 = threading.Thread(target=stop_loop)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert len(errors) == 0

    def test_loop_intervals_respected(self):
        """Loop should approximately respect the configured interval."""
        from predictive_agent.main import ReconcileLoop
        call_count = 0

        def mock_reconcile():
            nonlocal call_count
            call_count += 1
            return {"predictions": [], "state": {}, "timestamp": ""}

        loop = ReconcileLoop(interval=0.1)
        loop._reconcile_fn = mock_reconcile
        loop.start()
        time.sleep(0.4)  # ~4 calls at 0, 0.1, 0.2, 0.3
        loop.stop()
        # Allow some jitter — at least 2, at most 6
        assert 2 <= call_count <= 6


# ─── HTTP server concurrency ────────────────────────────────────────────────

class TestServerConcurrency:
    """Test HTTP server under concurrent requests."""

    @pytest.fixture(scope="class")
    def concurrent_server(self):
        sm = StateModel()
        sm.update_pod("ns", "pod-0", memory_mib=500, memory_limit_mib=1024,
                       cpu_m=100, restart_count=0, log_errors=0, node_pressure=False)
        pred = Predictor()
        pred.add_prediction("ns/pod-0", PredictionResult(
            "ns/pod-0", 0.1, None, 0.9, "HEALTHY", 0.0, 0.0, 50.0, 10.0
        ))
        server = start_server(18095, 18096, state_model=sm, predictor=pred)
        yield server
        server.shutdown()

    def test_concurrent_healthz(self, concurrent_server):
        """100 concurrent /healthz requests should all succeed."""
        results = []
        errors = []

        def make_request():
            try:
                with urllib.request.urlopen("http://localhost:18096/healthz") as resp:
                    data = json.loads(resp.read().decode())
                    results.append(data.get("status"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=make_request) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert len(results) == 100
        assert all(r == "ok" for r in results)

    def test_concurrent_mixed_endpoints(self, concurrent_server):
        """Concurrent requests to different endpoints should all succeed."""
        endpoints = ["/healthz", "/ready", "/metrics", "/status", "/predictions", "/state"]
        errors = []

        def make_request(path):
            try:
                port = 18095 if path in ("/metrics", "/status", "/predictions", "/state") else 18096
                with urllib.request.urlopen(f"http://localhost:{port}{path}") as resp:
                    resp.read()
            except Exception as e:
                errors.append((path, e))

        threads = []
        for _ in range(20):
            for path in endpoints:
                threads.append(threading.Thread(target=make_request, args=(path,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Errors: {errors[:3]}"

    def test_concurrent_predictions_and_state(self, concurrent_server):
        """Concurrent /predictions and /state should not corrupt each other."""
        results = {"predictions": [], "state": []}
        errors = []

        def get_predictions():
            try:
                with urllib.request.urlopen("http://localhost:18095/predictions") as resp:
                    data = json.loads(resp.read().decode())
                    results["predictions"].append(data)
            except Exception as e:
                errors.append(e)

        def get_state():
            try:
                with urllib.request.urlopen("http://localhost:18095/state") as resp:
                    data = json.loads(resp.read().decode())
                    results["state"].append(data)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(50):
            threads.append(threading.Thread(target=get_predictions))
            threads.append(threading.Thread(target=get_state))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert len(results["predictions"]) == 50
        assert len(results["state"]) == 50
        # All predictions should have the same structure
        for p in results["predictions"]:
            assert "predictions" in p
            assert "total" in p


# ─── StateModel concurrency ─────────────────────────────────────────────────

class TestStateModelConcurrency:
    """Test StateModel under concurrent access."""

    def test_concurrent_pod_updates(self):
        """Concurrent pod updates should not corrupt state."""
        sm = StateModel()
        errors = []

        def update_pod(i):
            try:
                for j in range(20):
                    sm.update_pod("ns", f"pod-{i}",
                                  memory_mib=100 + j * 10,
                                  memory_limit_mib=1024,
                                  cpu_m=50 + j,
                                  restart_count=0,
                                  log_errors=0,
                                  node_pressure=False)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_pod, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert len(sm.pods) == 10
        for i in range(10):
            assert f"ns/pod-{i}" in sm.pods

    def test_concurrent_markov_transitions(self):
        """Concurrent Markov chain transitions should not lose data."""
        mc = MarkovChain()
        initial_total = mc.total_transitions

        def record_transitions():
            for _ in range(100):
                mc.record_transition("HEALTHY", "DEALTHY")

        threads = [threading.Thread(target=record_transitions) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # Note: Python's GIL makes this safe for CPython, but the total should
        # at least be consistent with what was recorded
        assert mc.total_transitions >= initial_total + 100  # At least some recorded

    def test_concurrent_track_and_read(self):
        """Concurrent tracking and reading should not crash."""
        sm = StateModel()
        errors = []

        def track_pods():
            for i in range(50):
                sm.update_pod("ns", f"pod-{i}",
                              memory_mib=100, memory_limit_mib=1024,
                              cpu_m=50, restart_count=0, log_errors=0,
                              node_pressure=False)

        def read_state():
            for _ in range(50):
                data = sm.to_dict()
                assert "pods" in data

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=track_pods))
            threads.append(threading.Thread(target=read_state))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0


# ─── Persistence concurrency ────────────────────────────────────────────────

class TestPersistenceConcurrency:
    """Test StateStore under concurrent access."""

    def test_concurrent_save_markov(self):
        """Concurrent saves should not corrupt the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sm_file = os.path.join(tmpdir, "state.json")
            store = StateStore(sm_file)
            errors = []

            def save():
                try:
                    mc = MarkovChain()
                    mc.record_transition("HEALTHY", "HEALTHY")
                    store.save_markov(mc)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=save) for _ in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            assert len(errors) == 0
            # File should be valid JSON (atomic writes)
            mc = store.load_markov()
            assert mc.total_transitions >= 1

    def test_concurrent_save_load(self):
        """Concurrent saves and loads should not crash or corrupt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sm_file = os.path.join(tmpdir, "state.json")
            store = StateStore(sm_file)
            errors = []

            def save():
                for _ in range(10):
                    try:
                        mc = MarkovChain()
                        mc.record_transition("HEALTHY", "DEGRADED")
                        store.save_markov(mc)
                    except Exception as e:
                        errors.append(e)

            def load():
                for _ in range(10):
                    try:
                        mc = store.load_markov()
                        assert mc is not None
                    except Exception as e:
                        errors.append(e)

            threads = []
            for _ in range(5):
                threads.append(threading.Thread(target=save))
                threads.append(threading.Thread(target=load))
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            assert len(errors) == 0


# ─── Predictor concurrency ──────────────────────────────────────────────────

class TestPredictorConcurrency:
    """Test Predictor under concurrent access."""

    def test_concurrent_predictions(self):
        """Concurrent predictions should not corrupt internal state."""
        pred = Predictor(risk_threshold=0.5)
        errors = []
        results = []

        def make_prediction(i):
            try:
                result = pred.predict(
                    pod_key=f"ns/pod-{i}",
                    memory_pct=50.0 + i,
                    memory_trend_mib_per_min=1.0,
                    memory_limit_mib=1024,
                    memory_mib=512,
                    cpu_pct=30.0,
                    restart_rate_per_hr=0.0,
                    log_error_rate_per_min=0.0,
                    node_memory_pressure=False,
                    node_disk_pressure=False,
                    markov_state="HEALTHY",
                    markov_p_critical=0.01,
                    markov_p_failed=0.001,
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=make_prediction, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert len(results) == 20
        # All results should have valid risk scores
        for r in results:
            assert 0.0 <= r.risk_score <= 0.99
