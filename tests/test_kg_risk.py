"""Tests for KG integration (blast radius) and risk scoring with new signals."""

import pytest

from predictive_agent.kg_integration import KnowledgeGraphClient, get_kg_client
from predictive_agent.predictor import Predictor
from predictive_agent.risk import calculate_risk


class TestKnowledgeGraphClient:
    """Test the Dgraph knowledge graph client."""

    def test_kg_client_creation(self):
        """Test that client can be created."""
        client = KnowledgeGraphClient(url="http://localhost:9999")
        assert client.url == "http://localhost:9999"

    def test_kg_client_health_unreachable(self):
        """Test that health check returns False for unreachable Dgraph."""
        client = KnowledgeGraphClient(url="http://localhost:9999", timeout=1)
        assert client.health() is False

    def test_kg_blast_radius_unreachable(self):
        """Test that blast radius returns 0 when Dgraph is unreachable."""
        client = KnowledgeGraphClient(url="http://localhost:9999", timeout=1)
        radius = client.get_blast_radius("ns/pod-1")
        assert radius == 0

    def test_kg_blast_radius_caching(self):
        """Test that blast radius results are cached."""
        client = KnowledgeGraphClient(url="http://localhost:9999", timeout=1)
        # First call hits Dgraph (returns 0)
        radius1 = client.get_blast_radius("ns/pod-1")
        assert radius1 == 0
        # Second call should use cache (still 0)
        radius2 = client.get_blast_radius("ns/pod-1")
        assert radius2 == 0

    def test_kg_clear_cache(self):
        """Test clearing the cache."""
        client = KnowledgeGraphClient(url="http://localhost:9999", timeout=1)
        client.get_blast_radius("ns/pod-1")
        assert len(client._cache) > 0
        client.clear_cache()
        assert len(client._cache) == 0

    def test_get_kg_client_singleton(self):
        """Test that get_kg_client returns a singleton."""
        c1 = get_kg_client()
        c2 = get_kg_client()
        assert c1 is c2


class TestRiskWithAnomalyAndBlastRadius:
    """Test that anomaly scores and blast radius affect risk scoring."""

    def test_anomaly_score_increases_risk(self):
        """High anomaly score should increase risk."""
        base_metrics = {
            "memory_pct": 70.0,
            "cpu_pct": 50.0,
            "restart_rate_per_hr": 0.0,
            "log_error_rate_per_min": 0.0,
        }
        risk_low = calculate_risk(base_metrics, "HEALTHY", 0.0, 0.0)
        risk_high = calculate_risk(
            {**base_metrics, "memory_anomaly_score": 5.0},
            "HEALTHY", 0.0, 0.0,
        )
        assert risk_high > risk_low

    def test_cpu_anomaly_score_increases_risk(self):
        """High CPU anomaly score should increase risk."""
        base_metrics = {
            "memory_pct": 70.0,
            "cpu_pct": 50.0,
            "restart_rate_per_hr": 0.0,
            "log_error_rate_per_min": 0.0,
        }
        risk_low = calculate_risk(base_metrics, "HEALTHY", 0.0, 0.0)
        risk_high = calculate_risk(
            {**base_metrics, "cpu_anomaly_score": 4.0},
            "HEALTHY", 0.0, 0.0,
        )
        assert risk_high > risk_low

    def test_blast_radius_increases_risk(self):
        """High blast radius should increase risk."""
        base_metrics = {
            "memory_pct": 70.0,
            "cpu_pct": 50.0,
            "restart_rate_per_hr": 0.0,
            "log_error_rate_per_min": 0.0,
        }
        risk_low = calculate_risk(base_metrics, "HEALTHY", 0.0, 0.0)
        risk_high = calculate_risk(
            {**base_metrics, "blast_radius": 30},
            "HEALTHY", 0.0, 0.0,
        )
        assert risk_high > risk_low

    def test_blast_radius_zero_no_effect(self):
        """Blast radius of 0 should not increase risk."""
        base_metrics = {
            "memory_pct": 70.0,
            "cpu_pct": 50.0,
            "restart_rate_per_hr": 0.0,
            "log_error_rate_per_min": 0.0,
        }
        risk_base = calculate_risk(base_metrics, "HEALTHY", 0.0, 0.0)
        risk_with_zero = calculate_risk(
            {**base_metrics, "blast_radius": 0},
            "HEALTHY", 0.0, 0.0,
        )
        assert risk_with_zero == pytest.approx(risk_base)

    def test_cpu_trend_increases_risk(self):
        """High CPU trend should increase risk."""
        base_metrics = {
            "memory_pct": 70.0,
            "cpu_pct": 50.0,
            "restart_rate_per_hr": 0.0,
            "log_error_rate_per_min": 0.0,
        }
        risk_low = calculate_risk(base_metrics, "HEALTHY", 0.0, 0.0)
        risk_high = calculate_risk(
            {**base_metrics, "cpu_trend_m_per_min": 60.0},
            "HEALTHY", 0.0, 0.0,
        )
        assert risk_high > risk_low


class TestPredictorWithNewSignals:
    """Test that the predictor accepts and stores new signals."""

    def test_predictor_stores_cpu_trend(self):
        """Test that cpu_trend is stored in the prediction result."""
        p = Predictor(risk_threshold=0.5)
        result = p.predict(
            pod_key="ns/pod-0",
            memory_pct=50.0,
            memory_trend_mib_per_min=1.0,
            memory_limit_mib=1024,
            memory_mib=512,
            cpu_pct=30.0,
            restart_rate_per_hr=0.0,
            log_error_rate_per_min=0.0,
            node_memory_pressure=False,
            node_disk_pressure=False,
            markov_state="HEALTHY",
            markov_p_critical=0.0,
            markov_p_failed=0.0,
            cpu_trend_m_per_min=50.0,
        )
        assert result.cpu_trend == 50.0

    def test_predictor_stores_anomaly_score(self):
        """Test that anomaly_score is stored in the prediction result."""
        p = Predictor(risk_threshold=0.5)
        result = p.predict(
            pod_key="ns/pod-0",
            memory_pct=50.0,
            memory_trend_mib_per_min=1.0,
            memory_limit_mib=1024,
            memory_mib=512,
            cpu_pct=30.0,
            restart_rate_per_hr=0.0,
            log_error_rate_per_min=0.0,
            node_memory_pressure=False,
            node_disk_pressure=False,
            markov_state="HEALTHY",
            markov_p_critical=0.0,
            markov_p_failed=0.0,
            memory_anomaly_score=3.5,
            cpu_anomaly_score=2.0,
        )
        assert result.anomaly_score == 3.5  # max(3.5, 2.0)

    def test_predictor_stores_blast_radius(self):
        """Test that blast_radius is stored in the prediction result."""
        p = Predictor(risk_threshold=0.5)
        result = p.predict(
            pod_key="ns/pod-0",
            memory_pct=50.0,
            memory_trend_mib_per_min=1.0,
            memory_limit_mib=1024,
            memory_mib=512,
            cpu_pct=30.0,
            restart_rate_per_hr=0.0,
            log_error_rate_per_min=0.0,
            node_memory_pressure=False,
            node_disk_pressure=False,
            markov_state="HEALTHY",
            markov_p_critical=0.0,
            markov_p_failed=0.0,
            blast_radius=15,
        )
        assert result.blast_radius == 15

    def test_predictor_default_values(self):
        """Test that new parameters default to 0 when not provided."""
        p = Predictor(risk_threshold=0.5)
        result = p.predict(
            pod_key="ns/pod-0",
            memory_pct=50.0,
            memory_trend_mib_per_min=1.0,
            memory_limit_mib=1024,
            memory_mib=512,
            cpu_pct=30.0,
            restart_rate_per_hr=0.0,
            log_error_rate_per_min=0.0,
            node_memory_pressure=False,
            node_disk_pressure=False,
            markov_state="HEALTHY",
            markov_p_critical=0.0,
            markov_p_failed=0.0,
        )
        assert result.cpu_trend == 0.0
        assert result.anomaly_score == 0.0
        assert result.blast_radius == 0
