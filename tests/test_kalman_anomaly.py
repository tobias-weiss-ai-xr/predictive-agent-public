"""Tests for KalmanTrend anomaly_score and time_to_threshold confidence fix."""
from predictive_agent.kalman import KalmanTrend


class TestAnomalyScore:
    """Test the anomaly_score method for 3-sigma spike detection."""

    def test_anomaly_score_zero_on_init(self):
        """Anomaly score should be 0 before any update."""
        kt = KalmanTrend()
        assert kt.anomaly_score(100.0) == 0.0

    def test_anomaly_score_zero_for_stable_data(self):
        """Anomaly score should be low for data consistent with the filter."""
        kt = KalmanTrend(process_noise=0.1, measurement_noise=10.0)
        for val in [100, 100, 100, 100, 100]:
            kt.update(float(val))
        # Measurement close to level → low anomaly
        score = kt.anomaly_score(101.0)
        assert score < 2.0

    def test_anomaly_score_high_for_spike(self):
        """Anomaly score should be high for a sudden spike."""
        kt = KalmanTrend(process_noise=0.1, measurement_noise=10.0)
        for val in [100, 100, 100, 100, 100]:
            kt.update(float(val))
        # Large spike → high anomaly
        score = kt.anomaly_score(500.0)
        assert score > 3.0

    def test_anomaly_score_increases_with_spike_magnitude(self):
        """Larger spikes should produce higher anomaly scores."""
        kt = KalmanTrend(process_noise=0.1, measurement_noise=10.0)
        for val in [100, 100, 100, 100, 100]:
            kt.update(float(val))
        score_small = kt.anomaly_score(110.0)
        score_large = kt.anomaly_score(200.0)
        assert score_large > score_small

    def test_anomaly_score_zero_uncertainty(self):
        """When level uncertainty is ~0, anomaly score should be 0 (avoid div by zero)."""
        kt = KalmanTrend(process_noise=0.0, measurement_noise=0.0)
        kt.update(100.0)
        # With zero noise, P[0][0] = 0 → uncertainty = 0 → anomaly = 0
        score = kt.anomaly_score(200.0)
        assert score == 0.0


class TestTimeToThresholdConfidence:
    """Test the fixed time_to_threshold confidence calculation."""

    def test_ttf_confidence_in_range(self):
        """Confidence should be between 0 and 1."""
        kt = KalmanTrend()
        kt.update(100.0)
        kt.update(105.0)
        kt.update(110.0)
        steps, confidence = kt.time_to_threshold(150.0, max_steps=100)
        assert 0 <= confidence <= 1.0

    def test_ttf_confidence_high_for_clear_trend(self):
        """Confidence should be high when trend is clear and consistent."""
        kt = KalmanTrend(process_noise=0.01, measurement_noise=1.0)
        # Very consistent upward trend
        for val in [100, 110, 120, 130, 140]:
            kt.update(float(val))
        steps, confidence = kt.time_to_threshold(200.0, max_steps=100)
        assert confidence > 0.5  # Should be fairly confident

    def test_ttf_confidence_low_for_noisy_trend(self):
        """Confidence should be lower when data is noisy."""
        kt = KalmanTrend(process_noise=10.0, measurement_noise=1000.0)
        # Noisy upward trend
        for val in [100, 90, 120, 80, 130, 70, 140]:
            kt.update(float(val))
        steps, confidence = kt.time_to_threshold(200.0, max_steps=200)
        # With high noise, confidence should be moderate or low
        assert confidence < 0.95

    def test_ttf_zero_velocity(self):
        """Zero or negative velocity should return max_steps with 0 confidence."""
        kt = KalmanTrend()
        kt.update(100.0)
        kt.update(100.0)
        kt.update(100.0)
        steps, confidence = kt.time_to_threshold(200.0, max_steps=100)
        assert steps == 100  # max_steps
        assert confidence == 0.0

    def test_ttf_already_at_threshold(self):
        """When already at or above threshold, steps=0, confidence=1."""
        kt = KalmanTrend()
        kt.update(150.0)
        kt.update(155.0)
        # Threshold is 150, level is ~155 → already above
        steps, confidence = kt.time_to_threshold(150.0, max_steps=100)
        assert steps == 0
        assert confidence == 1.0

    def test_ttf_confidence_far_threshold(self):
        """Far thresholds should have reasonable confidence (large distance vs uncertainty)."""
        kt = KalmanTrend(process_noise=1.0, measurement_noise=50.0)
        for val in [100, 110, 120]:
            kt.update(float(val))
        # Far threshold — distance is large relative to prediction uncertainty
        steps_far, conf_far = kt.time_to_threshold(300.0, max_steps=200)
        assert conf_far > 0.0  # Should have some confidence
        assert 0 <= conf_far <= 1.0
