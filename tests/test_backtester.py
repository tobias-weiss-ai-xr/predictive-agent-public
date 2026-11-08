"""Tests for backtester prediction accuracy evaluation."""
import os
import tempfile

import pytest

from predictive_agent.backtester import Backtester, OutcomeRecord


@pytest.fixture
def bt():
    """Fresh backtester with temp file."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    os.unlink(path)  # Remove so backtester starts fresh
    yield Backtester(history_file=path)
    if os.path.exists(path):
        os.unlink(path)


class TestBacktester:
    def test_record_prediction(self, bt):
        """Test recording a prediction."""
        bt.record_prediction(
            pod_key="ns/pod-1",
            risk_score=0.7,
            ttf_minutes=15,
            confidence=0.8,
            markov_state="STRESSED",
        )
        assert bt.get_prediction_count() == 1

    def test_history_is_capped(self, bt):
        """Regression: in-memory prediction/outcome history must be bounded
        to prevent unbounded growth over long uptimes."""
        from predictive_agent.backtester import MAX_HISTORY_RECORDS
        # Record more than the cap
        for i in range(MAX_HISTORY_RECORDS + 500):
            bt.record_prediction(
                pod_key=f"ns/pod-{i % 10}",
                risk_score=0.5,
                ttf_minutes=30,
                confidence=0.7,
                markov_state="HEALTHY",
            )
            bt.record_outcome(
                pod_key=f"ns/pod-{i % 10}",
                actual_failure=False,
                restart_count=0,
                state="HEALTHY",
            )
        assert bt.get_prediction_count() <= MAX_HISTORY_RECORDS
        assert len(bt._outcomes) <= MAX_HISTORY_RECORDS

    def test_record_outcome(self, bt):
        """Test recording an outcome."""
        bt.record_outcome(
            pod_key="ns/pod-1",
            actual_failure=True,
            restart_count=3,
            state="CRITICAL",
        )
        assert bt.get_outcome_count() == 1

    def test_evaluate_empty(self, bt):
        """Evaluation with no data should return zeroed report."""
        report = bt.evaluate()
        assert report.total_predictions == 0
        assert report.total_outcomes == 0
        assert report.matched_pairs == 0
        assert report.brier_score == 0.0

    def test_evaluate_perfect_prediction(self, bt):
        """Test evaluation with perfect predictions."""
        # Record prediction: high risk
        bt.record_prediction(
            pod_key="ns/pod-1",
            risk_score=0.9,
            ttf_minutes=10,
            confidence=0.9,
            markov_state="CRITICAL",
        )
        # Record outcome: actual failure
        bt.record_outcome(
            pod_key="ns/pod-1",
            actual_failure=True,
            restart_count=5,
            state="CRITICAL",
        )
        report = bt.evaluate()
        assert report.matched_pairs == 1
        assert report.true_positives == 1
        assert report.false_positives == 0
        assert report.false_negatives == 0
        # Brier score: (0.9 - 1.0)^2 = 0.01
        assert report.brier_score < 0.02

    def test_evaluate_false_positive(self, bt):
        """Test evaluation with false positive."""
        bt.record_prediction(
            pod_key="ns/pod-1",
            risk_score=0.8,
            ttf_minutes=10,
            confidence=0.8,
            markov_state="STRESSED",
        )
        bt.record_outcome(
            pod_key="ns/pod-1",
            actual_failure=False,
            restart_count=0,
            state="HEALTHY",
        )
        report = bt.evaluate()
        assert report.false_positives == 1
        assert report.true_positives == 0
        # Brier score: (0.8 - 0.0)^2 = 0.64
        assert report.brier_score > 0.6

    def test_evaluate_false_negative(self, bt):
        """Test evaluation with false negative."""
        bt.record_prediction(
            pod_key="ns/pod-1",
            risk_score=0.1,
            ttf_minutes=None,
            confidence=0.5,
            markov_state="HEALTHY",
        )
        bt.record_outcome(
            pod_key="ns/pod-1",
            actual_failure=True,
            restart_count=5,
            state="CRITICAL",
        )
        report = bt.evaluate()
        assert report.false_negatives == 1
        assert report.true_positives == 0

    def test_evaluate_true_negative(self, bt):
        """Test evaluation with true negative."""
        bt.record_prediction(
            pod_key="ns/pod-1",
            risk_score=0.05,
            ttf_minutes=None,
            confidence=0.9,
            markov_state="HEALTHY",
        )
        bt.record_outcome(
            pod_key="ns/pod-1",
            actual_failure=False,
            restart_count=0,
            state="HEALTHY",
        )
        report = bt.evaluate()
        assert report.true_negatives == 1
        assert report.brier_score < 0.01

    def test_evaluate_ttf_error(self, bt):
        """Test TTF error tracking."""
        bt.record_prediction(
            pod_key="ns/pod-1",
            risk_score=0.7,
            ttf_minutes=30,
            confidence=0.7,
            markov_state="STRESSED",
        )
        bt.record_outcome(
            pod_key="ns/pod-1",
            actual_failure=True,
            actual_ttf_minutes=25,
            restart_count=3,
            state="CRITICAL",
        )
        report = bt.evaluate()
        assert len(report.ttf_errors) == 1
        assert report.ttf_errors[0]["error_minutes"] == 5

    def test_calibration_bins(self, bt):
        """Test that calibration bins are populated."""
        for i in range(10):
            bt.record_prediction(
                pod_key=f"ns/pod-{i}",
                risk_score=0.3 + i * 0.05,
                ttf_minutes=10,
                confidence=0.7,
                markov_state="STRESSED",
            )
            bt.record_outcome(
                pod_key=f"ns/pod-{i}",
                actual_failure=(i > 5),
                restart_count=0,
                state="CRITICAL" if i > 5 else "HEALTHY",
            )
        report = bt.evaluate()
        assert len(report.calibration) > 0
        # Each bin should have count and rates
        for bin_key, bin_data in report.calibration.items():
            assert "count" in bin_data
            assert "avg_predicted" in bin_data
            assert "actual_rate" in bin_data

    def test_persistence_roundtrip(self, bt):
        """Test that records are persisted to file and can be reloaded."""
        bt.record_prediction(
            pod_key="ns/pod-1",
            risk_score=0.7,
            ttf_minutes=15,
            confidence=0.8,
            markov_state="STRESSED",
        )
        bt.record_outcome(
            pod_key="ns/pod-1",
            actual_failure=True,
            restart_count=2,
            state="CRITICAL",
        )
        # Reload from file
        bt2 = Backtester(history_file=bt.history_file)
        assert bt2.get_prediction_count() == 1
        assert bt2.get_outcome_count() == 1
        report = bt2.evaluate()
        assert report.matched_pairs == 1

    def test_clear_history(self, bt):
        """Test clearing in-memory history."""
        bt.record_prediction(
            pod_key="ns/pod-1",
            risk_score=0.5,
            ttf_minutes=10,
            confidence=0.7,
            markov_state="HEALTHY",
        )
        assert bt.get_prediction_count() == 1
        bt.clear_history()
        assert bt.get_prediction_count() == 0

    def test_multiple_pods_evaluation(self, bt):
        """Test evaluation with multiple pods."""
        pods = [
            ("ns/pod-1", 0.9, True),   # TP
            ("ns/pod-2", 0.1, False),  # TN
            ("ns/pod-3", 0.8, False),  # FP
            ("ns/pod-4", 0.2, True),   # FN
        ]
        for pod_key, risk, failure in pods:
            bt.record_prediction(
                pod_key=pod_key,
                risk_score=risk,
                ttf_minutes=10 if risk > 0.5 else None,
                confidence=0.7,
                markov_state="STRESSED" if risk > 0.5 else "HEALTHY",
            )
            bt.record_outcome(
                pod_key=pod_key,
                actual_failure=failure,
                restart_count=3 if failure else 0,
                state="CRITICAL" if failure else "HEALTHY",
            )
        report = bt.evaluate()
        assert report.matched_pairs == 4
        assert report.true_positives == 1
        assert report.true_negatives == 1
        assert report.false_positives == 1
        assert report.false_negatives == 1

    def test_time_window_filtering(self, bt):
        """Test that outcomes outside the time window are not matched."""
        import time as _time
        bt.record_prediction(
            pod_key="ns/pod-1",
            risk_score=0.7,
            ttf_minutes=10,
            confidence=0.7,
            markov_state="STRESSED",
        )
        # Manually add an outcome with an old timestamp
        old_outcome = OutcomeRecord(
            pod_key="ns/pod-1",
            timestamp=_time.time() - 7200,  # 2 hours ago
            actual_failure=True,
            actual_ttf_minutes=None,
            restart_count=3,
            state="CRITICAL",
        )
        bt._outcomes.append(old_outcome)
        # With 1-hour window, should not match
        report = bt.evaluate(time_window=3600)
        assert report.matched_pairs == 0
