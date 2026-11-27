"""B4: positive-class observability for the backtester.

A confusion matrix with 0 TP / 0 FN reads as "perfect detection", but in a
healthy cluster it really means the positive class is starved and recall is
undefined. These tests pin the honest-footprint fields (positive_samples,
positive_rate, validated_on_failures) and the Prometheus gauges exposed on
/metrics so operators can tell a validated model from one that never saw a
failure.
"""

from unittest import mock

from predictive_agent.backtester import (
    MIN_POSITIVE_SAMPLES_FOR_VALIDATION,
    Backtester,
)


def _pred(pod_key, risk, ttf=30, markov="HEALTHY"):
    bt_kwargs = dict(
        pod_key=pod_key, risk_score=risk, ttf_minutes=ttf, confidence=0.8,
        markov_state=markov, memory_pct=40, cpu_pct=40,
        memory_trend=0.0, cpu_trend=0.0,
    )
    return bt_kwargs


def _realistic_healthy_history(bt: Backtester) -> None:
    """10k healthy pairs like the live cluster: everything resolves healthy,
    predicted risk around 0.1-0.6, a few at/above the 0.5 action threshold."""
    for i in range(10000):
        risk = round(0.1 + (i % 50) * 0.01, 4)  # 0.10 .. 0.59
        bt.record_prediction(**_pred(f"ns/pod-{i % 40}", risk))
        bt.record_outcome(
            pod_key=f"ns/pod-{i % 40}", actual_failure=False,
            restart_count=0, state="HEALTHY", actual_ttf_minutes=None,
        )


def _failure_spiked_history(bt: Backtester) -> None:
    _realistic_healthy_history(bt)
    # 15 genuine failures: pods that actually went FAILED and were predicted.
    for i in range(15):
        bt.record_prediction(**_pred(f"ns/bad-{i}", 0.8, 10, "DEGRADED"))
        bt.record_outcome(
            pod_key=f"ns/bad-{i}", actual_failure=True, restart_count=8,
            state="FAILED", actual_ttf_minutes=5,
        )


class TestPositiveClassFootprint:
    def test_healthy_cluster_is_not_failure_validated(self, tmp_path):
        bt = Backtester(history_file=str(tmp_path / "h.jsonl"))
        _realistic_healthy_history(bt)
        report = bt.evaluate()
        assert report.positive_samples == 0
        assert report.positive_rate == 0.0
        assert report.validated_on_failures is False
        assert report.false_negatives == 0
        # The healthy-cluster Brier is still meaningful (false-positive rate),
        # and recall must *not* be presented as validated.
        assert report.to_dict()["validated_on_failures"] is False

    def test_failures_observed_flips_validation_flag(self, tmp_path):
        bt = Backtester(history_file=str(tmp_path / "h.jsonl"))
        _failure_spiked_history(bt)
        report = bt.evaluate()
        assert report.positive_samples == 15
        assert report.positive_rate > 0.0
        assert report.validated_on_failures is True
        assert report.true_positives == 15
        assert report.false_negatives == 0

    def test_few_failures_are_not_enough_for_validation(self, tmp_path):
        bt = Backtester(history_file=str(tmp_path / "h.jsonl"))
        _realistic_healthy_history(bt)
        for i in range(MIN_POSITIVE_SAMPLES_FOR_VALIDATION - 1):
            bt.record_prediction(**_pred(f"ns/r-{i}", 0.8, 5, "DEGRADED"))
            bt.record_outcome(
                pod_key=f"ns/r-{i}", actual_failure=True, restart_count=5,
                state="FAILED", actual_ttf_minutes=3,
            )
        report = bt.evaluate()
        assert report.positive_samples == MIN_POSITIVE_SAMPLES_FOR_VALIDATION - 1
        assert report.validated_on_failures is False

    def test_healthy_report_still_scores_brier(self, tmp_path):
        bt = Backtester(history_file=str(tmp_path / "h.jsonl"))
        _realistic_healthy_history(bt)
        report = bt.evaluate()
        assert report.matched_pairs >= 9000
        assert 0.0 < report.brier_score < 0.25


class TestBacktestPrometheusGauges:
    def test_gauges_emitted_with_starved_positive_class(self):
        from predictive_agent.metrics import collect_backtest_metrics

        bt = mock.MagicMock()
        bt.evaluate.return_value = mock.MagicMock(
            matched_pairs=10000, brier_score=0.0335, true_positives=0,
            false_positives=55, true_negatives=9945, false_negatives=0,
            positive_samples=0, positive_rate=0.0, validated_on_failures=False,
        )
        metrics = {name: value for name, _, _, value in collect_backtest_metrics(bt)}
        assert metrics["backtest_positive_samples"] == 0.0
        assert metrics["backtest_failure_validated"] == 0.0
        assert metrics["backtest_false_negatives"] == 0.0
        assert metrics["backtest_matched_pairs"] == 10000.0

    def test_gauges_flag_real_failures(self):
        from predictive_agent.metrics import collect_backtest_metrics

        bt = mock.MagicMock()
        bt.evaluate.return_value = mock.MagicMock(
            matched_pairs=100, brier_score=0.1, true_positives=12,
            false_positives=2, true_negatives=80, false_negatives=6,
            positive_samples=18, positive_rate=0.18, validated_on_failures=True,
        )
        metrics = {name: value for name, _, _, value in collect_backtest_metrics(bt)}
        assert metrics["backtest_failure_validated"] == 1.0
        assert metrics["backtest_positive_samples"] == 18.0

    def test_none_backtester_is_empty(self):
        from predictive_agent.metrics import collect_backtest_metrics

        assert collect_backtest_metrics(None) == []
