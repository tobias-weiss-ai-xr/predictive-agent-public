"""Decision-layer Prometheus metrics collection (#7).

Aggregates operator-facing gauges/counters from the live
``RemediationManager`` (and optional ``Predictor`` / calibration monitor) so the
decision layer can be observed in production: intervention rate, mean
P(FAILED) reduction per intervention, learning volume, calibration, and
confidence. Exposed both as Prometheus text (via the server ``/metrics``
endpoint) and as JSON (``/decision-metrics``).
"""

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Each metric: (name_suffix, prometheus_type, help, value)
Metric = Tuple[str, str, str, float]

_PREFIX = "predictive_agent_"


def collect_backtest_metrics(backtester) -> List[Metric]:
    """Prometheus gauges for the backtest report (B4 observability).

    Includes the positive-class footprint so a starved 0 TP / 0 FN confusion
    matrix is not mistaken for perfect detection: ``backtest_positive_samples``
    and ``backtest_failure_validated`` make "recall undefined" explicit.
    """
    if backtester is None:
        return []
    try:
        report = backtester.evaluate()
    except Exception:  # pragma: no cover - defensive; report must not break /metrics
        logger.exception("Backtest evaluation failed during metric collection")
        return []
    prefix = "backtest_"
    g = lambda name, help_text, value: (  # noqa: E731
        prefix + name, "gauge", help_text, float(value)
    )
    return [
        g("matched_pairs", "Matched prediction/outcome pairs", report.matched_pairs),
        g("brier", "Brier score of P(FAILED) vs actual outcomes", report.brier_score),
        g("true_positives", "Matched pairs that failed and were predicted", report.true_positives),
        g("false_positives", "Matched pairs predicted to fail that did not", report.false_positives),
        g("true_negatives", "Matched pairs correctly predicted healthy", report.true_negatives),
        g("false_negatives", "Matched pairs that failed unpredicted", report.false_negatives),
        g("positive_samples", "Matched pairs with an actual failure (TP+FN)", report.positive_samples),
        g("positive_rate", "Share of matched pairs that actually failed", report.positive_rate),
        g("failure_validated", "1 when >= MIN_POSITIVE_SAMPLES real failures were observed",
          1 if report.validated_on_failures else 0),
    ]


def _learning_volume(rm) -> int:
    model = getattr(rm, "decision_model", None)
    counts = getattr(model, "action_counts", None)
    if not counts:
        return 0
    total = 0
    for state_rows in counts.values():
        for row in state_rows:
            total += sum(row)
    return total


def collect_decision_metrics(
    remediation_manager, predictor: Optional[object] = None
) -> List[Metric]:
    """Return the decision-layer metrics as (name_suffix, type, help, value)."""
    if remediation_manager is None:
        return []
    rm = remediation_manager
    metrics: List[Metric] = []

    interventions = getattr(rm, "interventions_executed", 0)
    cum_red = getattr(rm, "cumulative_p_failed_reduction", 0.0)
    mean_red = (cum_red / interventions) if interventions else 0.0

    metrics.append((
        "decision_layer_actions_registered", "gauge",
        "Number of registered remediation actions", len(getattr(rm, "actions", [])),
    ))
    metrics.append((
        "decision_layer_learning_transitions_total", "gauge",
        "Total action-conditioned transitions learned by the decision model",
        _learning_volume(rm),
    ))
    metrics.append((
        "decision_layer_interventions_total", "counter",
        "Total remediation actions executed by the planner", interventions,
    ))
    escalations = getattr(rm, "escalations", 0)
    metrics.append((
        "decision_layer_remediation_exhausted_total", "counter",
        "Remediation escalations: at-risk pods where the planner found no "
        "beneficial action (e.g. deterministic crash-loops)", escalations,
    ))
    metrics.append((
        "decision_layer_mean_p_failed_reduction", "gauge",
        "Mean predicted P(FAILED) reduction per executed intervention",
        round(mean_red, 4),
    ))
    metrics.append((
        "decision_layer_degraded", "gauge",
        "1 if the planner is in degraded (calibration-drift rollback) mode",
        1 if getattr(rm, "degraded", False) else 0,
    ))

    monitor = getattr(rm, "calibration_monitor", None)
    if monitor is not None:
        metrics.append((
            "decision_layer_calibration_brier", "gauge",
            "Calibration Brier score over the rolling window", round(monitor.brier(), 4),
        ))
        metrics.append((
            "decision_layer_calibration_drift", "gauge",
            "1 if calibration drift is detected", 1 if monitor.drift_detected() else 0,
        ))
        metrics.append((
            "decision_layer_calibration_samples", "gauge",
            "Number of calibration samples in the rolling window", monitor.n(),
        ))

    if predictor is not None:
        preds = getattr(predictor, "_predictions", None)
        if preds:
            confs = [getattr(p, "confidence", None) for p in preds.values()]
            confs = [c for c in confs if c is not None]
            if confs:
                metrics.append((
                    "decision_layer_mean_confidence", "gauge",
                    "Mean predictor confidence over active predictions",
                    round(sum(c for c in confs if c is not None) / len(confs), 4),
                ))
    return metrics


def render_prometheus(metrics: List[Metric]) -> str:
    """Render decision-layer metrics in Prometheus text exposition format."""
    lines = []
    for name_suffix, mtype, help_text, value in metrics:
        name = _PREFIX + name_suffix
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {mtype}")
        lines.append(f"{name} {value}")
    return "\n".join(lines)
