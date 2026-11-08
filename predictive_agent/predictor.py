"""Prediction engine combining Kalman trends, Markov state, and Bayesian risk."""

from dataclasses import dataclass
from typing import Optional

from predictive_agent.risk import calculate_risk


@dataclass
class PredictionResult:
    """Prediction result for a single pod."""
    pod_key: str
    risk_score: float
    ttf_minutes: Optional[int]
    confidence: float
    markov_state: str
    memory_trend: float
    cpu_trend: float
    memory_pct: float
    cpu_pct: float
    anomaly_score: float = 0.0
    blast_radius: int = 0


class Predictor:
    """Prediction engine combining multiple signals."""

    def __init__(self, risk_threshold: float = 0.5):
        """Initialize predictor.

        Args:
            risk_threshold: Risk score threshold for "at risk" classification.
        """
        self.risk_threshold = risk_threshold
        self._predictions: dict[str, PredictionResult] = {}

    def predict(
        self,
        pod_key: str,
        memory_pct: float,
        memory_trend_mib_per_min: float,
        memory_limit_mib: int,
        memory_mib: int,
        cpu_pct: float,
        restart_rate_per_hr: float,
        log_error_rate_per_min: float,
        node_memory_pressure: bool,
        node_disk_pressure: bool,
        markov_state: str,
        markov_p_critical: float,
        markov_p_failed: float,
        cpu_trend_m_per_min: float = 0.0,
        memory_anomaly_score: float = 0.0,
        cpu_anomaly_score: float = 0.0,
        blast_radius: int = 0,
        pod_phase: str = "Running",
        container_ready: bool = True,
        wait_state: Optional[str] = None,
        terminated: bool = False,
        terminated_reason: Optional[str] = None,
        pod_scheduled: bool = True,
    ) -> PredictionResult:
        """Predict pod health and time-to-failure.

        Args:
            pod_key: Unique pod identifier (e.g., "namespace/pod-name")
            memory_pct: Current memory usage percentage
            memory_trend_mib_per_min: Memory growth rate in MiB/min
            memory_limit_mib: Memory limit in MiB
            memory_mib: Current memory usage in MiB
            cpu_pct: Current CPU usage percentage
            restart_rate_per_hr: Pod restarts per hour
            log_error_rate_per_min: Log error count per minute
            node_memory_pressure: Whether node is under memory pressure
            node_disk_pressure: Whether node is under disk pressure
            markov_state: Current Markov chain state
            markov_p_critical: Probability of transitioning to CRITICAL
            markov_p_failed: Probability of transitioning to FAILED
            cpu_trend_m_per_min: CPU trend in millicores/min (from Kalman filter)
            memory_anomaly_score: Memory anomaly score (0=normal, >3=anomalous)
            cpu_anomaly_score: CPU anomaly score (0=normal, >3=anomalous)
            blast_radius: Impact radius from knowledge graph (0=not connected)
            pod_phase: Pod phase (Running, Pending, Failed, Succeeded, Unknown)
            container_ready: Whether the main container is ready
            wait_state: Container waiting state reason (e.g. CrashLoopBackOff)
            terminated: Whether the container was terminated
            terminated_reason: Termination reason (e.g. OOMKilled, Error)
            pod_scheduled: Whether the pod has been scheduled

        Returns:
            PredictionResult with risk score, TTF, confidence, and trends.
        """
        # Calculate Bayesian risk score
        pod_metrics = {
            "memory_pct": memory_pct,
            "memory_trend_mib_per_min": memory_trend_mib_per_min,
            "cpu_pct": cpu_pct,
            "restart_rate_per_hr": restart_rate_per_hr,
            "log_error_rate_per_min": log_error_rate_per_min,
            "node_memory_pressure": node_memory_pressure,
            "node_disk_pressure": node_disk_pressure,
            "memory_limit_mib": memory_limit_mib,
            "memory_mib": memory_mib,
            "cpu_trend_m_per_min": cpu_trend_m_per_min,
            "memory_anomaly_score": memory_anomaly_score,
            "cpu_anomaly_score": cpu_anomaly_score,
            "blast_radius": blast_radius,
            "pod_phase": pod_phase,
            "container_ready": container_ready,
            "wait_state": wait_state,
            "terminated": terminated,
            "terminated_reason": terminated_reason,
            "pod_scheduled": pod_scheduled,
        }

        risk_score = calculate_risk(
            pod_metrics=pod_metrics,
            markov_state=markov_state,
            markov_p_critical=markov_p_critical,
            markov_p_failed=markov_p_failed,
        )

        # Calculate time-to-failure (minutes until memory threshold)
        ttf_minutes = None
        if memory_limit_mib > 0 and memory_trend_mib_per_min > 0:
            remaining = memory_limit_mib - memory_mib
            if remaining > 0:
                # Guard against near-zero trends that produce infinity/overflow
                if memory_trend_mib_per_min < 1e-10:
                    ttf_minutes = None  # Trend too small to be meaningful
                else:
                    raw_ttf = remaining / memory_trend_mib_per_min
                    # Cap at a sane maximum (30 days) to avoid overflow
                    ttf_minutes = min(int(raw_ttf), 43200) if raw_ttf < 43200 else 43200
            else:
                ttf_minutes = 0

        # Calculate confidence based on trend stability and Markov state
        confidence = self._calculate_confidence(
            memory_trend_mib_per_min,
            markov_state,
            markov_p_critical,
            markov_p_failed,
        )

        # Create result
        result = PredictionResult(
            pod_key=pod_key,
            risk_score=risk_score,
            ttf_minutes=ttf_minutes,
            confidence=confidence,
            markov_state=markov_state,
            memory_trend=memory_trend_mib_per_min,
            cpu_trend=cpu_trend_m_per_min,
            memory_pct=memory_pct,
            cpu_pct=cpu_pct,
            anomaly_score=max(memory_anomaly_score, cpu_anomaly_score),
            blast_radius=blast_radius,
        )

        # Store prediction
        self._predictions[pod_key] = result

        return result

    def _calculate_confidence(
        self,
        memory_trend: float,
        markov_state: str,
        markov_p_critical: float,
        markov_p_failed: float,
    ) -> float:
        """Calculate prediction confidence score.

        Higher confidence when:
        - Trends are stable (low variance)
        - Markov state is stable (low transition probabilities to critical/failed)

        Returns:
            float: Confidence score 0.0 to 1.0
        """
        # Base confidence from Markov state
        state_confidence = {
            "HEALTHY": 0.9,
            "DEGRADED": 0.7,
            "STRESSED": 0.5,
            "CRITICAL": 0.3,
            "FAILED": 0.1,
        }
        confidence = state_confidence.get(markov_state, 0.5)

        # Reduce confidence based on transition probabilities
        confidence *= (1.0 - min(markov_p_critical + markov_p_failed, 1.0))

        # Reduce confidence for high volatility trends
        if abs(memory_trend) > 10:
            confidence *= 0.7
        elif abs(memory_trend) > 5:
            confidence *= 0.85

        return min(max(confidence, 0.0), 1.0)

    def add_prediction(self, pod_key: str, result: PredictionResult) -> None:
        """Add a prediction result directly.

        Args:
            pod_key: Pod identifier
            result: PredictionResult to store
        """
        self._predictions[pod_key] = result

    def get_at_risk(self) -> list[PredictionResult]:
        """Get predictions with risk score above threshold.

        Returns:
            List of PredictionResult with risk_score >= risk_threshold
        """
        return [
            result
            for result in self._predictions.values()
            if result.risk_score >= self.risk_threshold
        ]
