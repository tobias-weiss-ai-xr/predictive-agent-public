"""Calibration monitoring for the decision layer.

Tracks the predictor's predicted ``P(FAILED)`` against the realized outcome
(failure or not) over a rolling window and flags *drift* — when the model's
confidence no longer matches reality. The reconcile loop feeds it through the
learning path (``record_observations`` pairs the pending prediction with the
observed state), so calibration is measured on real production outcomes, not
synthetic backtests.

Drift is the trigger for the safety guardrail / circuit-breaker rollback
(see :mod:`predictive_agent.remediator` and :mod:`predictive_agent.main`):
when detected, the planner can fall back to the conservative reactive path and
raise an operator alert until calibration recovers.
"""

import threading
from typing import List, Tuple

logger = __import__("logging").getLogger(__name__)

Pair = Tuple[float, bool]  # (predicted_p_failed, actual_failed)


class CalibrationMonitor:
    """Rolling calibration tracker with drift detection."""

    def __init__(
        self,
        window: int = 100,
        brier_alert: float = 0.25,
        over_rate_alert: float = 0.5,
        min_samples: int = 20,
    ):
        self.window = window
        self.brier_alert = brier_alert
        self.over_rate_alert = over_rate_alert
        self.min_samples = min_samples
        self._pairs: List[Pair] = []
        self._lock = threading.Lock()

    # ─── ingestion ────────────────────────────────────────────────────────
    def record(self, predicted_p_failed: float, actual_failed: bool) -> None:
        with self._lock:
            self._pairs.append((float(predicted_p_failed), bool(actual_failed)))
            if len(self._pairs) > self.window:
                self._pairs.pop(0)

    # ─── metrics ───────────────────────────────────────────────────────────
    def _window(self) -> List[Pair]:
        with self._lock:
            return list(self._pairs)

    def n(self) -> int:
        return len(self._window())

    def brier(self) -> float:
        """Brier score over the window: mean((predicted - actual)^2)."""
        pairs = self._window()
        if not pairs:
            return 0.0
        return sum((p - (1.0 if a else 0.0)) ** 2 for p, a in pairs) / len(pairs)

    def over_rate(self) -> float:
        """Fraction of samples where we predicted failure but none occurred."""
        pairs = self._window()
        if not pairs:
            return 0.0
        over = sum(1 for p, a in pairs if p > 0.5 and not a)
        return over / len(pairs)

    def under_rate(self) -> float:
        """Fraction of samples where we predicted safety but failure occurred."""
        pairs = self._window()
        if not pairs:
            return 0.0
        under = sum(1 for p, a in pairs if p <= 0.5 and a)
        return under / len(pairs)

    def drift_detected(self) -> bool:
        pairs = self._window()
        if len(pairs) < self.min_samples:
            return False
        if self.brier() > self.brier_alert:
            return True
        if self.over_rate() > self.over_rate_alert:
            return True
        return False

    def status(self) -> str:
        if self.drift_detected():
            return "drifted"
        if self.n() >= self.min_samples:
            return "healthy"
        return "warmup"

    # ─── reporting ──────────────────────────────────────────────────────────
    def summary(self) -> dict:
        return {
            "enabled": True,
            "n": self.n(),
            "window": self.window,
            "brier": self.brier(),
            "over_rate": self.over_rate(),
            "under_rate": self.under_rate(),
            "brier_alert": self.brier_alert,
            "over_rate_alert": self.over_rate_alert,
            "drift_detected": self.drift_detected(),
            "status": self.status(),
        }

    def reset(self) -> None:
        with self._lock:
            self._pairs.clear()
