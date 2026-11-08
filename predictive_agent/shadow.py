"""Shadow-mode validation harness for the POMDP decision layer.

The learned planner can run *observe-only* next to the live remediation path so
it can be validated in production before any of its actions are trusted. The
harness:

  * computes what the planner would do (without executing anything),
  * compares that to the actions the live path actually executed, and logs
    agreements / disagreements,
  * records predicted ``P(FAILED)`` vs the actual outcome for calibration.

It never mutates cluster state or the decision model. It is wired into
``main.reconcile`` through the optional ``_shadow_harness`` global, so enabling
it is purely additive (no behavior change when unset).
"""

import json
import logging
import threading
import time
from collections.abc import Sequence
from typing import Optional

logger = logging.getLogger(__name__)


class ShadowModeHarness:
    """Run the planner alongside production and log disagreements + calibration."""

    def __init__(
        self,
        remediation_manager,
        max_recent: int = 200,
        log_path: Optional[str] = None,
    ):
        self.rm = remediation_manager
        self.max_recent = max_recent
        self.log_path = log_path
        self.recent: list = []
        self.calibrations: list = []
        self._lock = threading.Lock()
        self.agreement_count = 0
        self.disagreement_count = 0
        self.over_count = 0
        self.under_count = 0

    # ─── planner recommendation (no execution) ────────────────────────────
    def recommend(self, state, horizon=None) -> list:
        """Return the actions the planner would take from ``state`` (plan only)."""
        plan = self.rm.plan_remediation(state, horizon)
        return [s["action"] for s in plan["steps"] if s["action"] != "none"]

    # ─── disagreement logging ─────────────────────────────────────────────
    def observe(
        self,
        pod_key: str,
        state: str,
        executed_actions: Sequence[str],
        *,
        risk_score=None,
        prediction=None,
    ) -> dict:
        """Log the planner's recommendation vs what was actually executed."""
        planned = self.recommend(state)
        agreement = set(planned) == set(executed_actions)
        rec = {
            "ts": time.time(),
            "pod_key": pod_key,
            "state": state,
            "planned": planned,
            "executed": list(executed_actions),
            "agreement": agreement,
            "risk_score": risk_score,
        }
        with self._lock:
            if agreement:
                self.agreement_count += 1
            else:
                self.disagreement_count += 1
            self.recent.append(rec)
            if len(self.recent) > self.max_recent:
                self.recent.pop(0)
        if agreement:
            logger.info(
                "[shadow] %s state=%s planned=%s executed=%s agree=%s",
                pod_key, state, planned, list(executed_actions), agreement,
            )
        else:
            logger.warning(
                "[shadow] DISAGREEMENT %s state=%s planned=%s executed=%s",
                pod_key, state, planned, list(executed_actions),
            )
        self._write(rec)
        return rec

    # ─── calibration logging ──────────────────────────────────────────────
    def calibration(self, pod_key: str, predicted_p_failed: float, actual_failed: bool) -> dict:
        """Log predicted P(FAILED) vs the realized outcome for a pod."""
        over = predicted_p_failed > 0.5 and not actual_failed
        under = predicted_p_failed <= 0.5 and actual_failed
        rec = {
            "ts": time.time(),
            "pod_key": pod_key,
            "predicted_p_failed": float(predicted_p_failed),
            "actual_failed": bool(actual_failed),
            "over": over,
            "under": under,
        }
        with self._lock:
            if over:
                self.over_count += 1
            if under:
                self.under_count += 1
            self.calibrations.append(rec)
            if len(self.calibrations) > self.max_recent:
                self.calibrations.pop(0)
        logger.info(
            "[shadow] calib %s pred=%.3f actual=%s",
            pod_key, predicted_p_failed, actual_failed,
        )
        self._write(rec)
        return rec

    # ─── reporting ────────────────────────────────────────────────────────
    def summary(self) -> dict:
        with self._lock:
            return {
                "enabled": True,
                "agreements": self.agreement_count,
                "disagreements": self.disagreement_count,
                "over_predictions": self.over_count,
                "under_predictions": self.under_count,
                "disagreement_rate": (
                    self.disagreement_count
                    / (self.agreement_count + self.disagreement_count)
                    if (self.agreement_count + self.disagreement_count) > 0
                    else 0.0
                ),
                "recent": list(self.recent[-10:]),
                "calibrations_recent": list(self.calibrations[-10:]),
            }

    def _write(self, rec: dict) -> None:
        if self.log_path:
            try:
                with open(self.log_path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            except Exception:
                pass
