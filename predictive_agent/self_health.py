"""Self-health alerting for the predictive-agent's OWN failure modes.

The agent already emails humans about workload issues (high risk, remediation
taken, autonomy events) — but its own failure classes were only ever logged:

  * state persistence errors (PVC permission/disk problems — the 2026-11-06
    incident hammered ``Failed to persist state`` every cycle for hours with
    zero notifications),
  * sustained collector degradation (kubectl failures -> /ready 503, which
    does NOT restart the pod because liveness probes /healthz),
  * LLM backend unreachable (analysis silently skipped).

``SelfHealthAlerter`` turns any of these into a rate-limited email: at most
one mail per ``kind`` per ``cooldown_s`` (default 1 hour), thread-safe
(reconcile runs in a background thread), and it never raises — an alerting
failure must not break the reconcile loop.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Sentinel used as the "never notified" timestamp.
_NEVER = float("-inf")

# Default cooldown: one mail per issue kind per hour. Enough to notice,
# not enough to spam when a condition persists for hours.
DEFAULT_COOLDOWN_S = 3600.0


class SelfHealthAlerter:
    """Emit at most one email per issue-kind per cooldown window."""

    def __init__(
        self,
        notify_fn: Optional[Callable[[str, str], bool]] = None,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
        clock: Callable[[], float] = time.monotonic,
    ):
        """Create an alerter.

        Args:
            notify_fn: ``subject, body -> bool`` sender (e.g. bound
                ``EmailNotifier.send``). ``None`` = no-op alerter (alerts are
                rate-limited but not sent) — keeps tests and dry-runs safe.
            cooldown_s: minimum seconds between two mails of the same kind.
            clock: injectable monotonic clock for tests.
        """
        self._notify = notify_fn
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._last: dict = {}
        self._lock = threading.Lock()

    def report(self, kind: str, subject: str, detail: str = "") -> bool:
        """Report a self-health issue; True when a mail was actually sent.

        The first report of a ``kind`` sends immediately; repeats inside the
        cooldown window are swallowed (return False) so a condition that
        fires every 60 s reconcile cycle produces one mail per hour, not one
        per cycle.
        """
        now = self._clock()
        with self._lock:
            if (now - self._last.get(kind, _NEVER)) < self.cooldown_s:
                return False
            self._last[kind] = now
        if self._notify is None:
            logger.warning(
                "self-health[%s] (no notifier wired): %s %s", kind, subject, detail
            )
            return False
        try:
            body = "\n\n".join(
                part for part in (subject, detail, _footer()) if part
            )
            return bool(self._notify(f"[self-health:{kind}] {subject}", body))
        except Exception as e:  # noqa: BLE001 — alerting must never crash reconcile
            logger.error("self-health[%s] notify failed: %s", kind, e)
            return False

    def reset(self) -> None:
        """Clear cooldown state (tests; and after a successful recovery)."""
        with self._lock:
            self._last.clear()


def _footer() -> str:
    """Identity footer so mails are attributable without log access."""
    name = os.environ.get("OPERATOR_NAME", "predictive-agent")
    version = os.environ.get("OPERATOR_VERSION", "?")
    pod = os.environ.get("POD_NAME", "-")
    ns = os.environ.get("POD_NAMESPACE", "-")
    return f"{name} v{version} — pod {ns}/{pod}"


class DegradedCycleTracker:
    """Track consecutive degraded reconcile cycles; fire once per streak.

    The collector flips to DEGRADED on transient kubectl/API hiccups — a
    single cycle is noise, a streak of ``threshold`` cycles is an issue worth
    one email. The streak resets on the first healthy cycle.
    """

    def __init__(self, alerter: SelfHealthAlerter, threshold: int = 5):
        self._alerter = alerter
        self.threshold = threshold
        self._consecutive = 0

    def cycle(self, collector_state: str) -> bool:
        """Feed one cycle's collector state; True when an alert was sent."""
        if collector_state == "HEALTHY":
            self._consecutive = 0
            return False
        self._consecutive += 1
        if self._consecutive == self.threshold:
            return self._alerter.report(
                "collector",
                f"collector DEGRADED for {self._consecutive} consecutive cycles",
                "Kubernetes API/kubectl collection has been failing repeatedly. "
                "/ready returns 503 (liveness /healthz is unaffected, so the pod "
                "is NOT restarted). Last-known pod states are being reused for "
                "predictions — risk scores are stale until collection recovers.",
            )
        return False

    @property
    def consecutive(self) -> int:
        return self._consecutive
