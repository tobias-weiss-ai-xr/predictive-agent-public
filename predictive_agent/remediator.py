"""Remediation framework: RemediationAction ABC, RemediationManager, SafetyPolicy.

This module provides the core abstractions for autonomous self-healing:
- RemediationAction: base class for all remediation actions
- ActionContext: context passed to actions (namespace, dry_run, audit_log)
- RemediationResult: result of a remediation action
- SafetyPolicy: rate limiting, cooldowns, protected namespaces
- RemediationManager: registers actions, checks safety, executes with audit trail
- Binary observability: every remediation decision logged as acted, skipped, or failed
- Prometheus metric: predictive_remediation_total{action, outcome}
- P3 Idempotent remediation: per-action idempotent flag + idempotency key
  (namespace/target/action/window); re-runs within REMEDIATION_DEDUP_WINDOW are
  deduplicated (logged '[op] dedup: action already executed') so a crash
  mid-remediation + reconcile loop cannot double-execute an action.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Optional

from predictive_agent import config
from predictive_agent.pomdp import FAILURE_COST, ActionAwareTransitionModel, MarkovChain, Planner

if TYPE_CHECKING:
    from predictive_agent.calibration import CalibrationMonitor
    from predictive_agent.rl import QLearningPolicy

logger = logging.getLogger(__name__)


class RemediationOutcome(Enum):
    """Outcome of a remediation decision for binary observability (P2 + P6)."""
    ACTED = "acted"
    SKIPPED = "skipped"
    FAILED = "failed"
    # P6: action is at/above REMEDIATION_REVIEW_TIER and awaits human approval.
    REVIEW_REQUIRED = "review_required"
    # Closed-loop verification: post-action verification outcomes.
    VERIFIED = "verified"
    HELD = "held"
    REVERTED = "reverted"
    ESCALATED = "escalated"
    CANARY = "canary"
    WIDENED = "widened"
    # Closed-loop: an action whose retry requires operator approval after a
    # failed verification (irreversible actions).
    APPROVAL_REQUIRED = "approval_required"


# Prometheus metrics storage for binary observability
_remediation_metrics: dict = {}


def record_remediation_metric(action: str, outcome: RemediationOutcome) -> None:
    """Record a remediation decision for Prometheus metric predictive_remediation_total.
    
    This is called for every remediation decision to ensure binary observability.
    """
    key = (action, outcome.value)
    _remediation_metrics[key] = _remediation_metrics.get(key, 0) + 1


def get_remediation_metrics() -> dict:
    """Return all remediation metrics as a dict."""
    return _remediation_metrics.copy()


def reset_remediation_metrics() -> None:
    """Reset remediation metrics (useful for testing)."""
    global _remediation_metrics
    _remediation_metrics = {}


def _utc_now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _now() -> float:
    """Wall-clock seconds for idempotency dedup windows.

    Kept as a module-level helper so tests can monkeypatch it without
    disturbing other ``time.time()`` users.
    """
    return time.time()


@dataclass
class ActionContext:
    """Context passed to every remediation action."""

    namespace: str = "opendesk-predictive-agent"
    dry_run: bool = True
    audit_log: list = field(default_factory=list)
    cooldowns: dict = field(default_factory=dict)  # target -> last_action_timestamp
    rate_limiter: dict = field(default_factory=dict)  # window -> list of timestamps
    remediation_log: list = field(default_factory=list)  # structured remediation audit log
    node_name: str = ""  # Node name for node-cordon actions
    pod_phase: str = "Running"  # Current pod phase
    cpu_trend: float = 0.0  # CPU trend (millicores/min)
    mem_trend: float = 0.0  # Memory trend (MiB/min)
    # Closed loop: canary widening context. ``canary_mode`` is None for a
    # normal execution, "canary" for the bounded first leg, "widen" for the
    # post-verification full leg. Actions that support canary read
    # ``canary_fraction`` (clamped) and ``canary_replicas`` (target scale).
    canary_mode: Optional[str] = None
    canary_fraction: float = 0.0
    canary_replicas: int = 1


@dataclass
class RemediationResult:
    """Result of a remediation action with outcome for binary observability."""

    action: str
    target: str
    success: bool
    dry_run: bool
    message: str
    timestamp: str = field(default_factory=_utc_now)
    command: str = ""
    outcome: str = ""  # acted, skipped, or failed
    # P3 idempotent remediation: the desired state this action converged to
    # (e.g. scale -> {"replicas": N}, tune_resources -> resource values), so a
    # re-run within the dedup window converges rather than stacking.
    desired_state: Optional[dict] = None
    # P3 idempotent remediation: idempotency key (namespace/target/action/
    # timestamp-window) this execution belongs to, set by the RemediationManager.
    idempotency_key: str = ""

    def get_outcome(self) -> str:
        """Determine outcome based on success and message."""
        if self.outcome:
            return self.outcome
        if "skipped" in self.message.lower():
            return RemediationOutcome.SKIPPED.value
        if self.success:
            return RemediationOutcome.ACTED.value
        return RemediationOutcome.FAILED.value


class SafetyPolicy:
    """Safety policy for remediation actions.

    Enforces rate limits, cooldowns, and protected namespaces.
    """

    def __init__(
        self,
        max_per_minute: int = 5,
        max_per_hour: int = 50,
        cooldown_seconds: int = 300,
        min_healthy_nodes: int = 2,
        protected_namespaces: Optional[set] = None,
    ):
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        self.cooldown_seconds = cooldown_seconds
        self.min_healthy_nodes = min_healthy_nodes
        self.protected_namespaces = protected_namespaces or {
            "kube-system",
            "opendesk-predictive-agent",
        }

    def can_execute(
        self,
        target: str,
        action_type: str,
        context: ActionContext,
        namespace: str = "",
    ) -> tuple[bool, str]:
        """Check if an action can be executed under the safety policy.

        Returns (can_execute, reason).
        """
        now = time.time()

        # Check protected namespaces
        if namespace and namespace in self.protected_namespaces:
            return False, f"namespace '{namespace}' is protected"

        # Check cooldown
        cooldown_key = f"{action_type}:{target}"
        last_action = context.cooldowns.get(cooldown_key)
        if last_action is not None:
            elapsed = now - last_action
            if elapsed < self.cooldown_seconds:
                remaining = int(self.cooldown_seconds - elapsed)
                return False, f"cooldown active ({remaining}s remaining)"

        # Check per-minute rate limit
        minute_key = "minute"
        minute_actions = [t for t in context.rate_limiter.get(minute_key, []) if now - t < 60]
        if len(minute_actions) >= self.max_per_minute:
            return False, "per-minute rate limit exceeded"

        # Check per-hour rate limit
        hour_key = "hour"
        hour_actions = [t for t in context.rate_limiter.get(hour_key, []) if now - t < 3600]
        if len(hour_actions) >= self.max_per_hour:
            return False, "per-hour rate limit exceeded"

        return True, "ok"

    def record_action(
        self,
        target: str,
        action_type: str,
        context: ActionContext,
    ) -> None:
        """Record an action in the rate limiter and cooldown tracker."""
        now = time.time()
        cooldown_key = f"{action_type}:{target}"
        context.cooldowns[cooldown_key] = now

        # Update rate limiter
        minute_key = "minute"
        context.rate_limiter.setdefault(minute_key, [])
        context.rate_limiter[minute_key] = [
            t for t in context.rate_limiter[minute_key] if now - t < 60
        ] + [now]

        hour_key = "hour"
        context.rate_limiter.setdefault(hour_key, [])
        context.rate_limiter[hour_key] = [
            t for t in context.rate_limiter[hour_key] if now - t < 3600
        ] + [now]


class RemediationAction(ABC):
    """Base class for all remediation actions."""

    name: str = "base"
    # P3 idempotent remediation: True when re-running this action after a crash
    # is safe (it converges to the same state) — e.g. container_restart,
    # pod_restart, node_cordon. False when a re-run could double-scale /
    # double-restart (scale, rollout_restart, tune_resources, right_size). The
    # RemediationManager still dedups every action within REMEDIATION_DEDUP_WINDOW;
    # the flag records the action's re-run safety and is surfaced in logs.
    idempotent: bool = True
    # P6 bounded autonomy: risk tier of this action — "safe" (restart-class,
    # converges to the same state), "moderate" (changes desired state:
    # scale / tune / right-size), "destructive" (cluster-wide impact:
    # node_cordon). Actions at/above REMEDIATION_REVIEW_TIER require human
    # approval before executing.
    risk_tier: str = "moderate"
    # Closed loop: reversible actions declare an undo. When True, the action
    # MUST override ``revert`` to restore the recorded prior state. On failed
    # verification the manager reverts automatically (if enabled);
    # non-reversible actions under the same condition require operator
    # approval before any further attempt.
    reversible: bool = False
    # Closed loop: canary widening. >0 (and target scale > 1) executes the
    # action in two legs — a first leg bounded to ``canary_fraction`` of the
    # workload, then, only after that leg verifies, a widen leg. The value is
    # clamped to config.CANARY_MAX_FRACTION per leg. 0.0 disables canary.
    canary_fraction: float = 0.0
    # Closed loop: signal names compared at verification time (any subset of
    # state, memory_pct, cpu_pct, restart_count, log_errors, node_pressure).
    # Verification requires VERIFY_MIN_SIGNALS of them to show sustained
    # improvement.
    resolution_signals: tuple = ("state", "memory_pct", "cpu_pct", "restart_count", "log_errors", "node_pressure")

    def revert(self, target: str, prev_state: Optional[dict], context: ActionContext) -> Optional[RemediationResult]:
        """Undo this action, restoring the state captured at execution time.

        Subclasses that declare ``reversible=True`` override this; the default
        returns ``None`` (not reversible at runtime). ``prev_state`` is the
        snapshot (signal values + ``desired_state``) recorded when the action
        executed, so a revert can restore the exact prior state.
        """
        return None

    @abstractmethod
    def should_execute(self, pod_state, prediction, risk_score: float) -> bool:
        """Check if this action should be taken for the given pod.

        Args:
            pod_state: PodTracker with current pod state
            prediction: PredictionResult with trend predictions
            risk_score: Bayesian risk score (0-1)

        Returns:
            True if this action should be executed
        """
        ...

    @abstractmethod
    def execute(self, target: str, context: ActionContext) -> RemediationResult:
        """Execute the remediation action.

        Args:
            target: Target resource name (pod name, node name, deployment name)
            context: ActionContext with namespace, dry_run, audit_log

        Returns:
            RemediationResult with action outcome
        """
        ...


class RemediationManager:
    """Manages remediation actions, safety policy, and audit trail.

    Registers actions, checks safety policy before execution,
    and maintains an audit trail of all actions taken.
    """

    def __init__(
        self,
        dry_run: bool = True,
        risk_threshold: float = 0.7,
        safety_policy: Optional[SafetyPolicy] = None,
        decision_model: Optional[ActionAwareTransitionModel] = None,
        action_costs: Optional[dict] = None,
        failure_cost: Optional[float] = None,
        dedup_window: Optional[int] = None,
        review_tier: Optional[str] = None,
        review_ttl_s: Optional[float] = None,
        action_timeout_s: Optional[float] = None,
    ):
        self.dry_run = dry_run
        self.risk_threshold = risk_threshold
        self.safety_policy = safety_policy or SafetyPolicy()
        self.actions: list[RemediationAction] = []
        self.audit_trail: list[RemediationResult] = []
        self.context = ActionContext(dry_run=dry_run)
        # P3 idempotent remediation: dedup window (seconds) and recent-execution
        # history keyed by idempotency key (namespace/target/action/window). A
        # re-run within the window is skipped; expired entries are pruned lazily.
        self.dedup_window = int(
            dedup_window if dedup_window is not None else config.REMEDIATION_DEDUP_WINDOW
        )
        self._dedup_history: dict = {}
        self.deduped_count = 0
        # Decision-theoretic cost model (calibratable from incident data; #5).
        self.action_costs = action_costs
        self.failure_cost = failure_cost if failure_cost is not None else FAILURE_COST
        self.cost_source = "defaults" if (action_costs is None and failure_cost is None) else "explicit"
        # Decision-layer observability (#7): cumulative intervention counters.
        self.interventions_executed = 0
        self.cumulative_p_failed_reduction = 0.0
        # Crash-loop escalation (#watch): number of at-risk pods for which the
        # planner found no beneficial action and we escalated to a human.
        self.escalations = 0
        # target -> consecutive at-risk cycles with only 'none' actions decided.
        self._futile_cycles: dict = {}
        # Persistent POMDP decision model; reused across reconciles so that
        # action-conditioned transitions learned via feed_transition accumulate.
        self.decision_model = decision_model or ActionAwareTransitionModel()
        self.action_costs = action_costs
        # Pending feedback: pod_key -> list of (from_state, action,
        # predicted_p_failed) executed in the current cycle, awaiting the next
        # observed state to learn from (and to feed the calibration monitor).
        self._pending_feedback: dict = {}
        # Optional calibration monitor: records predicted P(FAILED) vs realised
        # outcome so drift can be detected in production. None = disabled.
        self.calibration_monitor: Optional[CalibrationMonitor] = None
        # Safety flag: when True the planner falls back to the conservative
        # reactive path (set by reconcile on detected calibration drift).
        self.degraded = False
        # Optional model-free Q-learning policy (#8); learns from the same
        # real outcomes as the transition model via record_observations.
        self.rl_policy: Optional[QLearningPolicy] = None
        # P6 bounded autonomy: review gate state. review_tier actions at or
        # above REMEDIATION_REVIEW_TIER never execute until a human approves
        # their (namespace, target, action) signature; approvals expire after
        # REMEDIATION_REVIEW_TTL_S. Per-action execution timeout in seconds.
        self.review_tier = (
            review_tier if review_tier is not None else config.REMEDIATION_REVIEW_TIER
        )
        self.review_ttl_s = (
            review_ttl_s if review_ttl_s is not None else config.REMEDIATION_REVIEW_TTL_S
        )
        self.action_timeout_s = (
            action_timeout_s
            if action_timeout_s is not None
            else config.REMEDIATION_TIMEOUT_S
        )
        self.review_queue: dict = {}   # id -> review item (pending/approved)
        self._review_seq = 0
        self._approved: dict = {}      # "ns/target/action" -> approved_at
        self.reviews_deferred = 0

        # ── Closed-loop verification (verify-after-act) ────────────────────
        # Per-idempotency-key verification state: "pending" -> "verified" |
        # "held". Held keys are not re-executed until VERIFY_HOLD_S elapses (or
        # the operator approves the retry via POST /review).
        self.verify_state: dict = {}
        # Consecutive failed verifications per key (drives escalation).
        self._verify_failures: dict = {}
        # Canary legs awaiting verification before widening (key base -> "awaiting"|"widened").
        self._canary_state: dict = {}
        # Outcome alerts queued by verification/hold/revert/canary events; the
        # reconcile loop drains them to the notifier (per-pod cooldowns).
        self.outcome_alerts: list = []

    def register_action(self, action: RemediationAction) -> None:
        """Register a remediation action."""
        self.actions.append(action)
        logger.info("Registered remediation action: %s", action.name)

    def idempotency_key(self, namespace: str, target: str, action_name: str) -> str:
        """Build the idempotency key for an action execution.

        Format: ``namespace/target/action/window`` (namespace/pod/action/
        timestamp-window from the P3 roadmap). The dedup window is embedded so a
        config change to REMEDIATION_DEDUP_WINDOW invalidates stale history
        instead of silently suppressing future runs.
        """
        return (
            f"{(namespace or '-')}/{(target or '-')}/{action_name}/w{self.dedup_window}"
        )

    def _record_execution(self, namespace: str, target: str, action: RemediationAction, result: RemediationResult, snapshot: Optional[dict] = None, leg_key: Optional[str] = None) -> None:
        """Record a successful execution in the dedup history (P3).

        Stores the timestamp, message/command, desired state (for scale /
        tune_resources so a re-run within the window converges), the signal
        snapshot at execution time (for post-action verification), and marks
        the entry ``verify=pending``. ``leg_key`` overrides the idempotency
        key, used for canary leg keys (``.../canary``, ``.../widen``) so each
        leg dedups and verifies independently.
        """
        key = leg_key or self.idempotency_key(namespace, target, action.name)
        self._dedup_history[key] = {
            "timestamp": _now(),
            "message": result.message,
            "command": result.command,
            "desired_state": getattr(result, "desired_state", None) or None,
            "idempotent": bool(getattr(action, "idempotent", True)),
            # Closed loop: signal snapshot + verification lifecycle.
            "snapshot": snapshot or {},
            "verify": "pending",
            "recorded_at": _now(),
            "sig": self._review_signature(namespace, target, action.name),
            "leg": ("canary" if leg_key and leg_key.endswith("/canary")
                    else "widen" if leg_key and leg_key.endswith("/widen")
                    else "full"),
        }
        result.idempotency_key = key
        self.verify_state[key] = {"status": "pending", "at": _utc_now()}

    def _hold_still_active(self, rec: dict, now: float) -> bool:
        """True when a verification hold on this record is still in force."""
        return rec.get("verify") == "held" and now < rec.get("held_until", 0)

    def _dedup_skip_result(self, namespace: str, target: str, action: RemediationAction) -> Optional[RemediationResult]:
        """Return a dedup-skip RemediationResult if this action was already
        executed for namespace/target within REMEDIATION_DEDUP_WINDOW.

        Returns None when the action may proceed (no prior execution, or the
        prior execution is outside the window / was a failure). Expired history
        entries are pruned lazily so the map stays bounded.
        """
        key = self.idempotency_key(namespace, target, action.name)
        rec = self._dedup_history.get(key)
        if rec is None:
            return None
        now = _now()
        if now - rec["timestamp"] >= self.dedup_window:
            if self._hold_still_active(rec, now):
                # Dedup window elapsed but a verification hold is still active:
                # keep the entry so the dispatch-time hold gate can block the
                # retry until VERIFY_HOLD_S elapses (or the operator approves).
                return None
            self._dedup_history.pop(key, None)
            self.verify_state.pop(key, None)
            # NOTE: the consecutive-failure counter is deliberately kept —
            # re-arming does NOT break the escalation chain; only a successful
            # verification resets it (see verify_remediations).
            return None
        detail = ""
        if rec.get("desired_state"):
            detail = f", desired_state={rec['desired_state']}"
        self.deduped_count += 1
        logger.warning(
            "[op] dedup: action already executed (idempotency_key=%s%s, last run %ss ago)",
            key, detail, int(now - rec["timestamp"]),
        )
        return RemediationResult(
            action=action.name,
            target=target,
            success=False,
            dry_run=self.dry_run,
            message=(
                f"[op] dedup: action already executed "
                f"(idempotency_key={key}{detail}, last run {int(now - rec['timestamp'])}s ago)"
            ),
            outcome=RemediationOutcome.SKIPPED.value,
            idempotency_key=key,
        )

    # ── Closed-loop verification (verify-after-act) ──────────────────────

    _STATE_ORDER = {"HEALTHY": 0, "DEGRADED": 1, "RECOVERED": 1, "STRESSED": 2, "CRITICAL": 3, "FAILED": 4}
    _SIGNAL_NAMES = ("state", "memory_pct", "cpu_pct", "restart_count", "log_errors", "node_pressure")

    @staticmethod
    def _signal_snapshot(pod_state) -> dict:
        """Capture the verification signals from a pod/target at a point in time.

        Defensive against test doubles and partial state objects: non-numeric
        values coerce to 0 and non-string states to "" — verification then
        treats unmeasurable signals as neutral (never as improvement).
        """
        def _num(value) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        def _int(value) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        state = getattr(pod_state, "state", "")
        pressure = getattr(pod_state, "node_pressure", False)
        return {
            "state": state if isinstance(state, str) else "",
            "memory_pct": _num(getattr(pod_state, "memory_pct", 0.0)),
            "cpu_pct": _num(getattr(pod_state, "cpu_pct", 0.0)),
            "restart_count": _int(getattr(pod_state, "restart_count", 0)),
            "log_errors": _int(getattr(pod_state, "log_errors", 0)),
            "node_pressure": pressure is True,
        }

    @classmethod
    def _signal_improved(cls, name: str, recorded, current) -> Optional[bool]:
        """Return True/False for an improved signal, None if neutral/unmeasurable.

        ``recorded`` is the snapshot at execution time; ``current`` is the
        re-sampled value. A signal that cannot be judged (e.g. an unknown
        state, or a 0% recorded baseline) is neutral and does not count toward
        VERIFY_MIN_SIGNALS.
        """
        if name == "state":
            rec_i = cls._STATE_ORDER.get(recorded or "")
            cur_i = cls._STATE_ORDER.get(current or "")
            if rec_i is None or cur_i is None:
                return None
            if cur_i < rec_i:
                return True
            if cur_i > rec_i:
                return False
            return None  # unchanged state is neutral
        if name == "node_pressure":
            if not recorded:
                return None  # no pressure recorded → neutral
            return not bool(current)
        if recorded in (None, 0) or current is None:
            return None
        # memory_pct / cpu_pct / restart_count / log_errors: lower is better.
        if float(current) < float(recorded):
            return True
        if float(current) > float(recorded):
            return False
        return None

    def _check_improvement(self, rec: dict, pod_state, action=None) -> tuple:
        """Compare the recorded snapshot against current signals.

        Only the action's ``resolution_signals`` are judged (default: all six).
        Returns ``(improved_count, criteria)`` where ``criteria`` describes the
        per-signal verdicts for operator visibility.
        """
        snapshot = rec.get("snapshot") or {}
        current = self._signal_snapshot(pod_state)
        names = tuple(getattr(action, "resolution_signals", ()) or ()) or self._SIGNAL_NAMES
        improved = 0
        criteria = {}
        for name in names:
            if name not in self._SIGNAL_NAMES:
                continue
            verdict = self._signal_improved(name, snapshot.get(name), current.get(name))
            criteria[name] = verdict
            if verdict is True:
                improved += 1
        return improved, criteria

    def verify_remediations(self, pod_state) -> None:
        """Run due post-action verifications for a pod (call once per cycle).

        For every executed action whose entry is ``verify=pending`` and whose
        settling window has elapsed, re-sample the pod's signals and mark the
        remediation verified / held. Idempotent: entries transition at most
        once; not-yet-due entries are no-ops.
        """
        if not config.VERIFY_ENABLED or pod_state is None:
            return
        namespace = getattr(pod_state, "namespace", "") or ""
        target = getattr(pod_state, "name", str(pod_state)) or ""
        for action in list(self.actions):
            base = self.idempotency_key(namespace, target, action.name)
            canary_key, widen_key = self._canary_leg_keys(namespace, target, action)
            # Verify base executions and canary/widen legs (each leg dedups and
            # verifies independently, so the widen gate sees its canary verdict).
            for key in (base, canary_key, widen_key):
                rec = self._dedup_history.get(key)
                if not rec or rec.get("verify") != "pending":
                    continue
                if _now() - rec.get("recorded_at", 0) < config.VERIFY_SETTLING_S:
                    continue  # not due yet → observe-only no-op
                improved, criteria = self._check_improvement(rec, pod_state, action)
                if improved >= config.VERIFY_MIN_SIGNALS:
                    rec["verify"] = "verified"
                    rec["verified_at"] = _utc_now()
                    self._verify_failures.pop(key, None)  # consecutive-failure chain broken
                    self.verify_state[key] = {
                        "status": "verified", "improved": improved,
                        "criteria": criteria, "at": _utc_now(),
                    }
                    self.outcome_alerts.append({
                        "kind": "remediation_verified", "action": action.name,
                        "target": key, "risk_score": None,
                        "details": f"verified: {improved} signals improved after "
                                    f"{config.VERIFY_SETTLING_S:.0f}s settling"
                                    f" (criteria={criteria})",
                    })
                else:
                    self._handle_verify_failure(action, namespace, target, key, rec, criteria)

    def _handle_verify_failure(self, action, namespace: str, target: str, key: str, rec: dict, criteria: dict) -> None:
        """Mark a verification failure: hold + optional revert + escalate."""
        fail_n = self._verify_failures.get(key, 0) + 1
        self._verify_failures[key] = fail_n
        rec["verify"] = "held"
        rec["held_until"] = _now() + config.VERIFY_HOLD_S
        self.verify_state[key] = {
            "status": "held", "failures": fail_n, "criteria": criteria,
            "at": _utc_now(),
        }
        logger.warning(
            "Remediation decision: action=%s target=%s outcome=held "
            "reason=verification failed (%d/%d signals improved, criteria=%s)",
            action.name, target, sum(1 for v in criteria.values() if v is True),
            config.VERIFY_MIN_SIGNALS, criteria,
        )
        kind = "remediation_held"
        reverted = False
        # Revert reversible actions automatically (best practice: armed rollback).
        if config.REMEDITATION_REVERT_ENABLED and getattr(action, "reversible", False):
            revert_res = self._execute_revert(action, namespace, target, rec)
            if revert_res is not None:
                reverted = True
                kind = "remediation_reverted"
        # Escalate via the existing futile-remediation channel (D3): every
        # verification failure accrues the shared futile-cycle counter, and
        # repeated failures surface as the established ``remediation_exhausted``
        # escalation. Irreversible actions additionally require operator
        # approval before any further attempt (D4).
        self.note_remediation_outcome(target, True, ["none"])
        if fail_n >= config.VERIFY_ESCALATE_AFTER:
            kind = "remediation_exhausted"
            if not getattr(action, "reversible", False):
                self._queue_retry_approval(action, namespace, target, key)
        self.outcome_alerts.append({
            "kind": kind, "action": action.name, "target": key,
            "risk_score": None,
            "details": (f"verification failed: {sum(1 for v in criteria.values() if v is True)}"
                         f"/{config.VERIFY_MIN_SIGNALS} signals improved (criteria={criteria})"
                         + ("; reverted" if reverted else "")),
        })

    def _execute_revert(self, action, namespace: str, target: str, rec: dict):
        """Revert a reversible action's recorded prior state (audited)."""
        try:
            snapshot = rec.get("snapshot") or {}
            # Reversible actions restore from the recorded prior state: the
            # signal snapshot plus the recorded desired_state (e.g. scale's
            # prev_replicas) so revert() can undo the exact change.
            prev_state = dict(snapshot)
            if rec.get("desired_state") is not None:
                prev_state["desired_state"] = rec["desired_state"]
            revert_res = action.revert(target, prev_state, self.context)
            if revert_res is None:
                logger.warning(
                    "Remediation decision: action=%s target=%s outcome=reverted "
                    "reason=revert() returned None — nothing to restore",
                    action.name, target,
                )
                return None
            if not revert_res.outcome:
                revert_res.outcome = RemediationOutcome.REVERTED.value
            record_remediation_metric(action.name, RemediationOutcome.REVERTED)
            self.audit_trail.append(revert_res)
            self._add_to_remediation_log(revert_res, namespace)
            logger.warning(
                "Remediation decision: action=%s target=%s outcome=reverted "
                "reason=verification failed after %.0fs settling",
                action.name, target, config.VERIFY_SETTLING_S,
            )
            return revert_res
        except Exception as e:  # noqa: BLE001
            logger.exception(
                "Remediation decision: action=%s target=%s outcome=failed reason=revert error: %s",
                action.name, target, e,
            )
            return None

    def _queue_retry_approval(self, action, namespace: str, target: str, key: str) -> None:
        """Queue an operator-approval item for an irreversible action retry.

        The signature is the same one POST /review approves; approving it also
        clears the hold so the manager may attempt again.
        """
        self._review_seq += 1
        rid = f"rev-{self._review_seq}"
        sig = self._review_signature(namespace, target, action.name)
        self.review_queue[rid] = {
            "id": rid, "signature": sig, "action": action.name,
            "target": target, "namespace": namespace,
            "tier": getattr(action, "risk_tier", "moderate"),
            "risk_score": None, "status": "pending",
            "reason": "verification_failed",
            "created_at": _utc_now(),
        }
        logger.warning(
            "Remediation decision: action=%s target=%s outcome=approval_required "
            "reason=verification failed on irreversible action (approve %s via POST /review)",
            action.name, target, sig,
        )
        result = RemediationResult(
            action=action.name, target=target, success=False,
            dry_run=self.dry_run,
            message=(f"approval_required: verification failed on irreversible action; "
                     f"approve signature {sig} (review id {rid})"),
            outcome=RemediationOutcome.APPROVAL_REQUIRED.value,
        )
        record_remediation_metric(action.name, RemediationOutcome.APPROVAL_REQUIRED)
        self.audit_trail.append(result)
        self._add_to_remediation_log(result, namespace)

    def _hold_skip(self, action, target: str, namespace: str):
        """Return a held SKIP result if this action is on a verification hold.

        Returns None when the action may proceed (no hold, or the hold expired
        — in which case the stale entry is dropped so it can re-run and
        re-verify). Cell-safe for canary leg keys via ``leg_key``.
        """
        key = self.idempotency_key(namespace, target, action.name)
        rec = self._dedup_history.get(key)
        if not rec or rec.get("verify") != "held":
            return None
        if _now() < rec.get("held_until", 0):
            logger.warning(
                "Remediation decision: action=%s target=%s outcome=held "
                "reason=verification hold active (retry after approval or hold expiry)",
                action.name, target,
            )
            return RemediationResult(
                action=action.name, target=target, success=False,
                dry_run=self.dry_run,
                message=(f"held: verification failed; waiting for hold expiry "
                         f"or operator approval (key={key})"),
                outcome=RemediationOutcome.HELD.value,
            )
        # Hold expired: allow the action to attempt again — drop the stale
        # entry so it re-executes and re-verifies from scratch.
        self._dedup_history.pop(key, None)
        self.verify_state[key] = {"status": "rearmed", "at": _utc_now()}
        return None

    # ── Canary widening (bounded-slice then widen) ─────────────────────────

    @staticmethod
    def _effective_canary_fraction(action) -> float:
        frac = float(getattr(action, "canary_fraction", 0.0) or 0.0)
        if frac <= 0.0 or config.CANARY_MAX_FRACTION <= 0.0:
            return 0.0
        return min(frac, config.CANARY_MAX_FRACTION)

    def _canary_leg_keys(self, namespace: str, target: str, action) -> tuple:
        base = self.idempotency_key(namespace, target, action.name)
        return f"{base}/canary", f"{base}/widen"

    def evaluate(
        self,
        pod_state,
        prediction,
        risk_score: float,
    ) -> list[RemediationResult]:
        """Evaluate all registered actions for a pod.

        Returns list of RemediationResult for actions that were executed.
        """
        results: list[RemediationResult] = []

        # Closed loop: run due post-action verifications for this pod first
        # (observe-only; a pod that recovered below the risk threshold is still
        # verified — verify must not be skipped just because risk dropped).
        self.verify_remediations(pod_state)

        if risk_score < self.risk_threshold:
            logger.debug(
                "Risk score %.1f below threshold %.1f — skipping remediation",
                risk_score,
                self.risk_threshold,
            )
            return results

        for action in self.actions:
            try:
                if not action.should_execute(pod_state, prediction, risk_score):
                    # Log skipped decision for binary observability
                    target = getattr(pod_state, "name", str(pod_state))
                    namespace = getattr(pod_state, "namespace", "")
                    outcome = RemediationOutcome.SKIPPED.value
                    logger.warning("Remediation decision: action=%s target=%s outcome=%s reason=should_execute returned False",
                                   action.name, target, outcome)
                    result = RemediationResult(
                        action=action.name,
                        target=target,
                        success=False,
                        dry_run=self.dry_run,
                        message="skipped: should_execute returned False",
                        outcome=outcome,
                    )
                    record_remediation_metric(action.name, RemediationOutcome.SKIPPED)
                    self.audit_trail.append(result)
                    self._add_to_remediation_log(result, namespace)
                    results.append(result)
                    continue

                target = getattr(pod_state, "name", str(pod_state))
                namespace = getattr(pod_state, "namespace", "")
                node_name = getattr(pod_state, "node_name", "")

                can_exec, reason = self.safety_policy.can_execute(
                    target, action.name, self.context, namespace
                )
                if not can_exec:
                    outcome = RemediationOutcome.SKIPPED.value
                    logger.warning("Remediation decision: action=%s target=%s outcome=%s reason=%s",
                                   action.name, target, outcome, reason)
                    result = RemediationResult(
                        action=action.name,
                        target=target,
                        success=False,
                        dry_run=self.dry_run,
                        message=f"skipped: {reason}",
                        outcome=outcome,
                    )
                    record_remediation_metric(action.name, RemediationOutcome.SKIPPED)
                    self.audit_trail.append(result)
                    self._add_to_remediation_log(result, namespace)
                    results.append(result)
                    continue

                # P3 idempotent remediation: skip if this action was already
                # executed for this namespace/target within the dedup window.
                dedup = self._dedup_skip_result(namespace, target, action)
                if dedup is not None:
                    record_remediation_metric(action.name, RemediationOutcome.SKIPPED)
                    self.audit_trail.append(dedup)
                    self._add_to_remediation_log(dedup, namespace)
                    results.append(dedup)
                    continue

                # P6 bounded autonomy: actions at/above REMEDIATION_REVIEW_TIER
                # never execute without human approval.
                gated = self._gate_reviews(action, target, namespace, risk_score)
                if gated is not None:
                    results.append(gated)
                    continue

                # Set per-pod context: namespace and node name for actions
                self.context.namespace = namespace
                self.context.node_name = node_name
                self.context.pod_phase = getattr(pod_state, "state", "Running")
                self.context.cpu_trend = getattr(pod_state, "cpu_trend", 0.0)
                self.context.mem_trend = getattr(pod_state, "memory_trend", 0.0)

                if getattr(action, "risk_tier", "moderate") == "destructive":
                    logger.warning(
                        "Destructive remediation executing: action=%s target=%s ns=%s dry_run=%s",
                        action.name, target, namespace, self.dry_run,
                    )
                # Closed loop: verification hold + canary widening are handled
                # by the dispatch step (after safety/review gates).
                result = self._dispatch_action(
                    action, pod_state, target, namespace, node_name,
                    from_state=getattr(pod_state, "state", ""),
                    risk_score=risk_score,
                )
                # Determine outcome based on success (keep pre-set outcomes like
                # canary/widen/held set by the dispatcher).
                if not result.outcome:
                    if result.success:
                        result.outcome = RemediationOutcome.ACTED.value
                    else:
                        result.outcome = RemediationOutcome.FAILED.value

                self.audit_trail.append(result)
                self._add_to_remediation_log(result, namespace)

                # Record metric for binary observability
                record_remediation_metric(
                    result.action,
                    RemediationOutcome(result.outcome if result.outcome else "acted"),
                )

                self.context.audit_log.append(
                    {
                        "action": result.action,
                        "target": result.target,
                        "success": result.success,
                        "dry_run": result.dry_run,
                        "message": result.message,
                        "timestamp": result.timestamp,
                        "command": result.command,
                        "outcome": result.outcome,
                    }
                )
                results.append(result)
                logger.info(
                    "Remediation decision: action=%s target=%s outcome=%s success=%s dry_run=%s",
                    result.action,
                    result.target,
                    result.outcome,
                    result.success,
                    result.dry_run,
                )
            except Exception as e:
                target = getattr(pod_state, "name", str(pod_state))
                namespace = getattr(pod_state, "namespace", "")
                outcome = RemediationOutcome.FAILED.value
                logger.exception("Remediation decision: action=%s target=%s outcome=%s error=%s",
                                 action.name, target, outcome, e)
                result = RemediationResult(
                    action=action.name,
                    target=target,
                    success=False,
                    dry_run=self.dry_run,
                    message=f"error: {e}",
                    outcome=outcome,
                )
                record_remediation_metric(action.name, RemediationOutcome.FAILED)
                self.audit_trail.append(result)
                self._add_to_remediation_log(result, namespace)
                # Failed attempts must reach the served audit trail too —
                # "never silent" applies to exceptions, not just decisions.
                self.context.audit_log.append(
                    {
                        "action": result.action,
                        "target": result.target,
                        "success": result.success,
                        "dry_run": result.dry_run,
                        "message": result.message,
                        "timestamp": result.timestamp,
                    }
                )
                results.append(result)

        return results

    def plan_and_execute(
        self,
        current_state: str,
        target: str,
        pod_state,
        prediction,
        risk_score: float,
        horizon: int = 3,
        namespace: str = "",
        node_name: str = "",
        max_actions: Optional[int] = None,
    ) -> list[RemediationResult]:
        """Plan a remediation sequence with the POMDP planner and execute it.

        Uses ``predictive_agent.pomdp`` to look ahead ``horizon`` steps and
        select the action sequence minimising expected P(FAILED), then executes
        each non-``none`` step through the registered action + ``SafetyPolicy``.
        Closes the loop: Markov prediction -> POMDP action selection ->
        RemediationManager execution. Safety guardrails applied here:

          * confidence gate (``config.MIN_ACTION_CONFIDENCE``): if the predictor
            is below threshold, do not act;
          * ``max_actions``: circuit breaker cap on actions executed this call
            (blast-radius control per reconcile cycle);
          * degraded fallback: when ``self.degraded`` is set (calibration drift
            detected), falls back to the conservative reactive ``evaluate`` path.
        """
        from predictive_agent import config
        from predictive_agent.pomdp import Planner

        # Safety guardrail: confidence gate — never act when the model is unsure.
        if prediction is not None:
            conf = getattr(prediction, "confidence", None)
            if conf is not None and conf < config.MIN_ACTION_CONFIDENCE:
                logger.info(
                    "Confidence %.2f below gate %.2f — skipping planner action on %s",
                    conf, config.MIN_ACTION_CONFIDENCE, target,
                )
                return []

        # Safety guardrail: confidence-gated acting (#10) — do not trust the
        # planner from a state it has barely observed. Fall back to the
        # conservative reactive path until enough outcome data has been learned
        # for this state.
        if config.MIN_MODEL_SUPPORT > 0 and not self.decision_model.is_confident(
            current_state, config.MIN_MODEL_SUPPORT
        ):
            logger.info(
                "Model support %.1f below MIN_MODEL_SUPPORT %.1f for %s — "
                "falling back to reactive remediation",
                self.decision_model.state_support(current_state),
                config.MIN_MODEL_SUPPORT, target,
            )
            return self.evaluate(pod_state, prediction, risk_score)

        # Safety guardrail: degraded fallback to reactive remediation.
        if not config.PLANNING_ENABLED or getattr(self, "degraded", False):
            return self.evaluate(pod_state, prediction, risk_score)

        # Closed loop: run due post-action verifications for this pod first
        # (observe-only; the reactive fallback above also calls verify, which
        # is idempotent — entries transition at most once).
        self.verify_remediations(pod_state)

        model = self.decision_model
        plan = Planner(model=model, action_costs=self.action_costs, failure_cost=self.failure_cost).plan(current_state, horizon)
        results: list[RemediationResult] = []
        steps = plan.steps
        if max_actions is not None and max_actions >= 0:
            steps = steps[:max_actions]
        for step in steps:
            if step.action == "none":
                continue
            res = self._apply_named_action(
                step.action, target, pod_state, prediction,
                risk_score, namespace, node_name,
                from_state=getattr(pod_state, "state", current_state),
            )
            if res is not None:
                results.append(res)
                if getattr(res, "success", False):
                    self.interventions_executed += 1
        # Decision-layer observability (#7): expected P(FAILED) reduction of the
        # executed plan vs doing nothing (the natural decay baseline).
        executed = sum(1 for r in results if getattr(r, "success", False))
        if executed and plan.steps:
            none_dist = {s: 0.0 for s in MarkovChain.STATES}
            none_dist[current_state] = 1.0
            for _ in range(horizon):
                none_dist = model.step_distribution(none_dist, "none")
            none_pf = model.failed_probability(none_dist)
            plan_pf = plan.steps[-1].expected_failed_prob_after
            self.cumulative_p_failed_reduction += max(0.0, none_pf - plan_pf)
        # Visibility (#watch): when the planner decides 'none' for a pod that is
        # genuinely at-risk, surface the decision instead of returning an empty
        # list (which would otherwise be logged as nothing happening). The
        # reconcile loop escalates after CRASHLOOP_ESCALATE_AFTER such cycles.
        # Skip when the circuit breaker fully capped this cycle (max_actions=0),
        # where an empty result is an intentional suppression, not a 'none'.
        if (not results and risk_score >= self.risk_threshold
                and (max_actions is None or max_actions > 0)):
            results.append(RemediationResult(
                action="none",
                target=target,
                success=False,
                dry_run=self.dry_run,
                message=(
                    "planner: no beneficial action \u2014 restart yields no durable "
                    "P(FAILED) reduction"
                ),
            ))
        return results

    def note_remediation_outcome(
        self, target: str, at_risk: bool, actions_taken: list
    ) -> tuple:
        """Track futile-remediation cycles and escalate crash-loops.

        A pod is "futile" when it is at-risk but every decided action this
        cycle is ``none`` (the planner concluded no action is beneficial, e.g.
        a deterministic crash-loop). Consecutive futile cycles are counted per
        target; once ``config.CRASHLOOP_ESCALATE_AFTER`` is reached the pod is
        escalated (the caller emits the alert + metric).

        Returns ``(escalated: bool, futile_count: int)``.
        """
        futile = bool(
            at_risk and actions_taken
            and all(a == "none" for a in actions_taken)
        )
        if not futile:
            self._futile_cycles.pop(target, None)
            return False, 0
        n = self._futile_cycles.get(target, 0) + 1
        self._futile_cycles[target] = n
        # Escalate exactly once, on the cycle that first crosses the threshold
        # (an ongoing crash-loop then stays visible via the per-cycle 'none'
        # decision log, but does not re-alert every reconcile).
        if n == config.CRASHLOOP_ESCALATE_AFTER:
            self.escalations += 1
            return True, n
        return False, n

    def plan_remediation(self, current_state: str, horizon: int = 3) -> dict:
        """Plan a remediation sequence without executing it (preview).

        Returns an execution-ready plan dict from ``self.decision_model``. Useful
        for surfacing the planned interventions via the ``/plan`` endpoint before
        any action is taken. Respects the same cooldowns as ``plan_and_execute``.
        """
        return Planner(model=self.decision_model, action_costs=self.action_costs, failure_cost=self.failure_cost).plan(
            current_state, horizon if horizon is not None else config.PLANNING_HORIZON
        ).to_dict()

    def feed_transition(self, from_state: str, action: str, to_state: str) -> None:
        """Teach the decision model an observed (state, action) -> next-state.

        Call this from the reconcile loop once a pod's post-remediation state is
        observed, so the planner's action-conditioned transitions improve from
        real outcomes instead of the conservative initial priors.
        """
        self.decision_model.record_transition(from_state, to_state, action)

    def refine_from_backtester(self, backtester) -> int:
        """Refine the base dynamics of ``self.decision_model`` from backtester data."""
        return self.decision_model.refine_from_backtester(backtester)

    def calibrate_from_incidents(
        self,
        incidents,
        slo_cost_per_minute: Optional[float] = None,
        typical_outage_minutes: Optional[float] = None,
        disruption_fraction: Optional[float] = None,
    ) -> dict:
        """Recalibrate action_costs / failure_cost from real incident history (#5).

        incidents: list of dicts with ``action`` and ``downtime_minutes`` (see
        :func:`predictive_agent.cost_model.calibrate_costs`). Returns the new
        cost model so callers (e.g. the /cost-model endpoint) can surface it.
        """
        from predictive_agent import config
        from predictive_agent.cost_model import calibrate_costs

        action_costs, failure_cost = calibrate_costs(
            incidents,
            slo_cost_per_minute=slo_cost_per_minute if slo_cost_per_minute is not None else config.SLO_COST_PER_MINUTE,
            typical_outage_minutes=typical_outage_minutes if typical_outage_minutes is not None else config.INCIDENT_TYPICAL_OUTAGE_MIN,
            disruption_fraction=disruption_fraction if disruption_fraction is not None else config.ACTION_DISRUPTION_FRACTION,
            default_action_cost=0.05,
            action_names=[a.name for a in self.actions],
        )
        self.action_costs = action_costs
        self.failure_cost = failure_cost
        self.cost_source = "calibrated"
        return {"action_costs": action_costs, "failure_cost": failure_cost}

    def _record_pending(
        self, target: str, from_state: str, action: str, predicted_p_failed=None
    ) -> None:
        """Remember an executed (state, action, predicted_p) triple for learning."""
        if not from_state or action == "none":
            return
        self._pending_feedback.setdefault(target, []).append(
            (from_state, action, predicted_p_failed)
        )

    def record_observations(self, pod_states: dict) -> None:
        """Learn from last cycle's actions using this cycle's observed states.

        For every pod with pending feedback, pair each remembered (from_state,
        action) with the now-observed ``pod_states[pod_key]`` and teach the
        decision model via ``feed_transition``. Clears pending feedback for pods
        whose new state is now known. Call this once per reconcile, after pods
        have been updated but before this cycle's remediation is recorded.
        """
        for pod_key, observed_state in pod_states.items():
            pending = self._pending_feedback.get(pod_key)
            if not pending:
                continue
            for from_state, action, predicted_p_failed in pending:
                self.feed_transition(from_state, action, observed_state)
                if self.calibration_monitor is not None and predicted_p_failed is not None:
                    actual_failed = observed_state in ("CRITICAL", "FAILED")
                    self.calibration_monitor.record(predicted_p_failed, actual_failed)
                if self.rl_policy is not None:
                    self.rl_policy.update(from_state, action, observed_state)
            del self._pending_feedback[pod_key]

    def clear_pending(self, pod_key: str) -> None:
        """Drop pending feedback for a removed/unknown pod (no outcome observed)."""
        self._pending_feedback.pop(pod_key, None)

    # ── P6: bounded remediation autonomy ───────────────────────────────

    def _review_signature(self, namespace: str, target: str, action_name: str) -> str:
        return f"{(namespace or '-')}/{(target or '-')}/{action_name}"

    def _gate_reviews(self, action, target: str, namespace: str, risk_score: float):
        """Return a deferred RemediationResult if the action needs review.

        An action at/above the configured REMEDIATION_REVIEW_TIER is never
        executed until its (namespace, target, action) signature is approved;
        the deferral is logged at WARN, recorded in the remediation audit log
        (binary observability) and queued for GET /review — never silent.
        """
        from predictive_agent import config as _cfg

        tier = getattr(action, "risk_tier", "moderate")
        if self.review_tier not in _cfg.RISK_TIERS:
            return None  # gate disabled ("none")
        if _cfg.RISK_TIERS.get(tier, 1) < _cfg.RISK_TIERS[self.review_tier]:
            return None  # below the configured tier: no review needed

        sig = self._review_signature(namespace, target, action.name)
        approved_at = self._approved.get(sig)
        if approved_at is not None and (_now() - approved_at) <= self.review_ttl_s:
            return None  # approved and still valid

        # Defer: queue for review + loud logs + binary-observability record.
        self._review_seq += 1
        rid = f"rev-{self._review_seq}"
        self.review_queue[rid] = {
            "id": rid,
            "signature": sig,
            "action": action.name,
            "target": target,
            "namespace": namespace,
            "tier": tier,
            "risk_score": round(float(risk_score), 4),
            "status": "pending",
            "created_at": _utc_now(),
        }
        self.reviews_deferred += 1
        logger.warning(
            "Remediation decision: action=%s target=%s outcome=review_required "
            "reason=tier %s >= REMEDIATION_REVIEW_TIER=%s (approve via POST /review)",
            action.name, target, tier, self.review_tier,
        )
        result = RemediationResult(
            action=action.name,
            target=target,
            success=False,
            dry_run=self.dry_run,
            message=(
                f"review_required: tier {tier} >= {self.review_tier}; "
                f"approve signature {sig} (review id {rid})"
            ),
            outcome=RemediationOutcome.REVIEW_REQUIRED.value,
        )
        record_remediation_metric(action.name, RemediationOutcome.REVIEW_REQUIRED)
        self.audit_trail.append(result)
        self._add_to_remediation_log(result, namespace)
        return result

    def approve_review(self, signature: str = "", review_id: str = "") -> Optional[dict]:
        """Approve a (namespace, target, action) signature (P6).

        Accepts either the signature directly or a pending review id. The
        approval allows the next reconcile to execute that action on that
        target; it expires after REMEDIATION_REVIEW_TTL_S seconds.
        """
        sig = signature
        if not sig and review_id:
            item = self.review_queue.get(review_id)
            if item is None:
                return None
            sig = item.get("signature", "")
        if not sig:
            return None
        self._approved[sig] = _now()
        approved = 0
        for item in self.review_queue.values():
            if item.get("signature") == sig and item.get("status") == "pending":
                item["status"] = "approved"
                item["approved_at"] = _utc_now()
                approved += 1
        # Closed loop: approving a signature also clears verification holds for
        # that signature (an operator approval un-blocks a held retry).
        for k, rec in list(self._dedup_history.items()):
            if rec.get("verify") == "held" and rec.get("sig") == sig:
                self._dedup_history.pop(k, None)
                self.verify_state[k] = {"status": "approved_retry", "at": _utc_now()}
        logger.warning(
            "[review] signature %s approved (%s queued item(s) released; "
            "valid for %ss)",
            sig, approved, self.review_ttl_s,
        )
        return {"signature": sig, "released": approved}

    def list_reviews(self, include_resolved: bool = False) -> list:
        """Pending (and optionally resolved) review items for GET /review."""
        items = []
        for item in self.review_queue.values():
            if not include_resolved and item.get("status") != "pending":
                continue
            items.append(item)
        return sorted(items, key=lambda i: i.get("created_at", ""))

    def _execute_with_timeout(self, action, target: str, context) -> RemediationResult:
        """Run action.execute with the P6 per-action checkpoint.

        An action that does not complete within REMEDIATION_TIMEOUT_S is
        abandoned (we stop waiting; the daemon thread cannot be killed) and
        logged as failed — never a silent hang of the reconcile loop.
        """
        if self.action_timeout_s <= 0:
            return action.execute(target, context)
        box: dict = {}

        def _run():
            try:
                box["result"] = action.execute(target, context)
            except Exception as e:  # surfaced by the caller's except path
                box["error"] = e

        import threading as _threading

        t = _threading.Thread(target=_run, daemon=True, name=f"rem-{action.name}")
        t.start()
        t.join(self.action_timeout_s)
        if t.is_alive():
            logger.error(
                "[op] remediation timeout: action=%s target=%s did not complete "
                "within %.0fs — abandoning",
                action.name, target, self.action_timeout_s,
            )
            return RemediationResult(
                action=action.name,
                target=target,
                success=False,
                dry_run=context.dry_run,
                message=f"timeout: action did not complete within {self.action_timeout_s:.0f}s",
                outcome=RemediationOutcome.FAILED.value,
            )
        if "error" in box:
            raise box["error"]
        return box["result"]

    # ── Closed loop: hold gate + canary widening dispatch ─────────────────

    def _leg_dedup_skip(self, action, target: str, leg_key: str, rec: dict):
        """Dedup-skip for a canary leg key (same semantics as _dedup_skip_result)."""
        now = _now()
        if now - rec["timestamp"] >= self.dedup_window:
            if self._hold_still_active(rec, now):
                return None
            self._dedup_history.pop(leg_key, None)
            self.verify_state.pop(leg_key, None)
            # consecutive-failure counter kept across re-arms (as above)
            return None
        self.deduped_count += 1
        logger.warning(
            "[op] dedup: action already executed (idempotency_key=%s)", leg_key,
        )
        return RemediationResult(
            action=action.name,
            target=target,
            success=False,
            dry_run=self.dry_run,
            message=(
                f"[op] dedup: action already executed "
                f"(idempotency_key={leg_key}, last run {int(now - rec['timestamp'])}s ago)"
            ),
            outcome=RemediationOutcome.SKIPPED.value,
            idempotency_key=leg_key,
        )

    def _dispatch_action(
        self,
        action: RemediationAction,
        pod_state,
        target: str,
        namespace: str,
        node_name: str,
        *,
        from_state: str = "",
        risk_score: float = 0.0,
    ) -> RemediationResult:
        """Execute one action with verification hold + canary widening.

        Order of gates (after the caller's safety/review gates):
          1. verification hold — a held action is not re-executed;
          2. canary widening — canary-capable actions on multi-replica targets
             run a bounded first leg, then (only after that leg verifies) a
             widen leg, each under its own dedup/verify key;
          3. plain execution under the base idempotency key.
        The returned result carries a pre-set outcome (canary/widen/held) or is
        plain (caller derives acted/failed from ``success``).
        """
        # Verification hold: a previously-failed verification blocks
        # re-execution until VERIFY_HOLD_S elapses or the operator approves.
        held = self._hold_skip(action, target, namespace)
        if held is not None:
            return held

        frac = self._effective_canary_fraction(action)
        replicas = max(1, int(getattr(pod_state, "replicas", 1) or 1))
        canary_key, widen_key = self._canary_leg_keys(namespace, target, action)

        if frac > 0.0 and replicas > 1:
            widen_rec = self._dedup_history.get(widen_key)
            if widen_rec is not None:
                if widen_rec.get("verify") == "held":
                    if _now() < widen_rec.get("held_until", 0):
                        # Widen leg failed verification: hold governs the whole
                        # cycle — never widen (or re-run) while held.
                        logger.info(
                            "Remediation decision: action=%s target=%s outcome=held "
                            "reason=widen verification hold active",
                            action.name, target,
                        )
                        return RemediationResult(
                            action=action.name,
                            target=target,
                            success=False,
                            dry_run=self.dry_run,
                            message="held: widen verification failed — hold active",
                            outcome=RemediationOutcome.HELD.value,
                            idempotency_key=widen_key,
                        )
                    # Widen hold expired → rearm the whole cycle (a fresh
                    # canary leg must verify again before the next widen).
                    self._dedup_history.pop(widen_key, None)
                    self._dedup_history.pop(canary_key, None)
                    self.verify_state[widen_key] = {"status": "rearmed", "at": _utc_now()}
                    widen_rec = None
                else:
                    # Already widened within the window → dedup-skip (never
                    # double-execute the full change).
                    skip = self._leg_dedup_skip(action, target, widen_key, widen_rec)
                    if skip is not None:
                        return skip
                    widen_rec = None  # dedup expired → fall through to rearm
            canary_rec = self._dedup_history.get(canary_key)
            if canary_rec is not None and canary_rec.get("verify") == "held":
                if _now() >= canary_rec.get("held_until", 0):
                    # Hold expired → re-arm: allow a fresh canary attempt (it
                    # will re-execute and re-verify from scratch).
                    self._dedup_history.pop(canary_key, None)
                    self.verify_state[canary_key] = {"status": "rearmed", "at": _utc_now()}
                    canary_rec = None
                else:
                    logger.info(
                        "Remediation decision: action=%s target=%s outcome=held "
                        "reason=canary verification hold active — widen blocked",
                        action.name, target,
                    )
                    return RemediationResult(
                        action=action.name,
                        target=target,
                        success=False,
                        dry_run=self.dry_run,
                        message="held: canary verification failed — widen blocked until hold expiry",
                        outcome=RemediationOutcome.HELD.value,
                        idempotency_key=canary_key,
                    )
            if canary_rec is not None:
                if canary_rec.get("verify") == "verified":
                    # Canary leg verified → widen now (the full intended change).
                    return self._run_and_record(
                        action, pod_state, target, namespace, from_state,
                        risk_score, leg_key=widen_key, canary_mode="widen",
                        canary_fraction=frac, replicas=replicas,
                    )
                # Canary leg ran but is not verified yet (pending or held) →
                # do not widen and do not re-run the canary within the window.
                logger.info(
                    "Remediation decision: action=%s target=%s outcome=held "
                    "reason=canary leg not yet verified — widen deferred",
                    action.name, target,
                )
                return RemediationResult(
                    action=action.name,
                    target=target,
                    success=False,
                    dry_run=self.dry_run,
                    message="held: canary leg not yet verified — widen deferred",
                    outcome=RemediationOutcome.HELD.value,
                    idempotency_key=canary_key,
                )
            # No canary leg yet → execute the bounded canary (first) leg.
            return self._run_and_record(
                action, pod_state, target, namespace, from_state,
                risk_score, leg_key=canary_key, canary_mode="canary",
                canary_fraction=frac, replicas=replicas,
            )

        # Non-canary (or single-replica) path: plain execution, base key.
        return self._run_and_record(
            action, pod_state, target, namespace, from_state,
            risk_score, leg_key=self.idempotency_key(namespace, target, action.name),
            canary_mode=None, canary_fraction=0.0, replicas=1,
        )

    def _run_and_record(
        self,
        action: RemediationAction,
        pod_state,
        target: str,
        namespace: str,
        from_state: str,
        risk_score: float,
        *,
        leg_key: str,
        canary_mode: Optional[str],
        canary_fraction: float,
        replicas: int,
    ) -> RemediationResult:
        """Execute the action and record the dedup entry with a signal snapshot.

        Sets the canary context for actions that support bounded legs, records
        the execution under ``leg_key`` (P3 dedup + verify-after-act snapshot),
        feeds the learning pipeline, and records the safety-policy cooldown.
        """
        # Canary context: bounded-leg actions read these from the context.
        self.context.canary_mode = canary_mode
        self.context.canary_fraction = canary_fraction
        self.context.canary_replicas = replicas
        result = self._execute_with_timeout(action, target, self.context)
        if result.success:
            # P3 idempotent remediation + verify-after-act: record the
            # execution with the signal snapshot at execution time so the
            # verification step can compare post-action signals.
            snapshot = self._signal_snapshot(pod_state)
            self._record_execution(
                namespace, target, action, result, snapshot=snapshot, leg_key=leg_key,
            )
            if canary_mode == "canary":
                result.outcome = RemediationOutcome.CANARY.value
                self.outcome_alerts.append({
                    "kind": "remediation_canary", "action": action.name,
                    "target": target, "risk_score": risk_score,
                    "details": (
                        f"canary leg executed (fraction={canary_fraction:.2f}, "
                        f"replicas={replicas}); widen awaits verification"
                    ),
                })
            elif canary_mode == "widen":
                result.outcome = RemediationOutcome.WIDENED.value
                base_key = self.idempotency_key(namespace, target, action.name)
                self._canary_state[base_key] = "widened"
                self.outcome_alerts.append({
                    "kind": "remediation_widened", "action": action.name,
                    "target": target, "risk_score": risk_score,
                    "details": "widen leg executed after canary leg verified",
                })
        elif not result.outcome:
            result.outcome = RemediationOutcome.FAILED.value
        self._record_pending(target, from_state, action.name, predicted_p_failed=risk_score)
        self.safety_policy.record_action(target, action.name, self.context)
        return result

    def _apply_named_action(
        self,
        action_name: str,
        target: str,
        pod_state,
        prediction,
        risk_score: float,
        namespace: str,
        node_name: str,
        from_state: str = "",
    ) -> Optional[RemediationResult]:
        """Execute one named action via the registered action + SafetyPolicy."""
        action = next((a for a in self.actions if a.name == action_name), None)
        if action is None:
            logger.info("Planned action %s not registered — skipping", action_name)
            return None

        can_exec, reason = self.safety_policy.can_execute(
            target, action_name, self.context, namespace
        )
        if not can_exec:
            logger.info("Planned action %s skipped: %s", action_name, reason)
            result = RemediationResult(
                action=action_name,
                target=target,
                success=False,
                dry_run=self.dry_run,
                message=f"skipped: {reason}",
                outcome=RemediationOutcome.SKIPPED.value,
            )
            record_remediation_metric(action_name, RemediationOutcome.SKIPPED)
            self.audit_trail.append(result)
            self.context.audit_log.append(
                {
                    "action": result.action,
                    "target": result.target,
                    "success": result.success,
                    "dry_run": result.dry_run,
                    "message": result.message,
                    "timestamp": result.timestamp,
                    "command": result.command,
                }
            )
            return result

        # The planner selects by objective, but we still honour each action's own
        # eligibility guard so we never run an ineligible intervention (e.g.
        # restart a pod the action itself deems healthy).
        if not action.should_execute(pod_state, prediction, risk_score):
            logger.info("Planned action %s not eligible (should_execute=False)", action_name)
            result = RemediationResult(
                action=action_name,
                target=target,
                success=False,
                dry_run=self.dry_run,
                message="skipped: action ineligible",
                outcome=RemediationOutcome.SKIPPED.value,
            )
            record_remediation_metric(action_name, RemediationOutcome.SKIPPED)
            self.audit_trail.append(result)
            self.context.audit_log.append(
                {
                    "action": result.action,
                    "target": result.target,
                    "success": result.success,
                    "dry_run": result.dry_run,
                    "message": result.message,
                    "timestamp": result.timestamp,
                    "command": result.command,
                }
            )
            return result

        # P3 idempotent remediation: same dedup gate as the reactive path — a
        # planner step for an action that already ran within the window is
        # skipped rather than double-executed (crash-mid-remediation safety).
        dedup = self._dedup_skip_result(namespace, target, action)
        if dedup is not None:
            self.audit_trail.append(dedup)
            self.context.audit_log.append(
                {
                    "action": dedup.action,
                    "target": dedup.target,
                    "success": dedup.success,
                    "dry_run": dedup.dry_run,
                    "message": dedup.message,
                    "timestamp": dedup.timestamp,
                    "command": dedup.command,
                }
            )
            return dedup

        # P6 bounded autonomy: the planner path honours the same review gate
        # and per-action checkpoint as the reactive path.
        gated = self._gate_reviews(action, target, namespace, risk_score)
        if gated is not None:
            self.context.audit_log.append(
                {
                    "action": gated.action,
                    "target": gated.target,
                    "success": gated.success,
                    "dry_run": gated.dry_run,
                    "message": gated.message,
                    "timestamp": gated.timestamp,
                    "command": gated.command,
                    "outcome": gated.outcome,
                }
            )
            return gated

        self.context.namespace = namespace
        self.context.node_name = node_name
        self.context.pod_phase = getattr(pod_state, "state", "Running")
        self.context.cpu_trend = getattr(pod_state, "cpu_trend", 0.0)
        self.context.mem_trend = getattr(pod_state, "memory_trend", 0.0)
        if getattr(action, "risk_tier", "moderate") == "destructive":
            logger.warning(
                "Destructive remediation executing: action=%s target=%s ns=%s dry_run=%s",
                action.name, target, namespace, self.dry_run,
            )
        result = self._dispatch_action(
            action, pod_state, target, namespace, node_name,
            from_state=from_state, risk_score=risk_score,
        )
        if not result.outcome:
            result.outcome = (
                RemediationOutcome.ACTED.value if result.success
                else RemediationOutcome.FAILED.value
            )
        record_remediation_metric(
            result.action,
            RemediationOutcome(result.outcome if result.outcome else "acted"),
        )
        self.audit_trail.append(result)
        self.context.audit_log.append(
            {
                "action": result.action,
                "target": result.target,
                "success": result.success,
                "dry_run": result.dry_run,
                "message": result.message,
                "timestamp": result.timestamp,
                "command": result.command,
                "outcome": result.outcome,
            }
        )
        return result

    def get_audit_trail(self, limit: int = 20) -> list[dict]:
        """Return recent audit trail entries."""
        return self.context.audit_log[-limit:]

    def get_remediation_log(self, limit: int = 20) -> list[dict]:
        """Return the structured remediation audit log for /remediation-log endpoint.
        
        Returns all remediation decisions with acted, skipped, or failed outcomes.
        """
        return self.context.remediation_log[-limit:]

    def get_remediation_state(self, limit: int = 50) -> dict:
        """Return in-flight verification/hold/canary state (GET /remediation/state).

        Surfaces the closed-loop lifecycle to operators: which executions are
        pending verification, which are held (and for how long), which canary
        legs await widening, and the cumulative outcome-alert queue.
        """
        entries = []
        for key, st in list(self.verify_state.items())[-limit:]:
            rec = self._dedup_history.get(key) or {}
            held_in = 0.0
            if rec.get("verify") == "held" and rec.get("held_until"):
                held_in = max(0.0, rec["held_until"] - _now())
            entries.append({
                "idempotency_key": key,
                "status": st.get("status"),
                "verify": rec.get("verify"),
                "leg": rec.get("leg", "full"),
                "failures": self._verify_failures.get(key, 0),
                "hold_remaining_s": round(held_in, 1),
                "criteria": st.get("criteria"),
                "at": st.get("at"),
            })
        return {
            "verify_enabled": bool(getattr(config, "VERIFY_ENABLED", True)),
            "settling_s": config.VERIFY_SETTLING_S,
            "min_signals": config.VERIFY_MIN_SIGNALS,
            "hold_s": config.VERIFY_HOLD_S,
            "escalate_after": config.VERIFY_ESCALATE_AFTER,
            "canary_max_fraction": config.CANARY_MAX_FRACTION,
            "revert_enabled": bool(getattr(config, "REMEDITATION_REVERT_ENABLED", True)),
            "entries": entries,
            "canary": dict(self._canary_state),
            "pending_outcome_alerts": len(self.outcome_alerts),
        }

    def _add_to_remediation_log(self, result: RemediationResult, namespace: str = "") -> None:
        """Add a remediation result to the structured audit log."""
        outcome = result.get_outcome()
        entry = {
            "action": result.action,
            "target": result.target,
            "namespace": namespace,
            "outcome": outcome,
            # Closed loop: lifecycle mirrors outcome with the extended verdicts
            # (verified/held/reverted/canary/widened/approval_required) so
            # consumers can track act → verify → widen/revert explicitly.
            "lifecycle": result.outcome or outcome,
            "idempotency_key": getattr(result, "idempotency_key", "") or "",
            "success": result.success,
            "dry_run": result.dry_run,
            "message": result.message,
            "timestamp": result.timestamp,
            "command": result.command,
        }
        self.context.remediation_log.append(entry)

    def get_stats(self) -> dict:
        """Return remediation statistics."""
        total = len(self.audit_trail)
        successful = sum(1 for r in self.audit_trail if r.success)
        dry_run = sum(1 for r in self.audit_trail if r.dry_run)
        acted = sum(1 for r in self.audit_trail if r.get_outcome() == RemediationOutcome.ACTED.value)
        skipped = sum(1 for r in self.audit_trail if r.get_outcome() == RemediationOutcome.SKIPPED.value)
        failed = sum(1 for r in self.audit_trail if r.get_outcome() == RemediationOutcome.FAILED.value)
        held = sum(1 for r in self.audit_trail if r.get_outcome() == RemediationOutcome.HELD.value)
        canary = sum(1 for r in self.audit_trail if r.get_outcome() == RemediationOutcome.CANARY.value)
        widened = sum(1 for r in self.audit_trail if r.get_outcome() == RemediationOutcome.WIDENED.value)
        reverted = sum(1 for r in self.audit_trail if r.get_outcome() == RemediationOutcome.REVERTED.value)
        return {
            "total_actions": total,
            "successful_actions": successful,
            "failed_actions": total - successful,
            "dry_run_actions": dry_run,
            "registered_actions": [a.name for a in self.actions],
            "risk_threshold": self.risk_threshold,
            "dry_run": self.dry_run,
            "dedup_window": self.dedup_window,
            "deduped_actions": self.deduped_count,
            "outcomes": {
                "acted": acted,
                "skipped": skipped,
                "failed": failed,
                "held": held,
                "canary": canary,
                "widened": widened,
                "reverted": reverted,
            },
            "verifications": {
                "tracked": len(self.verify_state),
                "pending": sum(1 for s in self.verify_state.values() if s.get("status") == "pending"),
                "verified": sum(1 for s in self.verify_state.values() if s.get("status") == "verified"),
                "held": sum(1 for s in self.verify_state.values() if s.get("status") == "held"),
            },
        }


def create_remediation_manager_from_config() -> RemediationManager:
    """Create a RemediationManager configured from environment variables."""
    enabled = os.environ.get("REMEDIATION_ENABLED", "false").lower() in ("true", "1", "yes")
    dry_run = os.environ.get("REMEDIATION_DRY_RUN", "true").lower() in ("true", "1", "yes")

    # If remediation is disabled, force dry_run
    if not enabled:
        dry_run = True

    protected_ns = {
        ns.strip()
        for ns in os.environ.get(
            "REMEDIATION_PROTECTED_NS", "kube-system,opendesk-predictive-agent"
        ).split(",")
        if ns.strip()
    }

    policy = SafetyPolicy(
        max_per_minute=int(os.environ.get("REMEDIATION_MAX_PER_MIN", "5")),
        max_per_hour=int(os.environ.get("REMEDIATION_MAX_PER_HOUR", "50")),
        cooldown_seconds=int(os.environ.get("REMEDIATION_COOLDOWN_S", "300")),
        protected_namespaces=protected_ns,
    )

    return RemediationManager(
        dry_run=dry_run,
        risk_threshold=float(os.environ.get("REMEDIATION_RISK_THRESHOLD", "0.7")),
        safety_policy=policy,
        dedup_window=int(os.environ.get("REMEDIATION_DEDUP_WINDOW", "300")),
    )
