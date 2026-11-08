"""Reconcile pipeline phases extracted from main.py (behavior preserved).

Each phase receives an explicit ``ReconcileContext`` holding the
component references and collection hooks that ``main`` resolves from
its module globals at reconcile time (late binding — patch targets such
as ``predictive_agent.main._get_pods_json`` or ``main.detect_runtime``
keep intercepting). Phases return result objects; ``main.reconcile()``
owns the module globals and applies the deltas.

The background LLM-analysis thread increments the main-module counters
(``_llm_calls`` / ``_llm_errors``) via a deferred main import — the same
runtime semantics as the original ``global`` declaration.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Set, Tuple

from predictive_agent import config
from predictive_agent.collector import docker_status_signals
from predictive_agent.pod_collection import kubectl_binary_missing
from predictive_agent.self_health import DegradedCycleTracker, SelfHealthAlerter

logger = logging.getLogger("predictive-agent")

# ── Self-health alerting (agent's OWN failure modes, rate-limited email) ──
# Lazily created per-notifier so tests that build ReconcileContext directly
# work unchanged; reset via reset_self_health().
_self_health: SelfHealthAlerter | None = None
_degraded_tracker: DegradedCycleTracker | None = None


def _get_self_health(ctx: "ReconcileContext") -> Tuple[SelfHealthAlerter, DegradedCycleTracker]:
    """Return the module-level (alerter, tracker), wiring ctx.notifier email."""
    global _self_health, _degraded_tracker
    if _self_health is None:
        notify = None
        email = getattr(ctx.notifier, "email", None) if ctx.notifier else None
        if email is not None:
            notify = email.send
        _self_health = SelfHealthAlerter(notify_fn=notify)
        _degraded_tracker = DegradedCycleTracker(_self_health)
    assert _self_health is not None and _degraded_tracker is not None
    return _self_health, _degraded_tracker


def reset_self_health() -> None:
    """Drop cached self-health state (tests / notifier re-wiring)."""
    global _self_health, _degraded_tracker
    _self_health = None
    _degraded_tracker = None

# One-shot diagnostics (avoid log spam).
_warned_top_failed = False

# ── Context and result types ────────────────────────────────────────────────


@dataclass(frozen=True)
class ReconcileContext:
    """Component references and hooks for one reconcile cycle.

    Built by ``main.reconcile()`` from its module globals at call time so
    tests patching ``predictive_agent.main.*`` keep working (late binding).
    """

    # Components
    state_model: Any
    predictor: Any
    state_store: Any
    remediation_manager: Any
    notifier: Any
    shadow_harness: Any
    policy_engine: Any
    llm_analyzer: Any
    kg_client: Any
    backtester: Any

    # Mutable module state (shared references — mutations are visible)
    llm_analysis_cache: Dict[str, float]
    llm_in_flight: Set[str]
    last_plan_notify: Dict[str, Tuple[str, float]]
    last_autonomy_notify: Dict[str, Tuple[str, float]]

    # Collection / status hooks (resolved from main's namespace)
    detect_runtime: Callable[[], str]
    discover_docker_containers: Callable[[], dict]
    collect_docker_metrics: Callable[[], dict]
    run_cmd: Callable
    collect_top_metrics: Callable[[str], dict]
    collect_top_nodes: Callable[[str], None]
    get_pods_json: Callable[[], dict]
    get_node_conditions: Callable[[dict], dict]
    get_pod_namespace: Callable[[dict], str]
    get_pod_name: Callable[[dict], str]
    get_pod_memory_limit: Callable[..., int]
    get_pod_restart_count: Callable[[dict], int]
    should_skip_namespace: Callable[[str], bool]
    is_pod_excluded: Callable[[dict], bool]
    get_node_pressure: Callable[[dict, str], Tuple[bool, bool]]
    matches_docker_selectors: Callable[[dict, dict], bool]
    get_container_memory_limit: Callable[[dict], int]
    get_watch_selectors: Callable[[], dict]
    get_pod_status_signals: Callable[[dict], dict]
    get_container_status_signals: Callable[[dict], dict]
    count_log_errors: Callable[[str], int]
    now_iso: Callable[[], str]
    get_collector_state: Callable[[], str]
    record_successful_collection: Callable[[], None]
    record_collection_failure: Callable[[], None]


@dataclass
class CollectResult:
    """Outcome of the metric-collection phase."""

    runtime: str
    pod_items: List[dict] = field(default_factory=list)
    pod_metrics: Dict[str, dict] = field(default_factory=dict)
    node_conditions: Dict[str, dict] = field(default_factory=dict)
    current_keys: Set[str] = field(default_factory=set)
    collection_successful: bool = True
    collector_state: str = "HEALTHY"


@dataclass
class PreparedCycle:
    """State-model reconciliation before the per-pod loop."""

    current_keys: Set[str]
    pod_items: List[dict]


@dataclass
class PodLoopResult:
    """Counters produced by the per-pod evaluation loop."""

    pods_tracked: int = 0
    at_risk_count: int = 0
    cycle_plans: Dict[str, Any] = field(default_factory=dict)
    actions_executed: int = 0


# ── Phase 1: runtime detection and metric collection ───────────────────────


def collect_metrics(ctx: ReconcileContext, cycle: int) -> CollectResult:
    """Detect the runtime and collect pod/container metrics."""
    global _warned_top_failed
    result = CollectResult(runtime=ctx.detect_runtime())
    logger.debug("Reconcile #%d: runtime=%s", cycle, result.runtime)

    current_keys: Set[str] = set()
    pod_metrics: Dict[str, dict] = {}
    node_conditions: Dict[str, dict] = {}
    pod_items: List[dict] = []

    try:
        if result.runtime == "docker":
            # Docker mode: discover containers and their metrics
            containers = ctx.discover_docker_containers()
            docker_metrics = ctx.collect_docker_metrics()

            pod_metrics = docker_metrics
            # For Docker, node_conditions is not applicable
            node_conditions = {}

            # An empty result is a *valid* result (no containers), not a
            # collection failure. Real failures (docker CLI unreachable) surface
            # as exceptions caught below -> DEGRADED -> keep last-known state.
            # Build list of container items with pod-like structure
            pod_items = []
            for container_id, info in containers.items():
                # Create a pod-like dict for processing
                # Use compose_project as namespace, or "docker" as default
                ns = info.get("compose_project", "docker") or "docker"
                name = info.get("name", container_id[:12])
                pod_key = f"{ns}/{name}"

                pod_items.append({
                    "metadata": {
                        "namespace": ns,
                        "name": name,
                        "creationTimestamp": info.get("created_at", ""),
                    },
                    "spec": {
                        "nodeName": "docker-host",
                    },
                    "status": {
                        "phase": "Running" if info.get("status") == "running" else "Succeeded" if info.get("status") == "exited" else "Unknown",
                        "containerStatuses": [{
                            "ready": info.get("healthy", True),
                            "restartCount": info.get("restart_count", 0),
                        }],
                    },
                    "_container_id": container_id,
                    "_docker_info": info,
                })
                current_keys.add(pod_key)
        else:
            # Kubernetes mode: use kubectl commands
            rc_pods, pods_output, _ = ctx.run_cmd(
                ["kubectl", "top", "pods", "-A", "--no-headers"], timeout=10
            )
            if rc_pods != 0 and not kubectl_binary_missing() and not _warned_top_failed:
                # kubectl exists but `kubectl top` fails: the resource metrics
                # API (metrics-server) is likely down/absent. Warn once so the
                # gap is visible — tracking still works via `get pods`.
                _warned_top_failed = True
                logger.warning(
                    "kubectl top pods failed (rc=%s) but kubectl is present — "
                    "metrics-server / resource metrics API likely unavailable; "
                    "per-pod CPU/memory telemetry is empty until it recovers.",
                    rc_pods,
                )
            pod_metrics = ctx.collect_top_metrics(pods_output) if rc_pods == 0 else {}

            rc_nodes, nodes_output, _ = ctx.run_cmd(
                ["kubectl", "top", "nodes", "--no-headers"], timeout=10
            )
            ctx.collect_top_nodes(nodes_output) if rc_nodes == 0 else {}

            pods_json = ctx.get_pods_json()

            rc_nodes_json, nodes_json_output, _ = ctx.run_cmd(
                ["kubectl", "get", "nodes", "-o", "json"], timeout=10
            )
            if rc_nodes_json == 0 and nodes_json_output:
                try:
                    node_conditions = ctx.get_node_conditions(json.loads(nodes_json_output))
                except json.JSONDecodeError:
                    pass

            pod_items = pods_json.get("items", [])

            # An empty pod list is a valid collection result (no pods match
            # the selectors), not a failure. Real failures (kubectl unreachable)
            # surface as exceptions caught below -> DEGRADED -> keep last-known.
            # Track current pod keys for cleanup
            for pod in pod_items:
                ns = ctx.get_pod_namespace(pod)
                name = ctx.get_pod_name(pod)
                if not ctx.should_skip_namespace(ns):
                    # Don't track completed Job/CronJob pods — they don't need monitoring
                    phase = pod.get("status", {}).get("phase", "")
                    if phase == "Succeeded":
                        continue
                    owner_refs = pod.get("metadata", {}).get("ownerReferences", [])
                    if any(ref.get("kind") in ("Job", "CronJob") for ref in owner_refs):
                        continue
                    # Explicitly excluded pods (KubeVirt virt-launcher VM pods or
                    # annotations/labels opted out) are not tracked at all.
                    if ctx.is_pod_excluded(pod):
                        continue
                    current_keys.add(f"{ns}/{name}")
    except Exception as e:  # noqa: BLE001 - collection must degrade, not crash
        # Collection failed with exception
        logger.warning("Collection failed with exception: %s", e)
        result.collection_successful = False

    result.pod_items = pod_items
    result.pod_metrics = pod_metrics
    result.node_conditions = node_conditions
    result.current_keys = current_keys

    # Update collector state based on collection result (P4)
    if result.collection_successful:
        ctx.record_successful_collection()
        logger.debug("Reconcile #%d: collector state = HEALTHY", cycle)
    else:
        ctx.record_collection_failure()
        logger.debug("Reconcile #%d: collector state = %s", cycle, ctx.get_collector_state())
    result.collector_state = ctx.get_collector_state()
    # Self-health: sustained collector degradation emails once per streak
    # (single DEGRADED cycles are transient kubectl noise; /ready 503 does
    # not restart the pod, so nothing else would surface a long streak).
    _get_self_health(ctx)[1].cycle(result.collector_state)

    return result


# ── Phase 2: reconcile the state model with collected reality ───────────────


def prepare_state_model(
    ctx: ReconcileContext,
    cycle: int,
    collector_state: str,
    current_keys: Set[str],
    pod_items: List[dict],
) -> PreparedCycle:
    """Apply DEGRADED keep-last-known semantics and finalize removed pods."""
    # In degraded mode with no new data, don't remove pods - use last-known state
    if collector_state == "DEGRADED" and not current_keys:
        # Keep all existing pods, don't remove any
        logger.info("Reconcile #%d: DEGRADED mode - keeping %d last-known pods",
                    cycle, len(ctx.state_model.pods))
        current_keys = set(ctx.state_model.pods.keys())

    # Finalize removed containers/pods
    removed_keys = set(ctx.state_model.pods.keys()) - current_keys
    for pod_key in removed_keys:
        # Parse pod_key to get namespace and name
        parts = pod_key.split("/", 1)
        if len(parts) == 2:
            ns, name = parts
            tracker = ctx.state_model.pods.get(pod_key)
            if tracker and tracker.state != "FAILED":
                # Record final transition to FAILED
                ctx.state_model.markov.record_transition(tracker.state, "FAILED")
            logger.info("Finalizing removed container/pod: %s", pod_key)
            # Remove from state model
            if pod_key in ctx.state_model.pods:
                del ctx.state_model.pods[pod_key]
            if ctx.remediation_manager is not None:
                ctx.remediation_manager.clear_pending(pod_key)
            # Also remove stale prediction to prevent memory leak
            if ctx.predictor is not None and pod_key in ctx.predictor._predictions:
                del ctx.predictor._predictions[pod_key]
            # Clean up LLM analysis cache entry for removed pod
            ctx.llm_analysis_cache.pop(pod_key, None)
            ctx.llm_in_flight.discard(pod_key)

    # In degraded mode with no new data, populate pod_items from existing state
    if collector_state == "DEGRADED" and not pod_items:
        logger.info("Reconcile #%d: DEGRADED mode - running predictions on %d last-known pods",
                    cycle, len(ctx.state_model.pods))
        for pod_key, tracker in ctx.state_model.pods.items():
            # Reconstruct a minimal pod dict for processing
            parts = pod_key.split("/", 1)
            if len(parts) == 2:
                ns, name = parts
                pod_items.append({
                    "metadata": {
                        "namespace": ns,
                        "name": name,
                        "creationTimestamp": "",
                    },
                    "spec": {
                        "nodeName": getattr(tracker, "node_name", "unknown"),
                    },
                    "status": {
                        "phase": "Unknown",
                        "containerStatuses": [{
                            "ready": getattr(tracker, "container_ready", True),
                            "restartCount": getattr(tracker, "restart_count", 0),
                        }],
                    },
                    "_degraded_mode": True,  # Marker for degraded mode processing
                })

    return PreparedCycle(current_keys=current_keys, pod_items=pod_items)


# ── Phase 3: per-pod evaluation loop ────────────────────────────────────────


def evaluate_pods(
    ctx: ReconcileContext,
    cycle: int,
    runtime: str,
    pod_items: List[dict],
    pod_metrics: Dict[str, dict],
    node_conditions: Dict[str, dict],
    current_keys: Set[str],
) -> PodLoopResult:
    """Run the per-pod OPA gate -> state update -> predict -> remediate loop."""
    del current_keys  # kept for signature symmetry; loop uses ctx state
    out = PodLoopResult()
    pods_tracked = 0
    at_risk_count = 0
    cycle_plans: dict = {}
    actions_executed_this_cycle = 0

    for pod in pod_items:
        # Handle degraded mode pods (use existing tracker data)
        if pod.get("_degraded_mode"):
            ns = pod.get("metadata", {}).get("namespace", "default")
            name = pod.get("metadata", {}).get("name", "unknown")
            pod_key = f"{ns}/{name}"
            node_name = pod.get("spec", {}).get("nodeName", "unknown")

            # Get existing tracker to use its data
            tracker = ctx.state_model.pods.get(pod_key)
            if tracker is None:
                # Skip if no tracker exists
                continue

            # Use existing tracker values (no new collection in degraded mode)
            restart_count = getattr(tracker, "restart_count", 0)
            cpu_m = getattr(tracker, "cpu_m", 0)
            memory_mib = getattr(tracker, "memory_mib", 0)
            memory_limit_mib = getattr(tracker, "memory_limit_mib", 0)
            mem_pressure = False  # No node pressure info in degraded mode
            pod_phase = "Unknown"
            container_ready = getattr(tracker, "container_ready", True)
            wait_state = None
            terminated = False
            terminated_reason = None
            pod_scheduled = True
            log_errors = 0  # Can't collect logs in degraded mode
        elif runtime == "docker":
            # Docker: use _docker_info
            ns = pod.get("metadata", {}).get("namespace", "docker")
            name = pod.get("metadata", {}).get("name", "")
            container_id = pod.get("_container_id", "")
            docker_info = pod.get("_docker_info", {})

            # For Docker, skip namespace filtering
            # but respect watch selectors if configured
            selectors = ctx.get_watch_selectors()
            if selectors.get("runtime") == "docker":
                # Check if container matches selectors
                if not ctx.matches_docker_selectors(docker_info, selectors):
                    continue

            pod_key = f"{ns}/{name}"

            # Get restart count from Docker info
            restart_count = docker_info.get("restart_count", 0)

            # Get metrics from docker stats (keyed by container id)
            metrics = pod_metrics.get(container_id[:12], {})
            cpu_m = metrics.get("cpu_m", 0)
            memory_mib = metrics.get("memory_mib", 0)

            # Get memory limit from Docker inspect info
            memory_limit_mib = ctx.get_container_memory_limit(pod.get("_docker_info", {}))

            # Docker has no node pressure (single host)
            mem_pressure = False
            node_name = "docker-host"

            # Pod-style status signals derived from the rich container_info
            # dict (get_container_status_signals expects raw `docker inspect`
            # JSON and so saw every docker pod as Running/ready).
            status_signals = docker_status_signals(docker_info)
            pod_phase = status_signals["pod_phase"]
            container_ready = status_signals["container_ready"]
            wait_state = status_signals["wait_state"]
            terminated = status_signals["terminated"]
            terminated_reason = status_signals["terminated_reason"]
            pod_scheduled = True  # Docker containers are always scheduled

            # Collect log errors from Docker logs
            log_errors = 0
            if pod_phase not in config.SKIP_LOGS_STATUSES and name:
                # Try to get logs from the container
                rc_logs, logs_output, _ = ctx.run_cmd(
                    ["docker", "logs", name, "--tail=50"],
                    timeout=5,
                )
                if rc_logs == 0:
                    log_errors = ctx.count_log_errors(logs_output)
        else:
            # Kubernetes mode
            ns = ctx.get_pod_namespace(pod)
            name = ctx.get_pod_name(pod)

            if ctx.should_skip_namespace(ns):
                continue

            # Excluded pods (KubeVirt virt-launcher VM pods or opted-out pods)
            # must not enter evaluation either: the per-pod loop iterates the
            # raw pod_items, so the collect-phase current_keys exclusion alone
            # is insufficient — without this, excluded pods are still scored,
            # LLM-analyzed and escalated (see remediation_exhausted on hourly
            # VM restart cycles).
            if ctx.is_pod_excluded(pod):
                continue

            pod_key = f"{ns}/{name}"
            restart_count = ctx.get_pod_restart_count(pod)

            # Get metrics from kubectl top
            metrics = pod_metrics.get(pod_key, {})
            cpu_m = metrics.get("cpu_m", 0)
            memory_mib = metrics.get("memory_mib", 0)

            # Get memory limit from pod spec
            memory_limit_mib = ctx.get_pod_memory_limit(pod)

            # Get node pressure
            node_name = pod.get("spec", {}).get("nodeName", "")
            mem_pressure, _disk_pressure = ctx.get_node_pressure(node_conditions, node_name)

            # Get pod status signals (phase, container ready, wait state, etc.)
            status_signals = ctx.get_pod_status_signals(pod)
            pod_phase = status_signals["pod_phase"]
            container_ready = status_signals["container_ready"]
            wait_state = status_signals["wait_state"]
            terminated = status_signals["terminated"]
            terminated_reason = status_signals["terminated_reason"]
            pod_scheduled = status_signals["pod_scheduled"]

            # Collect log errors (skip for certain statuses to save API calls)
            log_errors = 0
            if pod_phase not in config.SKIP_LOGS_STATUSES:
                container_statuses = pod.get("status", {}).get("containerStatuses", [])
                for cs in container_statuses:
                    if cs.get("ready", False) is False or cs.get("restartCount", 0) > 0:
                        rc_logs, logs_output, _ = ctx.run_cmd(
                            ["kubectl", "logs", "-n", ns, name, "--tail=50"],
                            timeout=5,
                        )
                        if rc_logs == 0:
                            log_errors = ctx.count_log_errors(logs_output)
                        break

        # Skip completed Job/CronJob pods and Succeeded pods — they don't need monitoring
        if pod_phase == "Succeeded":
            continue
        if runtime != "docker":
            owner_refs = pod.get("metadata", {}).get("ownerReferences", [])
            if any(ref.get("kind") in ("Job", "CronJob") for ref in owner_refs):
                continue

        # ─── OPA pod-security gate (fail-open) ────────────────────────────
        # Deny only on an explicit allow=False. Any error, unreachable OPA,
        # or policy_check_failed marker → allow (never break the reconcile loop).
        if ctx.policy_engine is not None:
            try:
                decision = ctx.policy_engine.check_pod_security(pod)
                if decision.get("result", {}).get("allow") is False:
                    logger.info("OPA denied pod %s: %s", pod_key,
                                 decision.get("violation"))
                    continue
            except Exception:  # noqa: BLE001
                logger.debug("OPA pod security check failed for %s", pod_key)

        # Update state model (skip for degraded mode - use existing tracker)
        if pod.get("_degraded_mode"):
            # In degraded mode, use existing tracker without updating
            tracker = ctx.state_model.pods.get(pod_key)
            if tracker is None:
                continue
        else:
            tracker = ctx.state_model.update_pod(
                namespace=ns,
                name=name,
                memory_mib=memory_mib,
                memory_limit_mib=memory_limit_mib,
                cpu_m=cpu_m,
                restart_count=restart_count,
                log_errors=log_errors,
                node_pressure=mem_pressure,
                node_name=node_name if runtime != "docker" else "docker-host",
            )
        pods_tracked += 1

        # Learn from last cycle's remediation: pair the (state, action) pairs
        # recorded then with this cycle's now-observed pod state, before this
        # cycle's remediation is itself recorded as new pending feedback.
        if ctx.remediation_manager is not None:
            ctx.remediation_manager.record_observations({pod_key: tracker.state})

        # Generate prediction
        if ctx.predictor is not None:
            markov_state = tracker.state
            markov_p_critical = 0.0
            markov_p_failed = 0.0
            if ctx.state_model.markov:
                transitions = ctx.state_model.markov.predict(markov_state, steps=1)
                markov_p_critical = transitions.get("CRITICAL", 0.0)
                markov_p_failed = transitions.get("FAILED", 0.0)

            # Calculate restart rate per hour from pod age (not raw count)
            ct = pod.get("metadata", {}).get("creationTimestamp", "")
            if ct:
                try:
                    pod_age_s = (datetime.now(timezone.utc) - datetime.fromisoformat(
                        ct.replace("Z", "+00:00")
                    )).total_seconds()
                except ValueError:
                    pod_age_s = 3600
            else:
                pod_age_s = 3600
            restart_rate = restart_count / max(pod_age_s / 3600, 1.0)  # restarts per hour
            log_error_rate = log_errors / 60.0  # Approximate per-minute rate

            # Query knowledge graph for blast radius (0 if KG unavailable)
            blast_radius = 0
            if ctx.kg_client is not None:
                try:
                    blast_radius = ctx.kg_client.get_blast_radius(pod_key)
                except Exception:  # noqa: BLE001
                    blast_radius = 0

            result = ctx.predictor.predict(
                pod_key=pod_key,
                memory_pct=tracker.memory_pct,
                memory_trend_mib_per_min=tracker.memory_trend,
                memory_limit_mib=memory_limit_mib,
                memory_mib=memory_mib,
                cpu_pct=tracker.cpu_pct,
                restart_rate_per_hr=restart_rate,
                log_error_rate_per_min=log_error_rate,
                node_memory_pressure=mem_pressure,
                node_disk_pressure=False,
                markov_state=markov_state,
                markov_p_critical=markov_p_critical,
                markov_p_failed=markov_p_failed,
                cpu_trend_m_per_min=tracker.cpu_trend,
                memory_anomaly_score=tracker.memory_anomaly_score,
                cpu_anomaly_score=tracker.cpu_anomaly_score,
                blast_radius=blast_radius,
                pod_phase=pod_phase,
                container_ready=container_ready,
                wait_state=wait_state,
                terminated=terminated,
                terminated_reason=terminated_reason,
                pod_scheduled=pod_scheduled,
            )

            if result.risk_score >= ctx.predictor.risk_threshold:
                at_risk_count += 1
                # Surface the planned intervention sequence for operator visibility.
                # This only *plans* (no execution); the REM-8 block below executes
                # via plan_and_execute using the same model and pod state.
                if ctx.remediation_manager is not None:
                    try:
                        cycle_plans[pod_key] = ctx.remediation_manager.plan_remediation(
                            tracker.state, config.PLANNING_HORIZON
                        )
                    except Exception:  # noqa: BLE001
                        pass
                # Notify operators of the planned remediation (throttled per pod).
                if ctx.remediation_manager is not None and ctx.notifier is not None:
                    try:
                        plan = cycle_plans.get(pod_key)
                        if plan and any(s["action"] != "none" for s in plan["steps"]):
                            actions = [s["action"] for s in plan["steps"] if s["action"] != "none"]
                            sig = ",".join(actions)
                            last = ctx.last_plan_notify.get(pod_key)
                            now_ts = time.time()
                            if last is None or last[0] != sig or (now_ts - last[1]) > 300:
                                ctx.notifier.notify(
                                    alert_type="plan",
                                    pod_name=pod_key,
                                    risk_score=result.risk_score,
                                    action_taken=";".join(actions),
                                    details=(
                                        f"planned horizon={plan['horizon']}: {actions} | "
                                        + " ".join(
                                            s["rationale"]
                                            for s in plan["steps"]
                                            if s["action"] != "none" and s.get("rationale")
                                        )
                                    ),
                                )
                                ctx.last_plan_notify[pod_key] = (sig, now_ts)
                    except Exception:  # noqa: BLE001
                        pass
                # Use WARNING for high risk, INFO for moderate risk to reduce log spam
                if result.risk_score >= 0.7:
                    logger.warning(
                        "Pod %s at risk: score=%.2f ttf=%s state=%s mem=%.1f%% cpu=%.1f%%",
                        pod_key, result.risk_score, result.ttf_minutes, markov_state,
                        tracker.memory_pct, tracker.cpu_pct,
                    )
                else:
                    logger.info(
                        "Pod %s elevated risk: score=%.2f state=%s",
                        pod_key, result.risk_score, markov_state,
                    )

                # LLM analysis for high-risk pods — runs in a background
                # thread so a slow/unreachable LLM never blocks the reconcile
                # loop. Capped to MAX_PODS_PER_CYCLE concurrent analyses.
                if (ctx.llm_analyzer is not None
                        and result.risk_score >= config.LLM_ANALYSIS_THRESHOLD
                        and len(ctx.llm_in_flight) < config.MAX_PODS_PER_CYCLE):
                    now_ts = time.time()
                    last_analysis = ctx.llm_analysis_cache.get(pod_key, 0)
                    if (now_ts - last_analysis >= config.LLM_ANALYSIS_TTL
                            and pod_key not in ctx.llm_in_flight):
                        ctx.llm_analysis_cache[pod_key] = now_ts
                        ctx.llm_in_flight.add(pod_key)

                        _spawn_llm_analysis(ctx, pod_key, ns, restart_count,
                                            result, markov_state, tracker,
                                            pod_phase)

                # ─── Remediation (REM-8) ────────────────────────────────
                results: list = []
                if ctx.remediation_manager is not None:
                    # Circuit breaker: cap actions executed this reconcile cycle.
                    remaining = (
                        max(0, config.MAX_ACTIONS_PER_RECONCILE - actions_executed_this_cycle)
                        if config.MAX_ACTIONS_PER_RECONCILE > 0
                        else None
                    )
                    # pod_state already has cpu_trend/memory_trend via Kalman filters
                    try:
                        if config.PLANNING_ENABLED:
                            # Default decision path: POMDP planner selects the
                            # intervention sequence that minimises expected
                            # P(FAILED); each step is still gated by SafetyPolicy,
                            # dry_run, and the action's own should_execute() guard.
                            results = ctx.remediation_manager.plan_and_execute(
                                current_state=tracker.state,
                                target=pod_key,
                                pod_state=tracker,
                                prediction=result,
                                risk_score=result.risk_score,
                                horizon=config.PLANNING_HORIZON,
                                max_actions=remaining,
                            )
                        else:
                            # Fallback: per-action reactive remediation.
                            results = ctx.remediation_manager.evaluate(
                                tracker, result, result.risk_score
                            )
                        at_risk = result.risk_score >= config.REMEDIATION_RISK_THRESHOLD
                        all_none = bool(
                            results
                            and all(getattr(r, "action", "none") == "none" for r in results)
                        )
                        # Crash-loop policy (#watch): if the planner suppressed an
                        # at-risk pod and the operator prefers visible churn over
                        # silence, force the reactive restart path (still respects
                        # SafetyPolicy cooldown + dry_run).
                        if config.CRASHLOOP_POLICY == "persist_restart" and at_risk and all_none:
                            try:
                                results = ctx.remediation_manager.evaluate(
                                    tracker, result, result.risk_score
                                )
                            except Exception as e:  # noqa: BLE001
                                logger.error("Forced remediation failed for %s: %s", pod_key, e)
                        for rem_result in results:
                            logger.info(
                                "Remediation: %s on %s — %s",
                                rem_result.action,
                                rem_result.target,
                                rem_result.message,
                            )
                            # Send notification for each remediation action
                            if ctx.notifier is not None and rem_result.success:
                                ctx.notifier.notify(
                                    alert_type="remediation",
                                    pod_name=pod_key,
                                    risk_score=result.risk_score,
                                    action_taken=rem_result.action,
                                    details=rem_result.message,
                                )
                        # Futile-remediation / crash-loop escalation (#watch): a
                        # pod that stays at-risk while the planner decides 'none'
                        # every cycle is escalated to a human instead of going
                        # silent. CRASHLOOP_POLICY=persist_restart avoids this by
                        # keeping the reactive restart path active.
                        # Closed loop: verification-hold futility is accrued by
                        # the manager itself (hold → note_remediation_outcome →
                        # remediation_exhausted), so a pod with a held result
                        # this cycle is skipped here to avoid double-counting.
                        _verify_held = any(
                            getattr(r, "outcome", "") == "held" for r in results
                        )
                        escalated, futile_n = (False, 0)
                        if not _verify_held:
                            escalated, futile_n = ctx.remediation_manager.note_remediation_outcome(
                                pod_key, at_risk, [getattr(r, "action", "none") for r in results]
                            )
                        if escalated:
                            logger.warning(
                                "Remediation exhausted for %s: %d consecutive at-risk "
                                "cycles with no beneficial action (likely a "
                                "deterministic crash-loop). Escalating to human.",
                                pod_key, futile_n,
                            )
                            if ctx.notifier is not None:
                                ctx.notifier.notify(
                                    alert_type="remediation_exhausted",
                                    pod_name=pod_key,
                                    risk_score=result.risk_score,
                                    action_taken="escalate",
                                    details=(
                                        f"{futile_n} consecutive at-risk cycles; planner: "
                                        f"restart yields no durable P(FAILED) reduction"
                                    ),
                                )
                    except Exception as e:  # noqa: BLE001
                        logger.error("Remediation evaluation failed for %s: %s", pod_key, e)

                # Shadow-mode validation: log the planner's recommendation
                # vs the actions that actually executed. Purely additive —
                # no execution, no state change. Only runs if a harness is set.
                if ctx.shadow_harness is not None:
                    try:
                        ctx.shadow_harness.observe(
                            pod_key,
                            tracker.state,
                            [r.action for r in results],
                            risk_score=result.risk_score,
                            prediction=result,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.debug("Shadow observe failed for %s: %s", pod_key, e)

                # Track executed actions for the circuit breaker.
                actions_executed_this_cycle += sum(
                    1 for r in results if getattr(r, "success", False)
                )

                # P6 bounded autonomy: destructive executions and review
                # deferrals are emitted to the notification channel (throttled
                # per pod), never silent.
                if ctx.notifier is not None:
                    for r in results:
                        kind = None
                        if getattr(r, "outcome", "") == "review_required":
                            kind = "review_required"
                        elif getattr(r, "action", "") == "node_cordon" and getattr(r, "success", False):
                            kind = "destructive_action"
                        if kind is None:
                            continue
                        try:
                            ts = time.time()
                            last = ctx.last_autonomy_notify.get(pod_key)
                            if last is None or last[0] != kind or (ts - last[1]) > 600:
                                ctx.notifier.notify(
                                    alert_type=kind,
                                    pod_name=pod_key,
                                    risk_score=result.risk_score,
                                    action_taken=getattr(r, "action", ""),
                                    details=(getattr(r, "message", "") or kind)[:400],
                                )
                                ctx.last_autonomy_notify[pod_key] = (kind, ts)
                        except Exception as e:  # noqa: BLE001
                            logger.debug("Autonomy notify failed for %s: %s", pod_key, e)

                # Closed loop: drain verification/hold/revert/canary outcome
                # alerts queued by the remediation manager (throttled per
                # target+kind like the autonomy alerts above; suppressed
                # repeats are dropped, not re-queued).
                if ctx.notifier is not None and getattr(
                    ctx.remediation_manager, "outcome_alerts", None
                ):
                    for ev in ctx.remediation_manager.outcome_alerts:
                        try:
                            ts = time.time()
                            throttle_key = f"{ev.get('target', '')}|{ev.get('kind', '')}"
                            last = ctx.last_autonomy_notify.get(throttle_key)
                            if last is None or (ts - last[1]) > 600:
                                ctx.notifier.notify(
                                    alert_type=ev.get("kind", "remediation"),
                                    pod_name=ev.get("target", pod_key),
                                    risk_score=ev.get("risk_score"),
                                    action_taken=ev.get("action", ""),
                                    details=(ev.get("details", "") or "")[:400],
                                )
                                ctx.last_autonomy_notify[throttle_key] = (
                                    ev.get("kind", ""), ts,
                                )
                        except Exception as e:  # noqa: BLE001
                            logger.debug(
                                "Outcome notify failed for %s: %s", ev.get("kind"), e
                            )
                    ctx.remediation_manager.outcome_alerts.clear()

    out.pods_tracked = pods_tracked
    out.at_risk_count = at_risk_count
    out.cycle_plans = cycle_plans
    out.actions_executed = actions_executed_this_cycle
    return out


def _spawn_llm_analysis(
    ctx: ReconcileContext,
    pod_key: str,
    ns: str,
    restart_count: int,
    result: Any,
    markov_state: str,
    tracker: Any,
    pod_phase: str,
) -> None:
    """Spawn the background LLM-analysis thread for one at-risk pod."""
    # Capture loop-local values for the thread
    _risk = result.risk_score
    _ttf = result.ttf_minutes
    _conf = result.confidence
    _mstate = markov_state
    _mempct = tracker.memory_pct
    _cpupct = tracker.cpu_pct
    _memtrend = tracker.memory_trend
    _cputrend = tracker.cpu_trend
    _nsc = ns
    _rest = restart_count
    _phase = pod_phase
    _policy_engine = ctx.policy_engine
    _llm_analyzer = ctx.llm_analyzer
    _llm_in_flight = ctx.llm_in_flight

    def _run_llm_analysis(
        _pk=pod_key, _risk=_risk, _ttf=_ttf, _conf=_conf,
        _mstate=_mstate, _mempct=_mempct, _cpupct=_cpupct,
        _memtrend=_memtrend, _cputrend=_cputrend,
        _nsc=_nsc, _rest=_rest, _phase=_phase,
    ) -> None:
        # Counters live on the main module (tests assert on them after
        # reconcile returns) — resolve via a deferred import, which keeps
        # the original ``global _llm_calls`` runtime semantics.
        from predictive_agent import main as _main
        try:
            # OPA LLM-usage gate (fail-open): deny only on
            # explicit allow_llm=False; errors/OPA-down → allow.
            if _policy_engine is not None:
                llm_request = {"model": config.OLLAMA_MODEL,
                               "pod_key": _pk,
                               "risk_score": _risk}
                try:
                    decision = _policy_engine.check_llm_usage(llm_request)
                    if decision.get("result", {}).get("allow_llm") is False:
                        logger.info("OPA denied LLM analysis for %s: %s",
                                     _pk, decision.get("violation"))
                        return
                except Exception as e:  # noqa: BLE001
                    logger.debug("OPA LLM gate failed for %s: %s", _pk, e)
            analysis = _llm_analyzer.analyze(
                issue=(
                    f"Pod {_pk} at risk "
                    f"(score={_risk:.2f}, state={_mstate})"
                ),
                context=(
                    f"namespace={_nsc}, memory={_mempct:.1f}%, "
                    f"cpu={_cpupct:.1f}%, "
                    f"memory_trend={_memtrend:.2f} MiB/min, "
                    f"cpu_trend={_cputrend:.2f} m/min, "
                    f"restarts={_rest}, "
                    f"ttf={_ttf}min, "
                    f"phase={_phase}"
                ),
                prediction={
                    "risk_score": _risk,
                    "ttf_minutes": _ttf,
                    "confidence": _conf,
                    "markov_state": _mstate,
                    "memory_trend": _memtrend,
                },
            )
            with _main._llm_lock:
                _main._llm_calls += 1
            logger.info(
                "LLM analysis for %s: severity=%s action=%s",
                _pk,
                analysis.get("severity", "unknown"),
                analysis.get("action", "none"),
            )
        except Exception as e:  # noqa: BLE001
            with _main._llm_lock:
                _main._llm_errors += 1
            logger.debug("LLM analysis failed for %s: %s", _pk, e)
        finally:
            _llm_in_flight.discard(_pk)

    threading.Thread(
        target=_run_llm_analysis, daemon=True
    ).start()


# ── Phase 4: calibration drift guardrail ────────────────────────────────────


def apply_calibration_guardrail(ctx: ReconcileContext) -> None:
    """Flip the planner to reactive fallback on calibration drift (and back)."""
    if ctx.remediation_manager is None or ctx.remediation_manager.calibration_monitor is None:
        return
    monitor = ctx.remediation_manager.calibration_monitor
    drift = monitor.drift_detected()
    was_degraded = getattr(ctx.remediation_manager, "degraded", False)
    ctx.remediation_manager.degraded = drift
    if drift:
        # Drift-aware reweighting (#10): unlearn potentially stale/misleading
        # transitions so the planner stops acting on outdated dynamics.
        try:
            removed = ctx.remediation_manager.decision_model.decay(config.DRIFT_DECAY_FACTOR)
            logger.info("Calibration drift: decayed %.1f transition-weight", removed)
        except Exception as e:  # noqa: BLE001
            logger.warning("Drift decay failed: %s", e)
        if not was_degraded and ctx.notifier is not None:
            ctx.notifier.notify(
                alert_type="calibration_drift",
                pod_name="decision-layer",
                risk_score=monitor.brier(),
                action_taken="rollback-to-reactive" if config.AUTO_ROLLBACK_ON_DRIFT else "monitor-only",
                details=(
                    f"calibration drift: brier={monitor.brier():.3f} "
                    f"over_rate={monitor.over_rate():.2f} status={monitor.status()}"
                ),
            )
    elif not drift and was_degraded:
        logger.info("Calibration recovered (status=%s); re-enabling planner", monitor.status())


# ── Phase 5: continual learning (decay + Markov transition recording) ──────


def apply_continual_learning(ctx: ReconcileContext, decay_counter: int) -> int:
    """Apply periodic model decay and record Markov transitions.

    Returns the updated decay counter (caller stores it back on main).
    """
    if ctx.remediation_manager is not None and config.MODEL_DECAY_INTERVAL > 0:
        decay_counter += 1
        if decay_counter >= config.MODEL_DECAY_INTERVAL:
            try:
                removed = ctx.remediation_manager.decision_model.decay(config.MODEL_DECAY_FACTOR)
                logger.info("Continual-learning decay removed %.1f transition-weight", removed)
            except Exception as e:  # noqa: BLE001
                logger.warning("Model decay failed: %s", e)
            decay_counter = 0

    # Update Markov chain
    if ctx.state_model.markov:
        for pod_key, tracker in ctx.state_model.pods.items():
            ctx.state_model.markov.record_transition(tracker.prev_state, tracker.state)

    return decay_counter


# ── Phase 6: persist state ──────────────────────────────────────────────────


def persist_state(ctx: ReconcileContext, state_saves: int) -> int:
    """Persist Markov/decision/Q-table/prediction state.

    Returns the updated save counter (caller stores it back on main).
    """
    if ctx.state_store is None:
        return state_saves
    try:
        ctx.state_store.save_markov(ctx.state_model.markov)
        state_saves += 1
        if ctx.remediation_manager is not None:
            try:
                ctx.state_store.save_decision_model(
                    ctx.remediation_manager.decision_model.to_action_dict()
                )
            except Exception as e:  # noqa: BLE001
                logger.error("Failed to persist decision model: %s", e)
                _get_self_health(ctx)[0].report(
                    "persistence",
                    "decision model not persisted",
                    f"save_decision_model failed: {e}",
                )
            try:
                if ctx.remediation_manager.rl_policy is not None:
                    ctx.state_store.save_q_table(ctx.remediation_manager.rl_policy.to_dict())
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to persist Q-table: %s", e)
        if ctx.predictor is not None:
            predictions_data = [
                {
                    "pod_key": p.pod_key,
                    "risk_score": p.risk_score,
                    "ttf_minutes": p.ttf_minutes,
                    "confidence": p.confidence,
                    "markov_state": p.markov_state,
                    "memory_trend": p.memory_trend,
                    "cpu_trend": p.cpu_trend,
                    "memory_pct": p.memory_pct,
                    "cpu_pct": p.cpu_pct,
                }
                for p in ctx.predictor._predictions.values()
            ]
            ctx.state_store.save_predictions(predictions_data)
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to persist state: %s", e)
        _get_self_health(ctx)[0].report(
            "persistence",
            "state not persisted",
            f"save_state/save_predictions failed: {e} "
            f"(file={config.STATE_MODEL_FILE})",
        )
    return state_saves


# ── Phase 7: backtest recording ─────────────────────────────────────────────


def record_backtest_predictions(ctx: ReconcileContext) -> None:
    """Pair this cycle's predictions with observed pod states for the backtester."""
    if ctx.backtester is None or ctx.predictor is None:
        return
    for pod_key, tracker in ctx.state_model.pods.items():
        pred = ctx.predictor._predictions.get(pod_key)
        if pred is not None:
            try:
                ctx.backtester.record_prediction(
                    pod_key=pod_key,
                    risk_score=pred.risk_score,
                    ttf_minutes=pred.ttf_minutes,
                    confidence=pred.confidence,
                    markov_state=pred.markov_state,
                    memory_pct=pred.memory_pct,
                    cpu_pct=pred.cpu_pct,
                    memory_trend=pred.memory_trend,
                    cpu_trend=pred.cpu_trend,
                )
                # Record outcome: failure if pod is in unhealthy state or has restarts
                actual_failure = (
                    tracker.state in ("CRITICAL", "FAILED")
                    or tracker.restart_count > 0
                )
                ctx.backtester.record_outcome(
                    pod_key=pod_key,
                    actual_failure=actual_failure,
                    restart_count=tracker.restart_count,
                    state=tracker.state,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("Backtester recording failed for %s: %s", pod_key, e)


# ── Phase 8: summary finalization + server context sync ────────────────────


def finalize_summary(
    ctx: ReconcileContext,
    cycle: int,
    pods_tracked: int,
    at_risk_count: int,
    cycle_plans: Dict[str, Any],
    timestamp: str,
    reconcile_duration: float,
    llm_calls: int,
    llm_errors: int,
    state_saves: int,
) -> Dict[str, Any]:
    """Assemble the reconcile summary and sync the server context."""
    summary = {
        "predictions": [
            {
                "pod_key": p.pod_key,
                "risk_score": p.risk_score,
                "ttf_minutes": p.ttf_minutes,
            }
            for p in (ctx.predictor._predictions.values() if ctx.predictor else [])
        ],
        "state": ctx.state_model.to_dict() if ctx.state_model else {},
        "timestamp": timestamp,
        "pods_tracked": pods_tracked,
        "at_risk_count": at_risk_count,
        "reconcile_count": cycle,
    }

    # ─── Add remediation stats to summary ─────────────────────────────
    if ctx.remediation_manager is not None:
        summary["remediation"] = ctx.remediation_manager.get_stats()

    if at_risk_count > 0:
        logger.warning("Reconcile #%d: %d pods at risk", cycle, at_risk_count)
    else:
        logger.info("Reconcile #%d: %d pods tracked, 0 at risk", cycle, pods_tracked)

    # ─── Update server context for Prometheus metrics ─────────────────
    try:
        from predictive_agent.server import _context as _server_context
        _server_context.reconcile_count = cycle
        _server_context.reconcile_duration = round(reconcile_duration, 3)
        _server_context.at_risk_count = at_risk_count
        _server_context.llm_calls = llm_calls
        _server_context.llm_errors = llm_errors
        _server_context.state_saves = state_saves
        _server_context.remediation_plans = cycle_plans
    except Exception:  # noqa: BLE001 - server may not be running
        pass

    return summary
