#!/usr/bin/env python3
"""openDesk Predictive Agent v4.0 — Reconcile loop.

Collects kubectl metrics, updates Kalman filters and Markov chain state,
runs Bayesian risk prediction, and persists state to PVC.

The reconcile loop is the heart of the operator: every ``RECONCILE_INTERVAL``
seconds it collects metrics from ``kubectl top pods``, ``kubectl top nodes``,
``kubectl get pods -o json``, and ``kubectl logs``, feeds them through the
state model (Kalman trend estimation + Markov chain transitions), generates
predictions via Bayesian risk scoring, and persists everything to the state
model and predictions files on the PVC.

An HTTP server (started in a background thread) exposes /healthz, /ready,
/metrics, /status, /predictions, /state, /history, /cache, and /reanalyze
endpoints.
"""

import logging
import signal
import threading
import time
from collections.abc import Callable
from typing import Any, Dict, Optional

from predictive_agent import bootstrap, config, pod_collection
from predictive_agent import reconcile_phases as phases
from predictive_agent.actions.container_restart import DockerSocketClient
from predictive_agent.backtester import Backtester  # noqa: F401 — patch target for tests
from predictive_agent.collector import (
    collect_top_metrics,
    collect_top_nodes,
    count_log_errors,
    detect_runtime,
    discover_docker_containers,
    get_collector_state,
    get_container_status_signals,
    get_node_conditions,
    get_pod_resources,  # noqa: F401 — patch target for tests
    get_pod_status_signals,
    get_watch_selectors,
    record_collection_failure,
    record_successful_collection,
    run_cmd,
)
from predictive_agent.kg_integration import get_kg_client
from predictive_agent.notifier import NotificationManager, create_notifier_from_config
from predictive_agent.opa_integration import get_policy_engine
from predictive_agent.persistence import StateStore
from predictive_agent.pomdp import ActionAwareTransitionModel  # noqa: F401 — patch target for tests
from predictive_agent.predictor import Predictor
from predictive_agent.remediator import RemediationManager, create_remediation_manager_from_config
from predictive_agent.state_model import StateModel

logger = logging.getLogger("predictive-agent")

# ─── Global state (shared with server.py) ──────────────────────────────────
_state_model: Optional[StateModel] = None
_predictor: Optional[Predictor] = None
_state_store: Optional[StateStore] = None
_cache: Dict[str, Any] = {}
_history: list = []
_reconcile_count = 0
_last_reconcile_time: Optional[str] = None
_last_reconcile_duration: float = 0.0
# Optional shadow-mode validation harness: runs the planner observe-only
# next to the live remediation path and logs disagreements / calibration.
# Disabled by default (None) so it never affects production behavior.
_shadow_harness = None
# Optional calibration monitor (created during start_agent and attached to the
# RemediationManager); surfaced read-only via the /calibration endpoint.
_calibration_monitor = None  # type: Optional[object]
_server = None  # HTTP server handle
_remediation_manager: Optional[RemediationManager] = None
_notifier: Optional[NotificationManager] = None
_llm_calls = 0
_llm_errors = 0
_state_saves = 0
_model_decay_counter = 0  # reconciles since last continual-learning decay (#10)
_last_plan_notify: dict = {}  # pod_key -> (action_signature, timestamp) for plan alerts
_last_autonomy_notify: dict = {}  # pod_key -> (kind, timestamp) for P6 autonomy alerts
_kg_client = None
_backtester = None
_policy_engine = None
_llm_analyzer = None
_llm_analysis_cache: dict[str, float] = {}  # pod_key -> last analysis timestamp
_llm_in_flight: set = set()  # pod_keys with an active background LLM analysis
_llm_lock = threading.Lock()  # guards _llm_calls/_llm_errors counters


# ── Collection helpers (implementations in pod_collection.py) ──────────
# Re-bound here as module attributes so existing patch targets
# ("predictive_agent.main._get_pods_json", ...) keep working. The
# reconcile pipeline resolves these names through main's namespace at
# call time (late binding).
_now_iso = pod_collection.now_iso
_kubectl_available = pod_collection.kubectl_available
_get_pods_json = pod_collection.get_pods_json
_get_pod_restart_count = pod_collection.get_pod_restart_count
_get_pod_namespace = pod_collection.get_pod_namespace
_get_pod_name = pod_collection.get_pod_name
_get_pod_memory_limit = pod_collection.get_pod_memory_limit
_should_skip_namespace = pod_collection.should_skip_namespace
_is_pod_excluded = pod_collection.is_pod_excluded
_get_node_pressure = pod_collection.get_node_pressure
_matches_docker_selectors = pod_collection.matches_docker_selectors
_collect_docker_container_metrics = pod_collection.collect_docker_container_metrics
_get_container_memory_limit = pod_collection.get_container_memory_limit


def _build_context() -> "phases.ReconcileContext":
    """Bundle current globals into a phase context (late binding: the
    hooks are resolved from THIS module's namespace at call time, so
    patch("predictive_agent.main.<hook>") keeps intercepting)."""
    return phases.ReconcileContext(
        state_model=_state_model,
        predictor=_predictor,
        state_store=_state_store,
        remediation_manager=_remediation_manager,
        notifier=_notifier,
        shadow_harness=_shadow_harness,
        policy_engine=_policy_engine,
        llm_analyzer=_llm_analyzer,
        kg_client=_kg_client,
        backtester=_backtester,
        llm_analysis_cache=_llm_analysis_cache,
        llm_in_flight=_llm_in_flight,
        last_plan_notify=_last_plan_notify,
        last_autonomy_notify=_last_autonomy_notify,
        detect_runtime=detect_runtime,
        discover_docker_containers=discover_docker_containers,
        collect_docker_metrics=_collect_docker_container_metrics,
        run_cmd=run_cmd,
        collect_top_metrics=collect_top_metrics,
        collect_top_nodes=collect_top_nodes,
        get_pods_json=_get_pods_json,
        get_node_conditions=get_node_conditions,
        get_pod_namespace=_get_pod_namespace,
        get_pod_name=_get_pod_name,
        get_pod_memory_limit=_get_pod_memory_limit,
        get_pod_restart_count=_get_pod_restart_count,
        should_skip_namespace=_should_skip_namespace,
        is_pod_excluded=_is_pod_excluded,
        get_node_pressure=_get_node_pressure,
        matches_docker_selectors=_matches_docker_selectors,
        get_container_memory_limit=_get_container_memory_limit,
        get_watch_selectors=get_watch_selectors,
        get_pod_status_signals=get_pod_status_signals,
        get_container_status_signals=get_container_status_signals,
        count_log_errors=count_log_errors,
        now_iso=_now_iso,
        get_collector_state=get_collector_state,
        record_successful_collection=record_successful_collection,
        record_collection_failure=record_collection_failure,
    )


def reconcile() -> Dict[str, Any]:
    """Run one reconcile cycle.

    Collects metrics, updates state model, generates predictions, persists
    state, and returns a summary dict. The pipeline delegates each phase to
    predictive_agent.reconcile_phases; this function owns the module globals.

    Handles both Docker and Kubernetes runtimes based on configuration.

    Returns:
        Dict with keys: predictions, state, timestamp, pods_tracked,
        at_risk_count, reconcile_count.
    """
    global _reconcile_count, _last_reconcile_time, _last_reconcile_duration
    global _state_saves, _model_decay_counter

    _reconcile_count += 1
    cycle = _reconcile_count
    logger.info("Reconcile #%d: starting", cycle)

    import time as _time_mod
    _reconcile_start = _time_mod.monotonic()

    ctx = _build_context()

    # ─── Detect runtime and collect metrics ────────────────────────────
    collected = phases.collect_metrics(ctx, cycle)

    collector_state = collected.collector_state

    if _state_model is None:
        logger.warning("State model not initialized")
        return {
            "predictions": [],
            "state": {},
            "timestamp": _now_iso(),
            "pods_tracked": 0,
            "at_risk_count": 0,
            "reconcile_count": cycle,
        }

    # ─── Update state model (degraded semantics + removals) ────────────
    prepared = phases.prepare_state_model(
        ctx, cycle, collector_state, collected.current_keys, collected.pod_items
    )

    # ─── Per-pod evaluation loop (OPA gate, predict, remediate) ────────
    loop = phases.evaluate_pods(
        ctx, cycle, collected.runtime, prepared.pod_items,
        collected.pod_metrics, collected.node_conditions, prepared.current_keys,
    )

    # ─── Calibration drift -> safety guardrail / circuit-breaker rollback ──
    phases.apply_calibration_guardrail(ctx)

    # ─── Continual learning: periodic forgetting (#10) + Markov update ──
    _model_decay_counter = phases.apply_continual_learning(ctx, _model_decay_counter)

    # ─── Persist state ─────────────────────────────────────────────────
    _state_saves = phases.persist_state(ctx, _state_saves)

    # ─── Record predictions for backtesting ───────────────────────────
    phases.record_backtest_predictions(ctx)

    _last_reconcile_time = _now_iso()
    _last_reconcile_duration = _time_mod.monotonic() - _reconcile_start

    # ─── Summary + server context sync ─────────────────────────────────
    return phases.finalize_summary(
        ctx, cycle, loop.pods_tracked, loop.at_risk_count, loop.cycle_plans,
        _last_reconcile_time, _last_reconcile_duration,
        _llm_calls, _llm_errors, _state_saves,
    )


class ReconcileLoop:
    """Background reconcile loop running in a daemon thread."""

    def __init__(self, interval: int = 60):
        self.interval = interval
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._reconcile_fn: Callable = reconcile

    def start(self) -> None:
        """Start the reconcile loop in a background daemon thread."""
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the reconcile loop and wait for the thread to finish."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        """Main loop body — calls reconcile at each interval."""
        while self.running:
            start = time.monotonic()
            try:
                self._reconcile_fn()
            except Exception as e:
                logger.error("Reconcile error: %s", e)
            elapsed = time.monotonic() - start
            sleep_time = max(0, self.interval - elapsed)
            # Sleep in small increments so stop() is responsive
            slept = 0.0
            while slept < sleep_time and self.running:
                time.sleep(min(0.5, sleep_time - slept))
                slept += 0.5


def _setup_logging() -> None:
    """Configure logging based on LOG_VERBOSITY env var."""
    level = logging.INFO
    if config.LOG_VERBOSITY.lower() == "debug":
        level = logging.DEBUG
    elif config.LOG_VERBOSITY.lower() == "warn":
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )


def _start_http_server() -> Any:
    """Start the HTTP metrics/health server."""
    from predictive_agent.server import start_server

    return start_server(
        metrics_port=config.METRICS_PORT,
        health_port=config.HEALTH_PORT,
        state_model=_state_model,
        predictor=_predictor,
        cache=_cache,
        reconcile_callback=reconcile,
        history=_history,
        remediation_manager=_remediation_manager,
        notifier=_notifier,
        backtester=_backtester,
        policy_engine=_policy_engine,
    )


def _check_runtime_prerequisites() -> None:
    """Fail fast when the configured collector cannot function at all.

    ``COLLECTOR_MODE=kubectl`` with no kubectl binary means the operator would
    collect 0 pods every cycle with no errors, no warnings and a green /ready
    — silent blindness for weeks (the exact incident this gate prevents).
    Any misbuilt image therefore crash-loops loudly at startup instead.
    """
    if config.COLLECTOR_MODE == "kubectl" and pod_collection.kubectl_binary_missing():
        logger.critical(
            "COLLECTOR_MODE=kubectl but the 'kubectl' binary is NOT installed in "
            "this image. The operator cannot collect pods or metrics and would "
            "silently track 0 pods forever (state model stays empty, no warnings, "
            "/ready stays 200). Install kubectl into the image (see Dockerfile) "
            "or set COLLECTOR_MODE=docker. Failing fast."
        )
        raise SystemExit(1)


def main() -> None:
    """Main entry point — initialize state, start server, run reconcile loop."""
    global _state_model, _predictor, _state_store, _server
    global _remediation_manager, _notifier, _kg_client, _backtester, _llm_analyzer
    global _policy_engine

    _setup_logging()
    logger.info("=== %s v%s starting ===", config.OPERATOR_NAME, config.OPERATOR_VERSION)

    # Detect runtime and get watch selectors
    runtime = detect_runtime()
    if runtime == "kubernetes":
        # kubectl does not auto-discover the mounted SA token / service host;
        # without a kubeconfig every collection call fails. Bootstrap it now.
        pod_collection.ensure_incluster_kubeconfig()
    _check_runtime_prerequisites()
    selectors = get_watch_selectors()
    logger.info("Runtime: %s", runtime)

    if runtime == "docker":
        logger.info("Docker watch labels: %s", selectors.get("labels", {}))
        logger.info("Docker watch compose projects: %s", selectors.get("compose_projects", []))
        logger.info("Docker watch names: %s", selectors.get("names", []))
    else:
        logger.info("Watch namespaces: %s", selectors.get("namespaces", []))

    logger.info("Reconcile interval: %ds", config.RECONCILE_INTERVAL)
    logger.info("LLM backend: %s", config.LLM_BACKEND)

    # Initialize all long-lived components (see bootstrap.py; hooks resolve
    # from THIS module's namespace so patch targets keep working)
    components = bootstrap.initialize_components(runtime, bootstrap.BootstrapHooks(
        detect_runtime=detect_runtime,
        get_watch_selectors=get_watch_selectors,
        state_store_cls=StateStore,
        predictor_cls=Predictor,
        state_model_cls=StateModel,
        create_remediation_manager=create_remediation_manager_from_config,
        docker_socket_client_cls=DockerSocketClient,
        create_notifier=create_notifier_from_config,
        get_kg_client=get_kg_client,
        get_policy_engine=get_policy_engine,
    ))
    _state_model = components.state_model
    _predictor = components.predictor
    _state_store = components.state_store
    _remediation_manager = components.remediation_manager
    _notifier = components.notifier
    _kg_client = components.kg_client
    _backtester = components.backtester
    _policy_engine = components.policy_engine
    _llm_analyzer = components.llm_analyzer

    # Start HTTP server
    try:
        _server = _start_http_server()
        logger.info("HTTP server started on %d (metrics) and %d (health)",
                    config.METRICS_PORT, config.HEALTH_PORT)
    except Exception as e:
        logger.error("Failed to start HTTP server: %s", e)

    # Handle signals for graceful shutdown
    loop = ReconcileLoop(interval=config.RECONCILE_INTERVAL)

    def _shutdown(signum, frame):
        logger.info("Received signal %d, shutting down...", signum)
        loop.stop()
        if _server:
            _server.shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Start reconcile loop
    loop.start()
    logger.info("Reconcile loop started with interval %ds", config.RECONCILE_INTERVAL)

    # Keep main thread alive
    try:
        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        loop.stop()
        if _server:
            _server.shutdown()


if __name__ == "__main__":
    main()
