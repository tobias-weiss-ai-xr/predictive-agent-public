"""Component initialization for the operator startup sequence.

Extracted from main.py (behavior preserved, including initialization
order and the double ``detect_runtime`` call in the remediation block).
All constructors/factories that tests patch on the main module are
received as explicit hooks — main passes its current module attributes
at call time (late binding).
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

from predictive_agent import config
from predictive_agent.actions import (
    DeploymentScaleAction,
    NodeCordonAction,
    PodRestartAction,
    ResourceTunerAction,
    RightSizeAction,
    RolloutRestartAction,
)

logger = logging.getLogger("predictive-agent")


@dataclass
class BootstrapHooks:
    """Constructors and factories resolved from main's module namespace."""

    detect_runtime: Callable[[], str]
    get_watch_selectors: Callable[[], dict]
    state_store_cls: Callable[..., Any]          # main.StateStore (patchable)
    predictor_cls: Callable[..., Any]            # main.Predictor (patchable)
    state_model_cls: Callable[..., Any]          # main.StateModel (patchable)
    create_remediation_manager: Callable[[], Any]
    docker_socket_client_cls: Callable[..., Any]  # main.DockerSocketClient
    create_notifier: Callable[[], Any]
    get_kg_client: Callable[[], Any]
    get_policy_engine: Callable[..., Any]


@dataclass
class ComponentSet:
    """All long-lived components created during startup."""

    state_model: Any
    predictor: Any
    state_store: Any
    remediation_manager: Any
    notifier: Any
    kg_client: Optional[Any]
    backtester: Any
    policy_engine: Optional[Any]
    llm_analyzer: Optional[Any]


def initialize_components(runtime: str, hooks: BootstrapHooks) -> ComponentSet:
    """Initialize every long-lived component in the original main() order."""
    # Initialize state
    state_model = hooks.state_model_cls()
    predictor = hooks.predictor_cls(risk_threshold=config.PREDICTION_RISK_THRESHOLD)
    state_store = hooks.state_store_cls(
        state_model_file=config.STATE_MODEL_FILE,
        predictions_file=config.PREDICTIONS_FILE,
    )

    # Load persisted state
    try:
        state_model.markov = state_store.load_markov()
        logger.info("Loaded Markov chain state from %s", config.STATE_MODEL_FILE)
    except Exception as e:  # noqa: BLE001 - missing/corrupt state must not abort startup
        logger.warning("Could not load Markov state: %s", e)

    # ─── Initialize remediation (REM-8) ───────────────────────────────
    remediation_manager = hooks.create_remediation_manager()
    # In Docker-native mode the restart action talks to the Docker socket; in
    # Kubernetes mode it shells out to kubectl. Both are registered under the
    # planner name "pod_restart" so the decision layer is runtime-agnostic.
    if hooks.detect_runtime() == "docker":
        from predictive_agent.actions import ContainerRestartAction
        _docker_restart = ContainerRestartAction(
            hooks.docker_socket_client_cls(config.DOCKER_SOCKET)
        )
        _docker_restart.name = "pod_restart"
        remediation_manager.register_action(_docker_restart)
    else:
        remediation_manager.register_action(PodRestartAction())
    remediation_manager.register_action(NodeCordonAction())
    remediation_manager.register_action(RightSizeAction())
    remediation_manager.register_action(RolloutRestartAction())
    remediation_manager.register_action(DeploymentScaleAction())
    remediation_manager.register_action(ResourceTunerAction())
    # Register ContainerRestartAction only in Docker runtime
    if runtime == "docker":
        remediation_manager.register_action(ContainerRestartAction())
    logger.info(
        "Remediation initialized: dry_run=%s, threshold=%.1f, actions=%s",
        remediation_manager.dry_run,
        remediation_manager.risk_threshold,
        remediation_manager.get_stats()["registered_actions"],
    )

    # POMDP decision model: share the live Markov chain as its base dynamics so
    # the planner uses the operator's learned transitions, and reload any
    # action-conditioned transitions previously learned from real outcomes.
    from predictive_agent import main as _main
    decision_model = _main.ActionAwareTransitionModel(base_chain=state_model.markov)
    loaded_dm = state_store.load_decision_model()
    if loaded_dm:
        decision_model.load_action_counts(loaded_dm)
        logger.info("Loaded decision model action transitions from state store")
    remediation_manager.decision_model = decision_model

    # Calibration monitor: track predicted P(FAILED) vs realised outcome so the
    # safety guardrail can detect drift in production. On by default; only
    # disabled if it cannot be constructed.
    try:
        from predictive_agent.calibration import CalibrationMonitor

        remediation_manager.calibration_monitor = CalibrationMonitor(
            window=config.CALIBRATION_WINDOW,
            brier_alert=config.CALIBRATION_BRIER_ALERT,
            over_rate_alert=config.CALIBRATION_OVER_RATE_ALERT,
            min_samples=config.CALIBRATION_MIN_SAMPLES,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Calibration monitor disabled: %s", e)

    # Model-free Q-learning policy (#8): learns Q(state, action) from the same
    # real outcomes as the transition model, via record_observations. Loaded
    # from / persisted to the state store so it accumulates across reconciles.
    try:
        from predictive_agent.rl import QLearningPolicy

        rl_policy = QLearningPolicy(
            model=decision_model,
            action_costs=remediation_manager.action_costs,
            failure_cost=remediation_manager.failure_cost,
        )
        loaded_q = state_store.load_q_table()
        if loaded_q:
            rl_policy.load(loaded_q)
            logger.info("Loaded Q-learning policy Q-table from state store")
        remediation_manager.rl_policy = rl_policy
    except Exception as e:  # noqa: BLE001
        logger.warning("Q-learning policy disabled: %s", e)

    # ─── Initialize notifications (REM-6) ──────────────────────────────
    notifier = hooks.create_notifier()
    logger.info("Notification manager initialized")

    # ─── Initialize knowledge graph integration ────────────────────────
    kg_client = hooks.get_kg_client()
    if kg_client and kg_client.health():
        from predictive_agent.kg_integration import DGRAPH_URL
        logger.info("Knowledge graph connected at %s", DGRAPH_URL)
    else:
        logger.info("Knowledge graph not available (blast radius will be 0)")
        kg_client = None

    # ─── Initialize backtester ────────────────────────────────────────
    from predictive_agent import main as _main
    backtester = _main.Backtester()
    logger.info("Backtester initialized (history: %d predictions, %d outcomes)",
                backtester.get_prediction_count(), backtester.get_outcome_count())

    # OPA policy engine (opt-in, fail-open). Off by default via OPA_ENABLED
    # so deployments without an OPA sidecar are unaffected.
    policy_engine: Optional[Any] = None
    try:
        policy_engine = hooks.get_policy_engine(
            opa_host=config.OPA_HOST, opa_port=config.OPA_PORT,
            use_tls=config.OPA_USE_TLS, enabled=config.OPA_ENABLED,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to initialize OPA policy engine: %s", e)
        policy_engine = None
    if policy_engine is not None:
        logger.info("OPA policy engine: enabled=%s", config.OPA_ENABLED)

    # ─── Initialize LLM analyzer (REM-7) ──────────────────────────────
    llm_analyzer = _initialize_llm_analyzer()

    return ComponentSet(
        state_model=state_model,
        predictor=predictor,
        state_store=state_store,
        remediation_manager=remediation_manager,
        notifier=notifier,
        kg_client=kg_client,
        backtester=backtester,
        policy_engine=policy_engine,
        llm_analyzer=llm_analyzer,
    )


def _initialize_llm_analyzer() -> Optional[Any]:
    """Build the LLM analyzer from config; None on unknown backend/error."""
    try:
        from predictive_agent.llm import LLMAnalyzer, LLMBackend
        backend_str = config.LLM_BACKEND.lower()
        if backend_str == "ollama":
            analyzer: Optional[Any] = LLMAnalyzer(
                backend=LLMBackend.OLLAMA,
                url=config.OLLAMA_URL,
                model=config.OLLAMA_MODEL,
                timeout=config.OLLAMA_TIMEOUT,
            )
            logger.info(
                "LLM analyzer initialized: backend=ollama, model=%s, url=%s",
                config.OLLAMA_MODEL, config.OLLAMA_URL,
            )
            return analyzer
        elif backend_str == "openai":
            analyzer = LLMAnalyzer(
                backend=LLMBackend.OPENAI,
                url=config.OPENAI_API_URL,
                model=config.OPENAI_MODEL,
                api_key=config.OPENAI_API_KEY,
                # Reasoning models routinely exceed 60s under load; 300s is
                # generous headroom (analysis runs in a background thread).
                timeout=int(os.environ.get("OPENAI_TIMEOUT", "300")),
            )
            logger.info("LLM analyzer initialized: backend=openai, model=%s", config.OPENAI_MODEL)
            return analyzer
        elif backend_str in ("saia", "tud"):
            api_url = config.SAIA_API_URL if backend_str == "saia" else config.TUD_API_URL
            api_key = config.SAIA_API_KEY if backend_str == "saia" else config.TUD_API_KEY
            api_model = config.SAIA_MODEL if backend_str == "saia" else config.TUD_MODEL
            backend = LLMBackend.SAIA if backend_str == "saia" else LLMBackend.TUD
            analyzer = LLMAnalyzer(
                backend=backend,
                url=api_url,
                model=api_model,
                api_key=api_key,
                # Reasoning models on direct vLLM/DSpark routinely exceed 60s
                # under JIT/load; 300s matches OPENAI_TIMEOUT (background thread).
                timeout=int(os.environ.get("LLM_TIMEOUT", "300")),
            )
            logger.info("LLM analyzer initialized: backend=%s, model=%s", backend_str, api_model)
            return analyzer
        elif backend_str == "jevlocal":
            analyzer = LLMAnalyzer(
                backend=LLMBackend.JEVLOCAL,
                url=config.JEVI_URL,
                model=config.JEVI_MODEL,
                timeout=config.JEVI_TIMEOUT,
            )
            logger.info(
                "LLM analyzer initialized: backend=jevlocal, url=%s", config.JEVI_URL
            )
            return analyzer
        else:
            logger.warning("Unknown LLM backend '%s', LLM analysis disabled", backend_str)
            return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to initialize LLM analyzer: %s", e)
        return None
