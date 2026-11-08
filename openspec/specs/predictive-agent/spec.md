## Purpose
Predictive health-monitoring operator for Kubernetes and Docker that shifts from reactive (detect crash → analyze) to predictive (detect trends → predict failure → warn before crash).

## Requirements

### Requirement: Reconcile loop
The operator runs a periodic reconcile loop driven by `RECONCILE_INTERVAL` (default 60s) that collects metrics, updates the state model, runs risk prediction, and persists state.

#### Scenario: Periodic reconcile
- **WHEN** the operator is running and `RECONCILE_INTERVAL` elapses
- **THEN** it collects current metrics, updates the Kalman/Markov state, computes risk predictions, and writes state + predictions to the persistence files

### Requirement: Metrics collection
The operator SHALL collect workload metrics from the runtime. In Kubernetes mode it uses `kubectl` (`top`/`get`/`logs`) scoped to `OPERATOR_WATCH_NAMESPACES`. In Docker mode it SHALL discover containers via the read-only Docker socket using label / compose-project / name selectors (`OPERATOR_WATCH_LABELS`, `OPERATOR_WATCH_COMPOSE_PROJECTS`, `OPERATOR_WATCH_NAMES`) and track each discovered container's CPU, memory, restart, log, and status signals.

#### Scenario: Kubernetes mode
- **WHEN** deployed in a cluster with kubeconfig access
- **THEN** it uses `kubectl` to collect pod/node metrics and logs for the watched namespaces

#### Scenario: Docker mode
- **WHEN** deployed as a Docker container with the socket mounted read-only
- **THEN** it discovers containers matching the configured label / compose-project / name selectors and tracks their metrics and status signals

### Requirement: State model (trend + transitions)
The operator maintains a state model combining a 2D Kalman filter (level + velocity) for memory/CPU trend estimation and a 6-state Markov chain for pod state-transition prediction.

#### Scenario: Trend estimation
- **WHEN** new metric samples arrive
- **THEN** the Kalman filter updates level/velocity estimates with confidence intervals

#### Scenario: State-transition prediction
- **WHEN** the Markov chain is updated with observed state changes
- **THEN** it predicts the next-state distribution for each tracked pod

### Requirement: Bayesian risk prediction
The operator computes a risk score combining six signals (memory %, trend, CPU %, restarts, logs, node pressure) using Bayesian scoring, producing predictions with uncertainty and time-to-failure.

#### Scenario: Risk scoring
- **WHEN** a pod is tracked
- **THEN** a risk score in [0,1] is produced and exposed via the predictions API

### Requirement: LLM analysis
The operator can call an LLM backend (Ollama, Saia API, or OpenAI-compatible/LiteLLM) for enhanced analysis with predictions, uncertainty, and time-to-failure.

#### Scenario: Multi-backend analysis
- **WHEN** `LLM_BACKEND` is configured
- **THEN** analysis requests are routed to the selected backend with its configured model

### Requirement: HTTP server (metrics + health)
The operator exposes an HTTP server with a metrics/API port (default 8080) and a health-probe port (default 8081).

#### Scenario: Metrics and API
- **WHEN** a client requests an API endpoint
- **THEN** it returns `/metrics` (Prometheus), `/status`, `/predictions`, `/state`, `/history`, `/reanalyze`, and `/cache`

#### Scenario: Health probe
- **WHEN** a health check queries the probe port
- **THEN** `/healthz` returns 200 (liveness) and `/ready` returns 200 (readiness)

### Requirement: State persistence
The operator persists the Markov transition matrix and prediction tracking to files on the state volume (`STATE_MODEL_FILE`, `PREDICTIONS_FILE`) so state survives restarts.

#### Scenario: Restart survival
- **WHEN** the operator restarts
- **THEN** it reloads the persisted state model and predictions

### Requirement: Packaging and deployment
The operator is built as a minimal nix image (Python 3 + kubectl + curl) and deployed either as a Docker Compose service (Zot registry `registry.example.org/predictive-agent:latest`) or as a Kubernetes Deployment with RBAC (ServiceAccount `opendesk-predictive-agent`).

#### Scenario: Docker deployment
- **WHEN** deployed via the `monitoring_agents` Ansible role
- **THEN** the container runs with the Docker socket mounted read-only and watches the configured namespaces

#### Scenario: Kubernetes deployment
- **WHEN** deployed via the k8s manifests
- **THEN** it runs with the `opendesk-predictive-agent` ServiceAccount/ClusterRole and reads cluster metrics

### Requirement: Docker workload discovery
The operator SHALL discover Docker containers to monitor without requiring Kubernetes namespaces. Discovery is driven by configurable selectors: a label key/value (`OPERATOR_WATCH_LABELS`), a compose project name (`OPERATOR_WATCH_COMPOSE_PROJECTS`), and/or explicit container-name patterns (`OPERATOR_WATCH_NAMES`). When no selector is configured in Docker mode it SHALL discover all running containers.

#### Scenario: Discover by label
- **WHEN** `OPERATOR_WATCH_LABELS` is set
- **THEN** only containers carrying the matching label are tracked

#### Scenario: Discover by compose project
- **WHEN** `OPERATOR_WATCH_COMPOSE_PROJECTS` is set
- **THEN** only containers belonging to the named compose projects are tracked

#### Scenario: No selector configured
- **WHEN** no watch selectors are configured in Docker mode
- **THEN** all running containers are discovered and tracked

#### Scenario: Container lifecycle
- **WHEN** a tracked container stops or is removed
- **THEN** it is removed from tracking and its state-model entry is finalized/cleaned up

### Requirement: Knowledge-graph blast-radius integration
The operator SHALL query the SCS knowledge graph (Dgraph) via `kg_integration` to compute the dependency blast-radius of a tracked pod (dependent pods, services, and namespaces). The computed blast-radius SHALL be fed into the Bayesian risk scorer to escalate risk for high-impact pods. If Dgraph is unavailable, the integration SHALL degrade gracefully (blast-radius = 0) and log a warning.

#### Scenario: Blast-radius escalation
- **WHEN** a pod is scored and Dgraph returns its dependency graph
- **THEN** its risk score is increased proportionally to the number of dependent workloads

#### Scenario: Dgraph unavailable
- **WHEN** the knowledge-graph endpoint is unreachable
- **THEN** blast-radius defaults to 0 and processing continues without failure

### Requirement: Remediation framework and actions
The operator SHALL provide an autonomous remediation framework (`remediator`: `RemediationAction` ABC, `RemediationManager`, `SafetyPolicy` with rate-limiting, cooldowns, and protected namespaces) and a library of actions under `actions/` (`pod_restart`, `rollout_restart`, `scale`, `node_cordon`, `right_size`, `tune_resources`, `container_restart`). Actions SHALL execute only after passing the `SafetyPolicy` and SHALL record an audit trail. `SAFETY_ENABLED=false` (dry-run) SHALL prevent any mutating action. The framework SHALL close the act–verify loop: every executed action SHALL be followed by post-action verification against the resolution criteria recorded at plan time, and unverified remediation SHALL hold and escalate rather than re-fire.

#### Scenario: Safe remediation
- **WHEN** a risk prediction exceeds the remediation threshold and the safety policy allows it
- **THEN** the matching action executes and an audit entry is recorded

#### Scenario: Protected namespace
- **WHEN** a pod resides in a protected namespace
- **THEN** no remediation action is taken regardless of risk score

#### Scenario: Post-action verification
- **WHEN** an action has executed and the configured settling window has elapsed
- **THEN** the operator re-samples at least two independent signals for the target and compares them against the resolution criteria recorded at plan time, and the remediation is recorded as verified only when improvement is sustained across the re-samples

#### Scenario: Hold on failed verification
- **WHEN** post-action verification does not confirm sustained improvement
- **THEN** the same action is not re-triggered immediately, the target enters a hold state until the hold period elapses, an escalation flag is raised to operators, and the futile-remediation counter for the target is incremented

#### Scenario: Reversible actions
- **WHEN** a reversible action's verification fails
- **THEN** the operator re-applies the recorded prior state (revert) before any retry, and logs the revert in the audit trail; an irreversible action under the same condition requires operator approval before any further attempt

#### Scenario: Canary for fleet-consequence actions
- **WHEN** an action could affect more than a bounded slice of a workload's capacity (restart or scale across the fleet)
- **THEN** the action first executes on a canary slice (a configured fraction of the target), is verified, and only then is widened to the remaining scope; a failed canary verification aborts the widen

#### Scenario: Verification surfaced to operators
- **WHEN** a remediation is verified, held, reverted, or escalated
- **THEN** an alert and a remediation life-cycle status entry are emitted so operators can audit the outcome of every executed action

### Requirement: Backtesting harness
The operator SHALL include a backtesting harness (`backtester`) that records predictions and evaluates them against actual outcomes (pod restarts, OOM kills, state transitions), computing Brier score, calibration, and time-to-failure accuracy to validate predictor quality.

#### Scenario: Prediction evaluation
- **WHEN** a prediction and its later actual outcome are recorded
- **THEN** `evaluate()` returns accuracy and calibration metrics

### Requirement: Performance optimization
The operator SHALL provide a performance-optimization module (`optimize`) offering caching configuration, batch processing, and stats collection for the LLM and predictor paths.

#### Scenario: Optimizer configuration
- **WHEN** the optimizer is configured with batching/caching enabled
- **THEN** subsequent LLM/predictor calls use the optimized paths and report stats

### Requirement: L1 cache
The operator SHALL use a hierarchical in-memory L1 cache (`l1_cache`) with TTL, request deduplication, pod-state/prediction caching, and token-usage tracking, with CPU-aware auto-disable (>95% CPU) and size-based eviction to bound memory.

#### Scenario: Cache hit
- **WHEN** a repeated LLM/prediction request arrives within TTL
- **THEN** the cached result is returned and no duplicate upstream call is made

#### Scenario: CPU pressure
- **WHEN** CPU usage exceeds 95%
- **THEN** the L1 cache auto-disables to protect the reconcile loop

### Requirement: LLM batch analysis
The operator SHALL support batched LLM analysis (`llm_batch`) to reduce duplicate calls and latency when multiple pods are analyzed in a cycle.

#### Scenario: Batched analysis
- **WHEN** several pods are queued for LLM analysis in one cycle
- **THEN** requests are batched into a single backend call where the backend supports it

### Requirement: Open Policy Agent (OPA) integration
The operator SHALL integrate with Open Policy Agent (`opa_client`, `opa_cache`, `opa_integration`) for policy-based decision governance of pod processing and LLM usage, using a stdlib-only client (no third-party deps). OPA integration is configurable via `OPA_ENABLED` (default off); when enabled, the operator gates pod processing and LLM calls through OPA policy decisions. When OPA is disabled or unreachable, checks SHALL **fail open** (allow) so policy gating never breaks the reconcile loop.

#### Scenario: OPA disabled (default)
- **WHEN** `OPA_ENABLED` is false or unset
- **THEN** all policy checks are no-ops and every pod/LLM request is allowed

#### Scenario: Policy-gated pod processing
- **WHEN** `OPA_ENABLED` is true and a pod enters the reconcile loop
- **THEN** `check_pod_security(pod)` is evaluated and the pod is skipped if the policy returns `allow=false`

#### Scenario: Policy-gated LLM usage
- **WHEN** `OPA_ENABLED` is true and an LLM analysis would run
- **THEN** `check_llm_usage(request)` is evaluated and the analysis is skipped if the policy returns `allow_llm=false`

#### Scenario: OPA unreachable (fail-open)
- **WHEN** OPA is enabled but the policy server is unreachable or errors
- **THEN** the check fails open (allow) and a violation note is recorded; processing/LLM calls continue

#### Scenario: Optional policy check
- **WHEN** OPA integration is enabled
- **THEN** policy decisions gate pod processing and LLM usage per configured Rego policies

### Requirement: Decision layer (POMDP planning)
The operator SHALL plan remediation as a sequence of interventions that minimise the expected probability of the `FAILED` state over a short horizon, using an action-conditioned transition model over the existing 6-state Markov vocabulary, before handing the plan to `RemediationManager` for execution.

#### Scenario: Plan from a degraded state
- **WHEN** the operator plans from `STRESSED`, `DEGRADED`, `CRITICAL`, or `FAILED`
- **THEN** it selects at least one intervention (not "none" only) and the expected `P(FAILED)` at the end of the horizon is ≤ the do-nothing baseline

#### Scenario: Action conditioning
- **WHEN** an action (e.g. `pod_restart`) is modelled from `CRITICAL`/`FAILED`
- **THEN** the predicted transition increases the probability of `RECOVERED` relative to the action-agnostic base chain

#### Scenario: Cooldown respect
- **WHEN** an action is inside its cooldown window
- **THEN** the planner does not select it for any step in the horizon

#### Scenario: Execution through safety policy
- **WHEN** `RemediationManager.plan_and_execute` runs a plan
- **THEN** each non-`none` step is executed through the registered action and `SafetyPolicy` (honouring `dry_run`), and the audit trail records every attempted step

#### Scenario: Action eligibility gate
- **WHEN** the planner selects an action whose `should_execute` returns False for the pod
- **THEN** that step is recorded as skipped (ineligible) and the action is not executed

#### Scenario: Default decision path in reconcile
- **WHEN** `PLANNING_ENABLED` is true and a pod's risk exceeds the threshold
- **THEN** the reconcile loop calls `plan_and_execute` (POMDP planner) as the default remediation decision path, falling back to `evaluate` when planning is disabled

#### Scenario: Calibrated uncertainty
- **WHEN** the planner emits an expected `P(FAILED)` for a planned step
- **THEN** it also reports a 95% credible interval derived from the learned transition counts, which narrows as more outcomes are observed

#### Scenario: Plan surfaced to operators
- **WHEN** a pod is at risk during reconcile
- **THEN** the planned intervention sequence is attached to its prediction in `/predictions` and a throttled `plan` notification is sent summarising the intended actions

#### Scenario: Decision-theoretic cost model
- **WHEN** the planner selects an action at each horizon step
- **THEN** it minimises expected cost `P(FAILED) * failure_cost + action_cost`, intervening only when the expected failure reduction outweighs the cost of acting

#### Scenario: World-model what-if query
- **WHEN** an operator or LLM issues `GET /world-model?state=&action=&horizon=`
- **THEN** the agent returns a read-only trajectory (per-step expected state distribution, `P(FAILED)` and its calibrated interval) for applying `action`, without executing anything

#### Scenario: Value-iteration stationary policy
- **WHEN** the agent computes a value-iteration (RL/DP) policy from the action-MDP
- **THEN** it yields a stationary optimal action per state minimising long-run expected cost, available as a benchmarked 4th strategy and via `GET /plan?strategy=policy`

#### Scenario: Sampled RL (Q-learning) strategy
- **WHEN** the agent computes a model-free Q-learning policy from real `(state, action, next_state)` outcomes
- **THEN** it yields a greedy policy available as a 5th benchmarked strategy and via `GET /plan?strategy=q`, persisted across reconciles

#### Scenario: Strategy ladder
- **WHEN** an operator requests a plan
- **THEN** the agent supports strategies `none`, `reactive`, `planner` (default), `policy` (value iteration), `q` (Q-learning), and `coordinated` (multi-agent)

### Requirement: Decision plan endpoint
The operator SHALL expose a `/plan` HTTP endpoint that returns the decision layer's execution-ready remediation plan for a given Markov state, and optionally executes it.

#### Scenario: GET /plan preview
- **WHEN** a client requests `GET /plan?state=CRITICAL&horizon=3`
- **THEN** the operator returns a 200 response with an `execution_ready` plan containing one entry per horizon step (action + expected `P(FAILED)` before/after)

#### Scenario: POST /plan with execute
- **WHEN** a client posts `{"pod_name": "ns/pod", "risk_score": 0.9, "execute": true}`
- **THEN** the operator plans from the pod's current state and runs the plan through `RemediationManager.plan_and_execute`, returning the plan and the execution results (each gated by `SafetyPolicy`)

#### Scenario: Invalid state
- **WHEN** a client supplies a state that is not one of the six Markov states
- **THEN** the endpoint returns a 400 error listing the valid states

#### Scenario: GET /plan?strategy=q
- **WHEN** a client requests `GET /plan?state=CRITICAL&strategy=q`
- **THEN** the operator returns the Q-learning rollout plan

#### Scenario: POST /coordinate multi-agent plan
- **WHEN** an operator posts a pod set to `POST /coordinate`
- **THEN** the operator returns a coordinated joint plan (deployment-action dedup, node-cordon conflict resolution, per-node sequencing waves)

### Requirement: Learning from observations
The operator SHALL refine the decision model from real outcomes so action-conditioned transitions improve over time rather than relying solely on the initial priors.

#### Scenario: Feed observed transition
- **WHEN** `RemediationManager.feed_transition(from, action, to)` is called after a post-remediation state is observed
- **THEN** the persistent `decision_model` learns that `(from, action)` leads to `to`

#### Scenario: Backtester refinement
- **WHEN** `RemediationManager.refine_from_backtester(backtester)` is called
- **THEN** the base transition dynamics are updated from observed prediction/outcome pairs (idempotent — applied once per model instance)

#### Scenario: Model-free policy learning
- **WHEN** `record_observations` pairs a pending `(state, action)` with an observed state
- **THEN** both the transition model and the Q-learning policy are updated from the same outcome

### Requirement: Shadow-mode validation
The operator SHALL support a shadow-mode harness that runs the planner observe-only next to live remediation, recording planner-vs-executed disagreements and predicted-vs-actual calibration without executing.

#### Scenario: Shadow observation
- **WHEN** shadow mode is enabled
- **THEN** the planner observes each cycle but executes nothing, and `GET /shadow` reports disagreements and calibration

#### Scenario: Disabled by default
- **WHEN** shadow mode is not configured
- **THEN** production behaviour is unchanged

### Requirement: Calibration & safety guardrails
The operator SHALL gate planner actions by predictor confidence and model support, cap actions per reconcile cycle, and monitor calibration drift, falling back to reactive remediation on detected drift.

#### Scenario: Confidence gate
- **WHEN** predictor confidence is below `MIN_ACTION_CONFIDENCE`
- **THEN** no planner action is taken

#### Scenario: Model-support gate
- **WHEN** a state has fewer than `MIN_MODEL_SUPPORT` observed transitions
- **THEN** the planner falls back to reactive remediation

#### Scenario: Circuit breaker
- **WHEN** actions executed in a cycle reach `MAX_ACTIONS_PER_RECONCILE`
- **THEN** no further actions execute this cycle

#### Scenario: Calibration drift
- **WHEN** predicted `P(FAILED)` diverges from realised outcomes beyond thresholds
- **THEN** degraded fallback engages (and optionally `AUTO_ROLLBACK_ON_DRIFT`), and stale transitions are decayed

### Requirement: Decision benchmarking
The operator SHALL expose `GET /benchmark` comparing all strategies (none, reactive, planner, policy, q) over real backtester predictions, with a synthetic fallback.

#### Scenario: Benchmark on real data
- **WHEN** `GET /benchmark` is called
- **THEN** it returns per-strategy expected `P(FAILED)` and agreement using scenarios derived from the backtester

### Requirement: Cost-model calibration
The operator SHALL recalibrate decision costs from incident data via `GET`/`POST /cost-model`.

#### Scenario: Calibrate from incidents
- **WHEN** incidents with downtime are posted
- **THEN** `failure_cost` and `action_costs` are recomputed from SLO economics and applied to the planner/policy

### Requirement: Decision-layer metrics
The operator SHALL expose decision-layer metrics for Prometheus (`GET /metrics`) and JSON (`GET /decision-metrics`): intervention rate, mean `P(FAILED)` reduction vs do-nothing, learning volume, calibration (Brier/over-rate/drift/samples), degraded flag, and mean predictor confidence.

#### Scenario: Scrape metrics
- **WHEN** Prometheus scrapes `GET /metrics`
- **THEN** decision-layer gauges/counters are present

#### Scenario: JSON metrics
- **WHEN** `GET /decision-metrics` is called
- **THEN** the same metrics are returned as JSON

### Requirement: Multi-agent coordination (Dec-POMDP)
The operator SHALL coordinate independent per-pod plans into one safe joint plan that deduplicates deployment-level actions, defers pod-level mutations on cordoned nodes, and sequences actions per node with risk priority (`POST /coordinate`).

#### Scenario: Deployment dedup
- **WHEN** two pods of one deployment both plan `scale`/`rollout_restart`
- **THEN** only one is kept

#### Scenario: Node-cordon conflict
- **WHEN** a node is cordoned
- **THEN** pod-level mutations on that node are deferred in that wave

#### Scenario: Per-node sequencing
- **WHEN** many pods share a node
- **THEN** actions are packed into waves respecting `MAX_ACTIONS_PER_NODE` with `CRITICAL`/`FAILED` first

### Requirement: Continual learning
The operator SHALL continually refine its decision model via forgetting of stale transitions (periodic decay), drift-aware reweighting (steeper decay on calibration drift), and confidence-gated acting.

#### Scenario: Periodic forgetting
- **WHEN** `MODEL_DECAY_INTERVAL` reconciles elapse
- **THEN** learned counts are decayed by `MODEL_DECAY_FACTOR`

#### Scenario: Drift reweighting
- **WHEN** calibration drift is detected
- **THEN** the model is decayed by `DRIFT_DECAY_FACTOR`

#### Scenario: Confidence-gated acting
- **WHEN** the model lacks support for a state
- **THEN** it defers to reactive remediation

### Requirement: Reconcile pipeline facade stability
The main module SHALL preserve its public facade while its internal
structure changes: the `reconcile()` entry point, the module-level
component and statistics attributes (`_state_model`, `_predictor`,
`_state_store`, `_remediation_manager`, `_notifier`, `_backtester`,
`_policy_engine`, `_llm_analyzer`, `_kg_client`, `_reconcile_count`,
`_last_reconcile_time`, `_last_reconcile_duration`), and the collection
hooks (`_get_pods_json`, `_collect_docker_container_metrics`) SHALL
remain importable and patchable at the module path
`predictive_agent.main`. The HTTP server and the test suite rely on
resolving these names on the main module at call time, not on copies.

#### Scenario: Server keeps reading live state through the facade
- **WHEN** the HTTP server reads component references or reconcile
  statistics from the main module
- **THEN** it observes the values updated by the most recent reconcile
  cycle, without importing any new internal module

#### Scenario: Collection hooks stay overridable
- **WHEN** a test patches a collection hook on the main module
- **THEN** the next reconcile cycle resolves and uses the patched hook,
  and no unpatched duplicate implementation is invoked instead

#### Scenario: Reconcile summary contract unchanged
- **WHEN** one reconcile cycle completes
- **THEN** it returns the same summary keys as before the
  decomposition (predictions, state, timestamp, pods_tracked,
  at_risk_count, reconcile_count, and the remediation/backtest
  statistics blocks)

### Requirement: Tier-2 static analysis gate

The CI pipeline SHALL run a static analysis tool over the `predictive_agent`
package that detects common security issues (hardcoded secrets, shell
injection via `run_cmd`, use of `eval`/`exec`, weak crypto, weak TLS, and
overly-broad `except` blocks that swallow errors silently).

#### Scenario: bandit finds no high/critical issues on main
- **WHEN** CI runs `bandit -r predictive_agent -ll -iii -ll`
- **THEN** the job reports zero findings at severity `HIGH` (B3) or `CRITICAL` (B4)
  and the reconciler's deliberately-broad `except Exception: # noqa: BLE001`
  fail-open blocks are not flagged at the configured severity/confidence level
- **AND** the job exits 0

#### Scenario: a new high-severity bandit finding blocks merge
- **WHEN** a PR introduces a `HIGH` or `CRITICAL` bandit finding
- **THEN** the bandit step fails and the PR is blocked from merging
  until the finding is resolved or explicitly justified with a
  `# nosec` annotation reviewed by a maintainer

### Requirement: Dependency vulnerability scanning gate

The CI pipeline SHALL scan project dependencies (runtime transitive deps
declared in `requirements.txt` / `pyproject.toml` build) for known
vulnerabilities and report versions with no fixed upstream patch.

#### Scenario: pip-audit clean on main
- **WHEN** CI runs `pip-audit` against the resolved dependency set
- **THEN** the job reports zero known vulnerabilities (or only
  `continue-on-error`-flagged advisory items with no upstream fix)
- **AND** the job exits 0

### Requirement: Coverage ratchet (non-blunt)

The coverage gate SHALL enforce `fail_under = 90` for the project while
explicitly excluding known-low-coverage entrypoint/demo files so the
ratchet cannot be defeated by padding, nor trip on intentionally-skipped
code.

#### Scenario: coverage floor holds after a feature PR
- **WHEN** CI computes aggregate coverage with
  `pytest --cov=predictive_agent --cov-report=term-missing`
- **THEN** the job fails if coverage drops below 90%
- **AND** `demo_optimizations.py` (demo script), `__main__.py` (3-line
  entrypoint), and `tests/` are excluded from the gate denominator

### Requirement: CLI entrypoint test coverage

The `predictive_agent/__main__.py` entrypoint SHALL have a regression
test so its 0% coverage is removed and the `python -m predictive_agent`
invocation path is protected against regressions.

#### Scenario: entrypoint imports and reports usage
- **WHEN** a test runs `python -m predictive_agent --help` (via subprocess
  or `runpy.run_module`)
- **THEN** it exits 0 (or exits cleanly with a usage message) and the test
  asserts the module is importable without error

### Requirement: Bounded-metric property tests for the risk model
A Hypothesis strategy over physically-valid pod metrics SHALL be used to
verify invariants of the prediction + state-classification pipeline.

#### Scenario: risk_score is always normalized to [0, 1]
- **WHEN** `Predictor.predict` is invoked with any combination of
  memory_pct ∈ [0, 100], cpu_pct ∈ [0, 100], restart_rate_per_hr ∈ [0, ∞),
  and log_error_rate_per_min ∈ [0, ∞)
- **THEN** `result.risk_score` is a float in the closed interval [0, 1]
- **AND** `predict` does not raise for any such input

#### Scenario: state classifier always yields a canonical Markov state
- **WHEN** `classify_state` is invoked with any combination of the metric
  arguments within their valid domains
- **THEN** the returned state is one of `HEALTHY`, `DEGRADED`, `STRESSED`,
  `CRITICAL` (never `None`, never a novel string)

#### Scenario: StateModel.update_pod keeps state within the Markov set
- **WHEN** a pod tracker is updated via `StateModel.update_pod` with
  random-but-valid metrics (memory ≥ 0, cpu ≥ 0, restart_count ≥ 0)
- **THEN** `tracker.state` and `tracker.prev_state` remain in the canonical
Markov set after the update

### Requirement: Hypothesis determinism in CI
The CI property-test job SHALL pin the Hypothesis random seed for
reproducible shrinkage.

#### Scenario: CI uses a pinned seed for reproducibility
- **WHEN** the Hypothesis job runs in CI
- **THEN** it passes `--hypothesis-seed=0` so a failing example shrinks
  to the same minimal input on every run, and `.hypothesis/` failures
  are committed for replay

### Requirement: Mutation testing on extracted phase modules
The project SHALL run a mutation-testing campaign (`mutmut`) over the
extracted phase modules (`reconcile_phases.py`, `pod_collection.py`,
`bootstrap.py`) on demand (NOT per-push: the self-hosted runner is a
shared host), and SHALL publish the killed/survived counts per module.
Survivors SHALL be triaged in family batches; safety-critical mutation
families (external-command argv/timeout contracts, threshold polarity,
boolean gate polarity, persisted-payload key contracts) SHALL be fully
killed by contract-pinning tests.

#### Scenario: campaign runs and reports per-module results
- **WHEN** `python -m mutmut run` completes for the three scoped modules
- **THEN** `python -m mutmut results` reports killed/survived counts and
  the overall kill rate is recorded in the change tasks (baseline: 48.0%)
- **AND** the campaign is run ad hoc (manual/on-demand), never scheduled
  per-push, to protect the shared self-hosted runner

#### Scenario: safety-critical mutant families are killed
- **WHEN** a survivor mutates an external-command argv/timeout contract,
  a state-classification threshold comparison, a log-collection gate
  polarity, or a persisted-state payload key
- **THEN** a contract-pinning regression test is added that kills the
  entire family (e.g. exact argv assertion kills all string-casing and
  timeout±1 mutants at that call site)

#### Scenario: equivalent-mutant classes are documented, not chased
- **WHEN** survivors are dominated by log-format-string mutations,
  `getattr` default fallbacks on loosely-mocked branches, or numeric
  constants in rarely-asserted diagnostic paths
- **THEN** they are documented as accepted classes in the change tasks
  rather than pinned with brittle per-log-line assertions
