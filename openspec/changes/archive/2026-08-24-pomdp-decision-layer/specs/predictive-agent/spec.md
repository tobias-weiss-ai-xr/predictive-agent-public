## Requirements

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
