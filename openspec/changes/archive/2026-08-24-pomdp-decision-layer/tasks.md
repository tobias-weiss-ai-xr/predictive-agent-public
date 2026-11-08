## 1. Action-conditioned transition model

- [x] 1.1 Add `ActionAwareTransitionModel` in `predictive_agent/pomdp.py` reusing
  `MarkovChain.STATES`/`STATE_IDX`; `transition_matrix(action)` returns the base
  chain matrix plus conservative action priors, and prefers learned
  `record_transition` counts when present.
- [x] 1.2 Add `step_distribution(dist, action)` and `failed_probability(dist)`
  helpers operating on state distributions.

## 2. Short-horizon planner

- [x] 2.1 Add `Planner` with `plan(current_state, horizon)` performing greedy
  Scenario-MPC lookahead minimising expected `P(FAILED)` plus a small action-cost
  penalty, respecting per-action cooldowns.
- [x] 2.2 Add `Plan` / `PlanStep` dataclasses and `to_dict()` (execution_ready).

## 3. Remediator integration

- [x] 3.1 Add `RemediationManager.plan_and_execute(...)` that plans and executes
  each non-`none` step through `_apply_named_action` (safety check → execute →
  audit trail), honouring `dry_run`.
- [x] 3.2 Keep the existing reactive `evaluate()` path untouched.

## 4. Tests & verification

- [x] 4.1 Add `tests/test_pomdp.py` covering: base-chain fallback, action-prior
  effect, learning from observed transitions, horizon risk reduction vs
  do-nothing baseline, cooldown respect, and plan-dict contract.
- [x] 4.2 Run `pytest` on Python ≥3.11 (CI) and confirm all new + existing tests pass.
- [x] 4.3 Add a `GET /plan?state=&horizon=` endpoint returning the execution-ready
  plan (preview) and a `POST /plan` endpoint that plans and optionally executes
  (`execute=true`) through `RemediationManager.plan_and_execute`. Add
  `tests/test_endpoints.py::test_plan_endpoint`.
- [x] 4.4 Refine action priors from observations: `RemediationManager.feed_transition`
  teaches the persistent `decision_model` observed (state, action) -> next-state;
  `refine_from_backtester` seeds the base dynamics from backtester prediction/
  outcome pairs (idempotent). Wire `/plan` to refine once when a backtester is present.
- [x] 4.5 Add unit tests for `plan_remediation` preview, `feed_transition` learning,
  and idempotent `refine_from_backtester`.

## 6. Close the learning loop (reconcile integration)

- [x] 6.1 Add `to_action_dict` / `load_action_counts` to `ActionAwareTransitionModel`
  and `save_decision_model` / `load_decision_model` to `StateStore`.
- [x] 6.2 Record each executed remediation (`state`, `action`) as pending feedback
  in both `evaluate` and `plan_and_execute`; pair it with the next observed pod
  state via `record_observations` in the reconcile loop and teach `feed_transition`.
- [x] 6.3 Persist the learned decision model every reconcile and reload on startup,
  sharing the live `MarkovChain` as the base dynamics (so the planner uses the
  operator's real learned transitions).
- [x] 6.4 Add `tests/test_pomdp.py` coverage for the learning loop + persistence.

## 7. Default decision path (planning in reconcile)

- [x] 7.1 Add `PLANNING_ENABLED` / `PLANNING_HORIZON` config; the reconcile loop
  uses `plan_and_execute` as the default remediation path when enabled, with
  `evaluate()` as the fallback when disabled.
- [x] 7.2 Add a `should_execute` safety gate inside `_apply_named_action` so the
  planner never executes an action the action itself deems ineligible.
- [x] 7.3 Add `tests/test_pomdp.py::test_plan_and_execute_respects_should_execute`.

## 8. Benchmark planner vs reactive baseline

- [x] 8.1 Add `predictive_agent/benchmark.py`: compares expected `P(FAILED)` under
  none / reactive (`evaluate()` would-fire set, dry-run) / planner per scenario,
  with `scenarios_from_backtester` deriving scenarios from real prediction data.
- [x] 8.2 Add `tests/test_benchmark.py` covering reactive gating, scenario structure,
  `run_benchmark` summary, and backtester scenario extraction.
- [x] 8.3 Wire a `python -m predictive_agent.benchmark` CLI that loads the persisted
  decision model + backtester and prints a text report.

## 9. Calibrated uncertainty on P(FAILED)

- [x] 9.1 Add `ActionAwareTransitionModel.predictive_interval(dist, action)` — an
  analytic Beta-posterior 95% credible interval for expected `P(FAILED)`, derived
  from the (base + prior, or learned) transition counts; narrows as data grows.
- [x] 9.2 Surface the interval in `PlanStep` / `Plan.to_dict` so `/plan` and the
  benchmark report communicate calibrated risk, not just a point estimate.
- [x] 9.3 Add `tests/test_pomdp.py::test_predictive_interval_shape_and_narrows_with_data`.

## 10. Surface plan to operators

- [x] 10.1 Add `remediation_plans` to the server `_ServerState`; the reconcile
  loop computes `plan_remediation(state, horizon)` for at-risk pods and pushes it
  to the server context (rebuilt each cycle, so only current at-risk pods show).
- [x] 10.2 Include `planned_actions` in `_serialize_prediction` so `/predictions`
  shows the intended intervention sequence, not just the risk score.
- [x] 10.3 Send a throttled `plan` notification (new `alert_type`) summarizing the
  planned actions for high-risk pods.
- [x] 10.4 Add `tests/test_endpoints.py::test_serialize_prediction_includes_planned_actions`.

## 11. Decision-theoretic cost model

- [x] 11.1 Add `DEFAULT_ACTION_COSTS` + `FAILURE_COST` and replace the flat
  `cost_weight` penalty with an expected-cost objective in `Planner`
  (`score = P(FAILED) * failure_cost + action_cost`), so the planner intervenes
  only when the expected failure reduction outweighs the intervention cost.
- [x] 11.2 Propagate `action_costs` through `RemediationManager`
  (`__init__` + `plan_and_execute` + `plan_remediation`) so production uses it.
- [x] 11.3 Add `tests/test_pomdp.py::test_decision_theoretic_cost_tradeoff` and
  confirm the benchmark still holds (planner <= reactive/none).

## 12. World-model what-if query

- [x] 12.1 Add `ActionAwareTransitionModel.what_if(start_state, action, horizon)`
  returning the per-step expected state distribution, `P(FAILED)` and its
  calibrated credible interval (reuses #4's `predictive_interval`, centred on the
  MLE so it matches the displayed point estimate).
- [x] 12.2 Add `GET /world-model?state=&action=&horizon=` (read-only; never
  executes) using the live decision model; invalid state/action -> 400.
- [x] 12.3 Add `tests/test_endpoints.py::test_world_model_endpoint`.

## 13. Value-iteration (RL) stationary policy

- [x] 13.1 Add `ValueIterationPolicy(model, ...)` to `pomdp.py`: exact value
  iteration over the action-MDP with reward `-failure_cost*P(FAILED)-action_cost`,
  producing a stationary optimal policy; `plan()` rolls it out for a horizon.
- [x] 13.2 Add the policy as a 4th strategy in `benchmark.py`
  (`policy_p_failed`/`policy_actions`) and surface it in the summary/text report.
- [x] 13.3 Expose `GET /plan?strategy=policy` so operators/LLM can compare the
  DP policy against the myopic planner.
- [x] 13.4 Add `tests/test_pomdp.py::test_value_iteration_policy_reduces_failure`
  and `tests/test_endpoints.py::test_plan_endpoint_policy_strategy`.

## 14. Testing & local spin-up

- [x] 14.1 Fix the local Python 3.9 import blocker in `optimize.py`
  (`Type[collector_module]` -> `types.ModuleType`) so the package and the full
  `pytest` suite run locally, not just on CI (>=3.11).
- [x] 14.2 Add `tests/test_decision_layer_integration.py`: spins up the real
  HTTP server + a learned `RemediationManager` (no-op fake actions, no cluster)
  and verifies the end-to-end loop — endpoints, closed learning loop, model
  persistence, and the four-strategy benchmark.
- [x] 14.3 Add `scripts/local_demo.py` so the agent can be "spun up" locally
  with one command (`python scripts/local_demo.py`) and demonstrates all eight
  decision-layer capabilities without a Kubernetes cluster.
- [x] 14.4 Fix three tests to match the cost-aware behaviour: relax
  `test_planner_reduces_failure_probability` (planner need not act when acting
  is not worth the cost) and correct two `/plan` + `/world-model` 400 assertions
  (urllib raises `HTTPError` on 4xx).

## 15. Reconcile-level end-to-end test (ranked option #2)

- [x] 15.1 Add `tests/test_reconcile_integration.py` that drives the *real*
  `predictive_agent.main.reconcile()` with a patched `run_cmd` returning
  synthetic Kubernetes output (no cluster), so it exercises the production
  wiring that unit tests bypass: `record_observations` -> `_record_pending` ->
  next cycle `record_observations` (observed state) -> `feed_transition`.
- [x] 15.2 Make the test run two reconcile cycles (critical pod, then recovered
  pod) and assert the closed learning loop actually closes through reconcile:
  the planner executes an action, records pending feedback, and the next cycle's
  observation teaches the decision model a new `CRITICAL -> observed`
  transition. Also assert the plan is surfaced in the server context.

## 16. Shadow-mode validation harness (ranked option #1)

- [x] 16.1 Add `predictive_agent/shadow.py::ShadowModeHarness` that runs the
  planner *observe-only* next to the live remediation path: it logs planner-vs-
  executed disagreements and predicted `P(FAILED)` vs actual-outcome calibration
  without executing anything or mutating the decision model.
- [x] 16.2 Wire the harness into `main.reconcile` through an optional
  `_shadow_harness` global (disabled by default) so enabling it is purely
  additive — no production behaviour change when unset.
- [x] 16.3 Expose a read-only `GET /shadow` endpoint returning the harness
  summary (enabled flag, agreement/disagreement counts, calibration
  over/under counts) so operators can watch the planner validate itself in
  production.
- [x] 16.4 Add `tests/test_shadow_mode.py` covering disagreement detection,
  agreement logging, no-execution guarantee, calibration over/under counting,
  and the `/shadow` endpoint.

## 17. Calibration monitoring + safety guardrails (ranked option #3)

- [x] 17.1 Add `predictive_agent/calibration.py::CalibrationMonitor`: rolling
  window of predicted `P(FAILED)` vs realised outcome; computes Brier score,
  over/under-prediction rates, and `drift_detected()` (Brier / over-rate above
  configurable thresholds, gated by `CALIBRATION_MIN_SAMPLES`).
- [x] 17.2 Feed the monitor from the production learning loop: `record_observations`
  pairs each pending `(state, action, predicted_p_failed)` with the observed
  state and calls `calibration_monitor.record(predicted_p_failed, actual_failed)`.
  `predicted_p_failed` is captured at execution time (the live risk score).
- [x] 17.3 Safety guardrails in `plan_and_execute`: confidence gate
  (`config.MIN_ACTION_CONFIDENCE` — refuse to act when the predictor is unsure),
  circuit-breaker `max_actions` (blast-radius cap per call), and degraded
  fallback to the conservative reactive `evaluate()` path.
- [x] 17.4 Wire the guardrails into `main.reconcile`: per-cycle action budget
  (`config.MAX_ACTIONS_PER_RECONCILE`), attach a `CalibrationMonitor` to the
  manager on startup, and on drift set `rem.degraded` + raise a
  `calibration_drift` alert (auto-rollback gated by `config.AUTO_ROLLBACK_ON_DRIFT`).
- [x] 17.5 Expose a read-only `GET /calibration` endpoint (monitor summary +
  `degraded` flag).
- [x] 17.6 Add `tests/test_calibration_guardrails.py` covering drift detection,
  the learning-loop feeding calibration, confidence gate, circuit breaker,
  degraded fallback, and the `/calibration` endpoint.

## 18. Benchmark on real backtester data (ranked option #4)

- [x] 18.1 Add a `GET /benchmark` endpoint that runs the 4-strategy decision
  benchmark (`none` / reactive / planner / value-iteration policy) over
  scenarios derived from real backtester predictions via
  `scenarios_from_backtester`, falling back to synthetic scenarios when none
  are recorded. Returns `source`, per-scenario results, and the summary.
- [x] 18.2 Add `tests/test_benchmark_backtester.py` covering scenario derivation
  from recorded predictions, the benchmark over real scenarios, and the
  `/benchmark` endpoint (backtester vs synthetic source).

## 19. Calibrate cost model from incident data (ranked option #5)

- [x] 19.1 Add `predictive_agent/cost_model.py::calibrate_costs`: derives
  `action_costs` (mean downtime x SLO $/min x disruption fraction) and
  `failure_cost` (SLO $/min x typical outage minutes) from incident history.
- [x] 19.2 `RemediationManager` gains a configurable `failure_cost` (default
  `FAILURE_COST`) threaded into `Planner`/`ValueIterationPolicy`, plus
  `calibrate_from_incidents(...)` that recalibrates the live cost model.
- [x] 19.3 Config knobs `SLO_COST_PER_MINUTE`, `INCIDENT_TYPICAL_OUTAGE_MIN`,
  `ACTION_DISRUPTION_FRACTION` drive the calibration defaults.
- [x] 19.4 `GET /cost-model` (current costs + source) and `POST /cost-model`
  (recalibrate from posted incident data) endpoints; the `/plan?strategy=policy`
  endpoint now uses the manager's calibrated costs.
- [x] 19.5 Add `tests/test_cost_model.py` covering calibration math, manager
  wiring, planner response to calibrated costs, and the `/cost-model` endpoint.

## 20. Operator-facing natural-language rationale (ranked option #6)

- [x] 20.1 `predictive_agent.pomdp`: `PlanStep.rationale` carries a
  human-readable explanation per step (action, before/after P(FAILED), cost),
  generated by `_step_rationale`; included in `Plan.to_dict()`.
- [x] 20.2 `main.reconcile`: the throttled `plan` operator notification now
  appends the per-step rationale (not just the action list) so operators see
  *why* each intervention is planned.
- [x] 20.3 Add `tests/test_rationale.py` covering rationale on plan steps,
  `to_dict`, the `/plan` endpoint, and the reconcile `plan` notification.

## 21. Decision-layer Prometheus metrics (ranked option #7)

- [x] 21.1 `predictive_agent/metrics.py`: `collect_decision_metrics` aggregates
  operator-facing metrics from the live `RemediationManager` (and optional
  `Predictor` / calibration monitor) — intervention count, mean P(FAILED)
  reduction vs the do-nothing baseline, learning volume, calibration Brier /
  drift / samples, degraded flag, mean predictor confidence. `render_prometheus`
  emits the text format.
- [x] 21.2 `RemediationManager` accumulates `interventions_executed` and
  `cumulative_p_failed_reduction` (expected P(FAILED) of the executed plan vs
  the natural-decay baseline) during `plan_and_execute`.
- [x] 21.3 Server: decision-layer metrics appended to `GET /metrics` (scraped
  by Prometheus) and surfaced as JSON via a new `GET /decision-metrics` endpoint.
- [x] 21.4 Add `tests/test_decision_metrics.py` covering the counters, the
  collection helper, Prometheus rendering, and both endpoints.

## 22. Sampled RL: model-free Q-learning from outcomes (ranked option #8)

- [x] 22.1 `predictive_agent/rl.py`: `QLearningPolicy` — tabular Q-learning over
  the 6-state action-MDP. Learns `Q(state, action)` from real `(s, a, s')`
  outcomes (same decision-theoretic reward as the DP policy). Pessimistic prior
  for untried remediation actions (conservative on a live cluster). `plan()`
  rolls out the greedy policy using a supplied transition model; `to_dict` /
  `load` for persistence.
- [x] 22.2 Reuse the production learning loop: `RemediationManager` gains a
  `rl_policy` slot; `record_observations` feeds the same outcomes to it as to
  the transition model.
- [x] 22.3 Benchmark: add Q-learning as a 5th strategy (`q_actions`,
  `q_p_failed`, `q_intervenes`, `mean_q_p_failed`) for direct comparison.
- [x] 22.4 Server: `GET /plan?strategy=q` returns the Q-learning rollout; the Q
  policy is created on agent startup, loaded from / persisted to the state store
  (`StateStore.save_q_table` / `load_q_table`) so it accumulates across
  reconciles.
- [x] 22.5 Add `tests/test_rl.py` covering learning-to-act, plan reduction,
  learning-loop wiring, the benchmark strategy, the endpoint, and persistence.

## 23. Dec-POMDP / multi-agent coordination (ranked option #9)

- [x] 23.1 `predictive_agent/coordinator.py`: `MultiAgentCoordinator` joins
  independent per-pod plans into one safe joint plan. Deduplicates
  deployment-level actions (`scale` / `rollout_restart`) to one per
  (deployment, action); defers pod-level mutations on a node being cordoned
  (records a `node_cordon_conflict`); sequences actions into waves respecting a
  per-node concurrency cap (`MAX_ACTIONS_PER_NODE`) with CRITICAL/FAILED-first
  priority.
- [x] 23.2 `config.MAX_ACTIONS_PER_NODE` knob (default 2).
- [x] 23.3 Server: `POST /coordinate` builds the joint plan from a pod set
  (each pod carries `state` plus optional `node` / `namespace` / `deployment`).
- [x] 23.4 Add `tests/test_coordinator.py` covering deployment dedup (real
  manager), node-cordon conflict deferral, priority + per-node cap, and the
  endpoint.

## 24. Continual-learning hardening (ranked option #10)

- [x] 24.1 `predictive_agent/pomdp.py`: `ActionAwareTransitionModel` gains
  `decay(alpha)` (exponential forgetting of stale transitions), `total_support`,
  `state_support`, and `is_confident(state, min_support)`.
- [x] 24.2 Config knobs `MIN_MODEL_SUPPORT`, `MODEL_DECAY_INTERVAL`,
  `MODEL_DECAY_FACTOR`, `DRIFT_DECAY_FACTOR`.
- [x] 24.3 Confidence-gated acting: `plan_and_execute` falls back to reactive
  remediation from states the model has barely observed (`MIN_MODEL_SUPPORT`).
- [x] 24.4 Drift-aware reweighting + periodic forgetting in `main.reconcile`:
  on detected calibration drift the decision model is decayed by
  `DRIFT_DECAY_FACTOR`; every `MODEL_DECAY_INTERVAL` cycles it is decayed by
  `MODEL_DECAY_FACTOR`.
- [x] 24.5 Add `tests/test_continual_learning.py` covering decay, support /
  confidence, the planning confidence gate, and drift-triggered decay.

## 25. ADR + operator runbook (ranked option #11)

- [x] 25.1 `docs/adr/0001-pomdp-decision-layer.md`: architecture decision record
  for the POMDP decision layer (state/action/objective, strategy ladder,
  safety & trust, observability, consequences).
- [x] 25.2 `docs/OPERATOR_RUNBOOK.md`: how to run locally, endpoint reference,
  configuration knobs, how to read `/decision-metrics`, common operational
  scenarios, and the safety posture.

## 26. CI / test hygiene (ranked option #12)

- [x] 26.1 Docker / container-restart socket tests skip gracefully when the
  AF_UNIX path exceeds the platform limit (macOS) instead of erroring.
- [x] 26.2 The LLM background-thread regression test is made deterministic by
  lowering the analysis threshold under test rather than depending on the
  exact risk model crossing 0.7.
- [x] 26.3 Tight performance micro-benchmarks use env-overridable ceilings so a
  loaded/CI runner does not red-CI on transient slowness while still catching
  gross regressions.
- [x] 26.4 Add `.github/workflows/ci.yml` running `pytest` on Python 3.11/3.12
  so the new integration tests run in CI; docker/LLM tests skip when unavailable.

