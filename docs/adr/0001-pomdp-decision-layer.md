# ADR 0001: POMDP Decision Layer for Predictive Agent

## Status
Accepted — implemented in the `2026-08-24-pomdp-decision-layer` change.

## Context
The `predictive-agent` already predicts pod failure risk (Kalman + Markov +
Bayesian risk) and remediates reactively. Operators wanted it to *plan* — choose
a remediation sequence that minimises expected failure risk over a horizon,
intervene only when worthwhile, and keep learning from the outcomes of its own
actions. The corpus-derived decision-making literature (POMDPs, MDP value
iteration, model-free RL, Dec-POMDP coordination) informed the design.

## Decision
Model the agent as a **POMDP over a 6-state Markov chain** (HEALTHY, DEGRADED,
STRESSED, CRITICAL, FAILED, RECOVERED):

- **State** — the pod's Markov state (belief derived from predictors).
- **Actions** — `none`, `pod_restart`, `rollout_restart`, `node_cordon`,
  `scale`, `right_size`, `tune_resources`.
- **Transition model** — `ActionAwareTransitionModel`: action-conditioned 6×6
  counts, seeded from conservative priors and refined from real backtester /
  reconcile outcomes. Falls back to the base chain when no action data exists.
- **Objective** — decision-theoretic, not a flat penalty:
  `score(a) = P(FAILED|a) * failure_cost + action_cost`. The planner
  intervenes only when the expected risk reduction exceeds the cost of acting.

### Strategy ladder (all exposed via `GET /plan?strategy=...`)
1. `none` — do-nothing baseline (never learns from "none" outcomes).
2. `reactive` — existing threshold/cooldown remediation (baseline).
3. `planner` — greedy Scenario-MPC over the learned model (default production
   path).
4. `policy` — exact value iteration over the action-MDP (`ValueIterationPolicy`).
5. `q` — model-free Q-learning from real outcomes (`QLearningPolicy`), persisted
   across reconciles.
6. `coordinated` (multi-pod, `POST /coordinate`) — Dec-POMDP: joins per-pod
   plans, deduplicates deployment actions, defers node-cordon conflicts, and
   sequences actions per-node with CRITICAL/FAILED-first priority.

### Safety & trust
- **Confidence gate** (`MIN_ACTION_CONFIDENCE`): never act when the predictor is
  unsure.
- **Model-support gate** (`MIN_MODEL_SUPPORT`, #10): fall back to reactive
  remediation from states the model has barely observed.
- **Circuit breaker** (`MAX_ACTIONS_PER_RECONCILE`): caps blast radius per cycle.
- **Calibration monitor** + drift detection (#3): tracks predicted P(FAILED) vs
  realised; on drift, flips the planner into reactive fallback and (optionally)
  rolls back. Decay reweights stale transitions.
- **Continual learning** (#10): exponential forgetting (`MODEL_DECAY_INTERVAL`),
  drift-aware reweighting (`DRIFT_DECAY_FACTOR`), and confidence-gated acting.
- **Shadow mode** (#1): the planner runs observe-only next to live remediation
  and logs disagreements + calibration; it never executes.

### Observability
- `GET /metrics` (Prometheus) and `GET /decision-metrics` (JSON) expose
  intervention rate, mean P(FAILED) reduction, learning volume, calibration, and
  confidence (#7).
- `GET /benchmark` compares all strategies on real backtester data (#4).
- `POST /cost-model` recalibrates costs from incident data (#5).
- `GET /shadow` and `GET /calibration` surface the validation/guardrail state.

## Consequences
- The planner output is **execution-ready** and routed through `SafetyPolicy` +
  `should_execute` — it is advisory-to-executing, not advisory-only.
- The closed learning loop runs through production wiring:
  `plan_and_execute → _record_pending → record_observations → feed_transition`
  (and feeds the calibration monitor and Q-learning policy).
- Multiple learning mechanisms (model-based DP, value iteration, model-free Q,
  Dec-POMDP coordination) coexist and are individually disablable, so production
  behaviour is unchanged until explicitly enabled.
