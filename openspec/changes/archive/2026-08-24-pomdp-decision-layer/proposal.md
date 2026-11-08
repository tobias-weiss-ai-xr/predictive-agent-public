## Why

The operator already predicts failure (Kalman + 6-state Markov chain + Bayesian
risk) **and** remediates it (`remediator.RemediationManager` + six registered
actions). But the two halves are not yet connected by a *decision policy*: the
Markov chain only forecasts `state → next state`, it does not reason about how
an **action** changes that transition, and remediation is currently triggered
reactively per-action (`should_execute`) rather than planned as a sequence that
minimises expected failure over a horizon.

This gap is exactly what the decision-making research corpus
(`dm-research`, 12,860 papers, built knowledge graph) identifies as the
evolution from *prediction* to *decision-making under uncertainty*:

- POMDP maintenance planning — *"Bridging POMDPs and Bayesian decision making
  for robust maintenance planning"* (2023)
- POMDP + Scenario MPC planning under uncertainty — *"On Integrating POMDP and
  Scenario MPC for Planning under Uncertainty"* (2022)
- Model-based / world-model forecasting for "what-if" intervention analysis —
  *"Model-based Reinforcement Learning: A Survey"* (2023)

A POMDP-style decision layer lets the agent choose *which* intervention (and in
what order) actually reduces the probability of `FAILED`, instead of firing
every action whose threshold trips.

## What Changes

- Add `predictive_agent/pomdp.py` with:
  - `ActionAwareTransitionModel` — action-conditioned 6-state transitions
    `P(next | state, action)`, extending the existing `MarkovChain` states.
    Action priors are principled pseudo-counts (clearly marked) meant to be
    refined from `backtester` data via `record_transition`.
  - `Planner` — short-horizon (Scenario-MPC-style) lookahead that selects the
    action sequence minimising expected `P(FAILED)`, respecting per-action
    cooldowns. Output is an **execution-ready** plan.
- Wire the planner into `RemediationManager.plan_and_execute(...)`, which runs
  each planned step through the existing `SafetyPolicy` + registered actions —
  closing the loop: Markov prediction → POMDP action selection → execution.
- Add `tests/test_pomdp.py` (pytest) covering action-conditioning, learning
  from observed transitions, horizon reduction, cooldown respect, and the
  plan-dict contract.

## Capabilities

### New Capabilities
- `pomdp-decision-layer`: action-conditioned transition model + short-horizon
  remediation planner that selects interventions to minimise expected
  `P(FAILED)` over a horizon, feeding `RemediationManager`.

### Modified Capabilities
- `predictive-agent`: adds a "Decision layer (POMDP planning)" requirement; the
  existing predictor (Kalman/Markov/risk) and `RemediationManager` now connect
  through the planner rather than per-action thresholds alone. Adds a
  "Decision plan endpoint" requirement (`GET`/`POST /plan`) and a "Learning from
  observations" requirement (`feed_transition` / `refine_from_backtester`).

## Impact

- **Code**: `predictive_agent/pomdp.py` (new), `predictive_agent/remediator.py`
  (`plan_and_execute` + `_apply_named_action`), `tests/test_pomdp.py` (new).
- **Config/Deploy**: none — planner is stdlib-only and reuses existing actions.
- **Tests**: `tests/test_pomdp.py` run via `pytest` (project requires Python ≥3.11).
- **Dependencies**: none new — stays stdlib-only (project convention).

## Non-goals

- Not replacing the existing reactive `evaluate()` per-action path — both
  coexist; `plan_and_execute` is the planned/sequential alternative.
- Not auto-tuning the action priors yet beyond backtester/observation refinement
  — the initial priors are conservative and are refined from `backtester` data
  and `feed_transition` calls as real outcomes are observed.
