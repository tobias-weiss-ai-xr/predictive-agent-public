## Context

The reconcile loop drives collection → state model (Kalman + Markov) → Bayesian
risk → `RemediationManager`. The Markov chain (`markov.py`) models `state →
next state` for six states (`HEALTHY, DEGRADED, STRESSED, CRITICAL, FAILED,
RECOVERED`) but is *action-agnostic*: it cannot say how a remediation changes
the transition. Remediation (`remediator.py`) already executes six registered
actions (`actions/`) under a `SafetyPolicy`, but each action self-decides via
`should_execute` thresholds — there is no joint selection of the *sequence* that
best avoids `FAILED`.

The decision-making corpus shows the missing piece is a **decision policy under
uncertainty**: a POMDP / Scenario-MPC planner that, given the predicted state
trajectory, picks interventions minimising expected failure. This change adds
exactly that, kept execution-safe by routing every planned step back through the
existing `SafetyPolicy` and `dry_run` flag.

## Goals / Non-Goals

**Goals:**
- Model `P(next | state, action)` reusing the existing 6-state vocabulary.
- Plan a short horizon of interventions minimising expected `P(FAILED)`.
- Execute the plan through the existing `RemediationManager` + `SafetyPolicy`.
- Stay stdlib-only and pytest-covered.

**Non-Goals:**
- No replacement of the reactive per-action `evaluate()` path.
- No learned action models beyond what `feed_transition` / `refine_from_backtester`
  accrue; the initial priors remain conservative and are refined from real outcomes.

## Decisions

1. **Action-conditioned transitions, not a full POMDP solver.** Model
   `P(next | state, action)` as a 6×6 count matrix per action, seeded from the
   base `MarkovChain` plus conservative priors. A full belief-state POMDP is
   overkill for six discrete states; the 1-step lookahead × horizon is enough
   and stays interpretable. *Alternative*: full POMDP value iteration — rejected
   (more complexity, no observable benefit at this state count).

2. **Greedy Scenario-MPC lookahead.** At each of `horizon` steps, evaluate every
   action's resulting `P(FAILED)` and pick the lowest, adding a small
   `cost_weight` penalty for acting so the planner only intervenes when it
   strictly reduces risk. This is the standard MPC "receding horizon" idea
   applied to a discrete state chain. *Alternative*: exhaustive tree search over
   action sequences — rejected (branching × horizon explodes; greedy suffices).

3. **Cooldowns shared with `SafetyPolicy`.** The planner carries its own
   `cooldowns` map so it will not re-select an action still inside its
   `cooldown_seconds` window; `SafetyPolicy` remains the hard gate at execution.

4. **Execution reuses existing machinery.** `RemediationManager.plan_and_execute`
   calls a new `_apply_named_action` that mirrors the per-action flow in
   `evaluate()` (safety check → `action.execute` → audit trail), so the plan is
   genuinely executed, not merely advised.

## Risks

- **Priors are guesses.** Until `backtester` refines `record_transition`, the
  action effects are conservative. Mitigation: priors are clearly marked and the
  model falls back to the base chain when no action data exists.
- **Greedy myopia.** A single intervention may look suboptimal over a longer
  horizon. Mitigation: `horizon` is configurable (default 3); can be raised.
- **Reactive path coexistence.** Adding a planned path must not change existing
  per-action behaviour. Mitigation: `plan_and_execute` is additive; `evaluate()`
  is untouched.
