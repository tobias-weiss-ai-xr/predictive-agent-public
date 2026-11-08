# Verify-after-act & reversible remediation

## Why

The audit in `docs/BEST_PRACTICES_AUDIT.md` found the agent's act-and-verify
loop is its biggest safety gap: after an action executes, nothing confirms it
*worked* against recorded resolution criteria, nothing holds before re-acting,
and actions have no defined undo. Closed-loop AIOps guidance (Detect → Decide →
Act → **Verify**; armed rollback; canary-first) is unanimous that an autonomous
remediation system which cannot verify, hold, or revert its own actions is not
safe to leave unsupervised.

## What Changes

- **Verify-after-act**: after an executed remediation passes its cooldown, the
  agent re-samples ≥ 2 independent signals after a settling window and compares
  them against the resolution criteria recorded at plan time. Only sustained
  improvement closes the loop.
- **Hold, don't re-fire**: a failed verification does NOT immediately re-trigger
  the same action. It enters a hold period and escalates to operators with an
  escalation flag, and the futile-remediation counter is fed so repeated
  failures fast-track human attention.
- **Reversible actions**: each remediation action declares reversibility
  metadata (e.g. prior scale/replicas, prior restart count). When verification
  fails, reversible actions are reverted automatically; irreversible actions
  require approval to retry.
- **Canary for fleet-consequence actions**: restart/scale actions that could
  affect more than a bounded slice of a workload run on a small first slice,
  verify, then widen.

## Capabilities

### New Capabilities
- *(none — this extends the existing `predictive-agent` capability)*

### Modified Capabilities
- `predictive-agent`: the "Remediation framework and actions" requirement
  gains verification, hold/escalation, reversibility, and canary scenarios.

## Impact

- `predictive_agent/remediator.py` — verification phase in
  `plan_and_execute`/`evaluate`; reversibility metadata on actions; canary
  gating; hold state.
- `predictive_agent/reconcile_phases.py` — orchestration of verify step and
  escalation signaling.
- `predictive_agent/config.py` — new env knobs (verification settling window,
  signal count, canary fraction, hold duration), all with safe defaults.
- `predictive_agent/server.py` — verify/status endpoints for operators
  (remediation life-cycle view).
- `predictive_agent/self_health.py` / `notifier.py` — alert types for
  "verification failed" / "reverted".
- `tests/` — new tests for verification, hold, revert, canary.
- **Non-breaking**: all new behavior defaults to off or to today's behavior
  (verification adds observation only; revert requires the action to declare
  reversibility, which existing actions do not — they keep today's semantics
  until opted in).
