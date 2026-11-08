# Design — Verify-after-act & reversible remediation

## Context

The remediator already executes actions only through `SafetyPolicy` (rate
limits, cooldowns, protected namespaces), dedups via idempotency keys +
`REMEDIATION_DEDUP_WINDOW`, records desired state, and escalates after futile
remediation (`reconcile_phases.py:912-947` regression-escalation +
`note_remediation_outcome`). What is missing is the post-action half of the
loop: verification against recorded resolution criteria, a hold state that
prevents re-firing the same failed action, an explicit undo, and canary
widening for fleet-consequence actions. See proposal.md — Why.

Design constraints from the repo:
- stdlib-only runtime; no new third-party dependencies.
- Config via env vars in `config.py`; every new knob needs a safe default.
- All new behavior must be opt-in / behavior-preserving: existing actions keep
  today's semantics until they declare reversibility or a canary fraction.
- Follow this repo's OpenSpec conventions: one capability (`predictive-agent`), delta spec with MODIFIED requirements carrying the full updated content, and a pytest suite (`python3.12 -m pytest tests/`) for every behavioral change.

## Goals / Non-Goals

**Goals:**
- Close the act→verify loop with a settling window, ≥2 independent signals,
  recorded resolution criteria, and a hold + escalation path on failure.
- Give remediation actions an explicit undo (reversibility metadata) and
  auto-revert on failed verification.
- Canary gating for actions that can affect more than a bounded slice of a
  workload.
- Surface the full remediation life-cycle (planned → executed → verified /
  held / reverted / widened) to operators via alerts, `/remediation-log`, and
  a status endpoint.

**Non-Goals:**
- Not adding LLM retry/circuit breakers, alert dedup/grouping, fleet-relative
  % caps, or the operator labeling loop — those stay backlog per the audit.
- No change to the POMDP planner's planning logic or the 6-signal risk score.
- No new permissions in `k8s/rbac.yaml` (rollback reuses existing patch/scale
  verbs; scale-back uses `deployments/scale` patch already granted).
- No multi-cycle in-flight action state machine persisted to disk in v1.
  In-memory hold/verify state on `RemediationManager` (agent is single
  replica; process restart loses in-flight verification state, which degrades
  to "no verification" for that action — acceptable and logged).

## Decisions

**D1 — Where verification lives.** Verification is a phase inside
`RemediationManager.evaluate` / `plan_and_execute`, keyed by the existing
`idempotency_key` (`ns/target/action/window`). When an action already ran for
that key within `REMEDIATION_DEDUP_WINDOW`, the manager records the resolution
criteria (the signal values captured at plan time: risk signals, markov
`p_failed`, chosen signals) in the dedup history entry. On a later cycle
(after the settling window), the verify step re-samples the target's current
signals and compares.

- *Alternative considered:* a separate `Verifier` module driven from
  `reconcile_phases`. Rejected — the verification is a property of a
  remediation event, and the manager already owns the event lifecycle
  (dedup, hold, audit). Keeps one owner.
- *Alternative considered:* persistent verify state. Rejected for v1
  (non-goal) — in-memory `dict[key -> verify_state]` with bounded pruning,
  mirroring `_dedup_history` and `_pending`.

**D2 — Multi-signal, sustained confirmation (not a single re-sample).**
Verification requires ≥2 independent signal types to show the recorded
improvement criterion (e.g. `memory%` below the recorded level *and* no new
restarts) across both the settling-window re-sample. Single-signal transient
recovery (the classic "memory drops for 45 s then the leak resumes") must not
close the loop. Config: `VERIFY_MIN_SIGNALS` (default 2), `VERIFY_SETTLING_S`
(default 90). If fewer than 2 signals are available for a target, verification
is recorded as "unverified" and the target follows the hold/escalate path —
never a blind pass.

**D3 — Hold, don't re-fire.** On failed verification the manager records a
hold on the idempotency key for `VERIFY_HOLD_S` (default 600) and raises an
`escalation` flag. While held, `evaluate()` skips that key (logs
`outcome=HOLD`). It also calls the existing `note_remediation_outcome(...)`
futile path explicitly, so repeated failures compound into
`remediation_exhausted` escalation to operators. This realizes the
zop.dev "re-entry is a hold, not an immediate re-act" rule without a new
escalation channel.

**D4 — Reversibility as declarative metadata on the action.**
`RemediationAction` gains optional attributes: `reversible: bool = False` and
`revert(self, target, prev_state)` hook, where `prev_state` is what the action
records in its dedup entry at execution time (e.g. prior `scale` replicas,
prior `rollout_restart` marker). On failed verification of a `reversible`
action, the manager calls `revert`, records outcome in the audit trail, and
reapplies the safety policy + dedup (a revert is itself a recorded op).
Existing actions keep `reversible=False` — they degrade to hold+escalate
(approved-retry only), giving us the "irreversible ⇒ approval" rule for free.

**D5 — Canary as a property of fleet-consequence actions.**
`RemediationAction` gains `canary_fraction: float = 0.0` (0 = no canary,
applies to e.g. `rollout_restart`, `scale` beyond a pod). When set and the
target exposes a scale count > 1, the manager executes the action scoped to
the canary slice, verifies D2, and only then executes the remaining scope
(each leg through the same SafetyPolicy/dedup path). A failed canary leg
aborts the widen and follows D3. Config: `CANARY_MAX_FRACTION` default 0.15
(≥ Metasphere's conservative 10–15 % guidance).

**D6 — Operator surfacing.** New alert types `remediation_verified`,
`remediation_held`, `remediation_reverted`, `remediation_widened` sent through
the existing notifier with per-pod cooldowns (as plans/autonomy already do).
`GET /remediation-log` rows gain a `lifecycle` field; a
`GET /remediation/state` endpoint exposes in-flight verify/hold/canary state.

**D7 — Configuration defaults.** All new env knobs default to the least
intrusive behavior:
`VERIFY_ENABLED=true` (verification is observation-only until an action opts
into revert/canary; it cannot make today's flows worse), `VERIFY_SETTLING_S=90`,
`VERIFY_MIN_SIGNALS=2`, `VERIFY_HOLD_S=600`, `CANARY_MAX_FRACTION=0.15`,
`REMEDITATION_REVERT_ENABLED=true`. No existing env var changes.

## Risks / Trade-offs

- [Verification adds one reconcile-cycle latency before an action is
  "verified"] → settled by async state: the verify step runs on the next
  cycle's `evaluate()` without blocking the loop; a not-yet-due verification
  is a no-op.
- [A transient recovery at exactly the re-sample closes the loop (missed
  verification false-negative)] → multi-signal sustained check (D2) + the
  existing Markov/continued monitoring still catch re-degradation; the next
  cycle re-detect re-enters the loop.
- [Replay/sidecar OPA path sees new lifecycle fields and must stay
  tolerant] → additive fields only; JSON consumers use `.get()` (repo
  convention).
- [Feature interplay: verification could suppress legitimate re-actions
  (dedup window vs hold)] → verification runs inside the dedup window by
  design: an action that verified is legitimately deduped; an action that
  failed verification is *not* recorded as executed (dedup only records
  success today) so the hold — not the dedup — governs retry timing.
- [In-memory verify state is lost on restart (D1 non-goal)] → degraded to
  "not verified" with a log line; the external-runbook/watchdog layer
  (already present) covers operator awareness.

## Migration Plan

1. Land the config knobs and `RemediationAction` attribute additions
   (backward-compatible, default no-op).
2. Add verification observe-path (D1/D2) — behavior observability only,
   dry-run safe; verify with existing tests + new unit tests.
3. Add hold/escalation (D3) and reversibility + revert (D4) — gated on
   actions that opt in.
4. Add canary (D5) for `rollout_restart`/`scale` only, then surface (D6) and
   document in README + OPERATOR_RUNBOOK.
5. Rollback: each stage is independently revertible (env-var gated); none
   changes persisted schemas.

## Open Questions

- Whether `scale` should list `canary_fraction` in its shipped defaults or
  leave it operator-configured per deployment. Default decision recorded:
  leave 0.0 in shipped defaults; document how to enable.
