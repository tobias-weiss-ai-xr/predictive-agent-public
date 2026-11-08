## 1. Config & action-contract groundwork

- [x] 1.1 Add new env knobs to `predictive_agent/config.py` with documented defaults (`VERIFY_ENABLED=true`, `VERIFY_SETTLING_S=90`, `VERIFY_MIN_SIGNALS=2`, `VERIFY_HOLD_S=600`, `CANARY_MAX_FRACTION=0.15`, `REMEDITATION_REVERT_ENABLED=true`) and verify a `config.py` unit test asserts the parsed values
- [x] 1.2 Extend `RemediationAction` (ABC) with additive attributes `reversible: bool = False`, optional `canary_fraction: float = 0.0`, `resolution_signals: list[str] = []`, an optional `revert(target, prev_state)` hook, and verify a unit test confirms existing concrete actions still construct unchanged (backward compatible)
- [x] 1.3 Extend the dedup-history entry (in `_record_execution`) to also persist the recorded `resolution_signals` + execution-time signal snapshot, and verify an existing remediator unit test still passes (additive fields only)

## 2. Verification observe-path

- [x] 2.1 Add a verification state registry on `RemediationManager` (bounded dict keyed by idempotency key: pending → due → verified/unverified, pruned like `_dedup_history`), and verify unit tests cover pending → due lifecycle and pruning
- [x] 2.2 Implement the verify step in `evaluate()`: when due, re-sample ≥ `VERIFY_MIN_SIGNALS` independent current signals and compare against the recorded criteria, requiring sustained improvement to record `verified`, and verify unit tests cover pass, single-signal-only (→ unverified/hold), and transient-recovery (held-open) cases
- [x] 2.3 Wire `VERIFY_ENABLED=false` and "not due yet" to be strict no-ops, and verify that existing remediator/reconcile tests pass unchanged (behavior-preserving)
- [x] 2.4 Surface verification outcome in the audit trail + `record_remediation_metric` (new `VERIFIED`/`UNVERIFIED` outcomes or equivalent), and verify a test asserts the audit row contains the lifecycle field

## 3. Hold & escalation

- [x] 3.1 Implement hold state: on failed/unverified verification, record hold on the idempotency key for `VERIFY_HOLD_S` and skip the key in `evaluate()` with `outcome=HOLD`, and verify a unit test shows an immediate re-cycle does not re-fire while held and fires after expiry
- [x] 3.2 Call the existing futile-remediation path (`note_remediation_outcome`) on held targets so repeated failures escalate (`remediation_exhausted`), and verify an integration test shows escalation fires after the configured consecutive failed verifications

## 4. Reversibility & revert

- [x] 4.1 Implement revert execution: for a `reversible` action whose verification fails, call `revert(target, prev_state)`, record the revert in the audit trail (through the same SafetyPolicy/dedup path), and verify a unit test with a fake reversible action asserts revert ran once and only once
- [x] 4.2 Enforce the irreversible rule: non-reversible actions under failed verification only proceed via operator approval, and verify a unit test asserts no auto-retry and the review/Escalate path is used
- [x] 4.3 Gate behavior on `REMEDITATION_REVERT_ENABLED` and verify that when disabled, failed verification degrades to hold+escalate (today's semantics)

## 5. Canary widening

- [x] 5.1 Implement canary execution: when `canary_fraction > 0` and target scale > 1, execute action on the canary slice (bounded by `CANARY_MAX_FRACTION`), verify (reuse step 2.2), then widen to the remaining scope, and verify a unit test with a fake fleet action asserts canary leg → verify → widen leg ordering
- [x] 5.2 Implement canary abort: failed canary verification aborts the widen and follows the hold/escalate path, and verify a unit test asserts the widen leg never executes after canary failure
- [x] 5.3 Set `rollout_restart`/`scale` as the canary-capable actions (documented, operator-enable), leaving all other actions at `0.0`, and verify the shipped action registry test still passes

## 6. Operator surfacing & docs

- [x] 6.1 Add alert types `remediation_verified`, `remediation_held`, `remediation_reverted`, `remediation_widened` through the existing notifier with per-pod cooldowns (mirroring plan/autonomy throttles), and verify unit tests assert each new alert type is emitted on its trigger
- [x] 6.2 Add `GET /remediation/state` exposing in-flight verify/hold/canary state, and extend `/remediation-log` rows with the `lifecycle` field, and verify HTTP tests cover the new endpoint and additive JSON field
- [x] 6.3 Update `README.md` env-var table, endpoints table, and `docs/OPERATOR_RUNBOOK.md` with the verification/hold/revert/canary behavior, and verify a `git diff --stat` review shows only the intended files changed
- [x] 6.4 Update `docs/arc42.md` architecture notes (act→verify loop) and mark `docs/BEST_PRACTICES_AUDIT.md` recommendations 1–3 as implemented, and verify the docs build/lint pass

## 7. Full validation

- [x] 7.1 Run `python3.12 -m pytest tests/` and verify the full suite passes (existing 1763-green + new tests; the 4 pre-existing environmental k8s-failures may remain, documented)
- [x] 7.2 Run `openspec validate --all` (or repo-equivalent) and verify the change validates (MODIFIED delta carries all current scenarios), then mark `openspec/changes/verify-after-act-reversible-remediation/tasks.md` fully checked off
