## MODIFIED Requirements

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
