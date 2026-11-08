# Operator Runbook — Predictive Agent Decision Layer

This runbook covers running and operating the POMDP decision layer on the
`predictive-agent`. It assumes the base predictor (Kalman/Markov/Bayesian risk)
is already running.

## 1. Local spin-up / smoke test
```bash
# Validate the decision layer end-to-end without a cluster:
python scripts/local_demo.py

# Or run the focused integration suite:
pytest tests/test_reconcile_integration.py tests/test_decision_layer_integration.py -q
```

## 2. Endpoints quick reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/plan?state=CRITICAL&strategy=planner` | GET | Preview the chosen remediation plan. Strategies: `none`, `reactive`, `planner` (default), `policy`, `q`. |
| `/plan?state=CRITICAL&strategy=q` | GET | Q-learning (model-free) rollout. |
| `/coordinate` | POST | Dec-POMDP multi-pod joint plan. Body: `{"pods": {"ns/pod": {"state": "...", "node": "...", "deployment": "..."}}}` (#9). |
| `/benchmark` | GET | Compares all strategies on real backtester data (#4). |
| `/cost-model` | GET/POST | View / recalibrate decision costs from incident data (#5). |
| `/calibration` | GET | Calibration monitor state (Brier, over-rate, drift) (#3). |
| `/shadow` | GET | Shadow-mode validation: planner-vs-executed disagreements (#1). |
| `/decision-metrics` | GET | Decision-layer metrics as JSON (#7). |
| `/remediation-log` | GET | Remediation audit trail (per-row `lifecycle`, incl. held/reverted/canary/widened verdicts). |
| `/remediation/state` | GET | In-flight verify/hold/canary state; per-entry status, hold remaining, failures (#12). |
| `/metrics` | GET | Prometheus scrape format (includes decision metrics) (#7). |

All plans are **execution-ready** but are still gated by `SafetyPolicy` and each
action's `should_execute()` before anything runs on a real cluster.

## 3. Key configuration knobs (env vars)

### Behaviour
- `PLANNING_ENABLED` (`true`) — master switch for the planner path.
- `PLANNING_HORIZON` (`3`) — look-ahead steps.
- `CRASHLOOP_POLICY` (`escalate`) — `escalate` (default) pages a human after N
  futile at-risk cycles via `remediation_exhausted`; `persist_restart` keeps the
  reactive restart path active on every cooldown (demos / churn-over-silence).
- `CRASHLOOP_ESCALATE_AFTER` (`3`) — consecutive at-risk cycles with only `none`
  decided before escalation.

### Safety & trust
- `MIN_ACTION_CONFIDENCE` (`0.0`) — predictor-confidence gate; raise (e.g. `0.6`)
  to force manual review of low-confidence actions.
- `MIN_MODEL_SUPPORT` (`0.0`) — planner only trusted from states with at least
  this many observed transitions; otherwise reactive fallback (#10).
- `MAX_ACTIONS_PER_RECONCILE` (`10`) — per-cycle blast-radius cap.
- `MAX_ACTIONS_PER_NODE` (`2`) — per-node concurrency cap in `/coordinate` (#9).
- `AUTO_ROLLBACK_ON_DRIFT` (`false`) — flip the planner to reactive on detected
  calibration drift.

### Verify-after-act & reversibility (closed loop)
- `VERIFY_ENABLED` (`true`) — post-action verification master switch.
  Observe-only when off; execution snapshots are still recorded.
- `VERIFY_SETTLING_S` (`90`) — settling window before signals are re-sampled.
- `VERIFY_MIN_SIGNALS` (`2`) — independent signals that must show sustained
  improvement for `verified`; fewer → held, never a blind pass.
- `VERIFY_HOLD_S` (`600`) — hold duration after a failed verification; the
  same action is not re-fired while held (re-arm = fresh retry after expiry).
- `VERIFY_ESCALATE_AFTER` (`2`) — consecutive failed verifications before the
  manager feeds the futile path and raises `remediation_exhausted`.
- `CANARY_MAX_FRACTION` (`0.15`) — bounds a canary leg's share of the workload.
- `REMEDITATION_REVERT_ENABLED` (`true`) — master switch for auto-revert of
  reversible actions (`reversible=true`, e.g. `scale`) on failed verification.

### Calibration (#3)
- `CALIBRATION_WINDOW` (`100`), `CALIBRATION_BRIER_ALERT` (`0.25`),
  `CALIBRATION_OVER_RATE_ALERT` (`0.5`), `CALIBRATION_MIN_SAMPLES` (`20`).

### Continual learning (#10)
- `MODEL_DECAY_INTERVAL` (`0`) — decay learned transitions every N reconciles.
- `MODEL_DECAY_FACTOR` (`0.99`) — forgetting factor per step (1.0 = off).
- `DRIFT_DECAY_FACTOR` (`0.5`) — steeper forgetting applied on drift detection.

### Cost model (#5)
- `SLO_COST_PER_MINUTE`, `INCIDENT_TYPICAL_OUTAGE_MIN`, `ACTION_DISRUPTION_FRACTION`
  — derive `failure_cost` / `action_costs` from ops data.

## 4. Reading the metrics (`/decision-metrics`, `/metrics`)
- `decision_layer_interventions_total` — actions executed by the planner.
- `decision_layer_mean_p_failed_reduction` — expected P(FAILED) avoided vs doing
  nothing. A healthy, learning agent trends this upward.
- `decision_layer_learning_transitions_total` — size of the learned model.
- `decision_layer_calibration_brier` / `_drift` / `_samples` — prediction
  reliability. `_drift == 1` means the guardrail has engaged.
- `decision_layer_degraded` — `1` when the planner is in safe reactive fallback.
- `decision_layer_mean_confidence` — mean predictor confidence.

## 5. Common operational scenarios

### Crash-loop / futile remediation (agent detects but stops acting)
- A pod that stays `CRITICAL`/`at risk` while the planner keeps choosing
  `none` is usually a **deterministic crash-loop**: a restart resets it, but it
  re-crashes immediately, so the cost model correctly sees **no durable
  P(FAILED) reduction** and suppresses the action. This is learned behaviour,
  not a stall.
- The agent does **not** go silent: every such cycle is logged
  (`Remediation: none on <pod> — planner: no beneficial action …`) and, after
  `CRASHLOOP_ESCALATE_AFTER` (default 3) consecutive at-risk cycles, it emits a
  `remediation_exhausted` alert + the
  `decision_layer_remediation_exhausted_total` metric so a human is paged.
- Prefer churn over silence? Set `CRASHLOOP_POLICY=persist_restart` — the agent
  then keeps using the reactive restart path on every cooldown instead of
  escalating. (Good for demos; not recommended for production crash-loops.)
- To verify escalation fired: `curl -s localhost:8080/metrics | grep
  remediation_exhausted_total`.

### Plan is empty / agent "does nothing"
- Check `MIN_ACTION_CONFIDENCE` / `MIN_MODEL_SUPPORT` — the agent may be
  correctly deferring (uncertain or insufficient data). Use `/benchmark` to
  compare strategies and confirm the planner *would* act given more data.
- If `decision_layer_degraded == 1`, calibration drift has engaged the fallback;
  see `/calibration`.

### Suspected bad remediation
- Review `/shadow` — it shows where the planner disagreed with what was executed
  (observe-only, never acts).
- Tighten `MIN_ACTION_CONFIDENCE` or enable `AUTO_ROLLBACK_ON_DRIFT`.

### Action executed but no improvement (verify-after-act)
- `GET /remediation/state` lists every execution awaiting verification, on hold
  (with remaining hold time), or widened. An entry with `verify=held` means the
  action returned no sustained improvement after `VERIFY_SETTLING_S` — it will
  **not** re-fire for `VERIFY_HOLD_S`.
- Reversible actions (`scale`) are auto-reverted on failed verification (alert
  `remediation_reverted`); irreversible ones queue a `POST /review` approval
  item (alert `remediation_exhausted` after `VERIFY_ESCALATE_AFTER` failures).
- Approving the review signature clears the hold, letting the agent retry.
- Expected alerts: `remediation_verified`, `remediation_held`,
  `remediation_reverted`, `remediation_widened`, `remediation_canary`,
  `remediation_exhausted` (throttled per target + kind).

### Calibration drift alert
- `/calibration` shows Brier/over-rate. The agent auto-decays stale transitions
  (`DRIFT_DECAY_FACTOR`) and falls back to reactive remediation. Re-enable the
  planner by confirming calibration recovers (drift → 0).

### Tuning costs
- `POST /cost-model` with real incident downtime to recalibrate
  `failure_cost` / `action_costs` from SLO economics; the planner then intervenes
  only when risk reduction is worth the (real) cost.

## 6. Safety posture
- The planner never executes outside `SafetyPolicy` + `should_execute`.
- `MAX_ACTIONS_PER_RECONCILE` bounds blast radius per cycle.
- Shadow mode and calibration guardrails are observe-only / fail-safe by default.
- Nothing learned from `none` (do-nothing) outcomes — the baseline never
  "learns" to do nothing from itself.

## 7. Alert response (self-health + watchdog)

The agent emails its own failure modes. Alerts arrive as mail to
`ALERT_EMAIL_TO` via the configured `SMTP_HOST` relay (no auth; the From
address must be a resolvable domain or the relay rejects with `550 5.1.8`).

| Mail / signal | Meaning | First response |
|---|---|---|
| `[self-health:persistence]` | decision-model or state save is failing | `kubectl -n openedusuite logs deploy/openedusuite-predictive-agent \| grep -i persist`; check PVC mount + ownership (uid 1000 / `fsGroup 1000`; known incident 2026-11-06: root-owned ceph-rbd contents blocked writes after the non-root migration — `fsGroupChangePolicy: Always` fixed it) |
| `[self-health:collector]` | ≥5 consecutive DEGRADED collect cycles | `kubectl top pods -A \| head`; node pressure / kubectl breakage; single DEGRADED cycles are kubectl noise — the streak mail is the actionable one |
| Watchdog mail | `/ready` not `ready` for a probe | `kubectl -n openedusuite get pods \| grep predictive-agent`; frequent causes: Kyverno admission blocking the new pod, image pull from the node-registry mirror failing |
| Watchdog mail repeats every 30 min | ongoing outage | the CronJob stamps `lastNotified` in CM `predictive-agent-watchdog-state`; one mail per 30 min per outage by design — silence after a mail means either recovery (stamp cleared) or continued suppression |
| No mail but suspicion of agent death | agent down while watchdog also broken | run the watchdog manually: `kubectl -n openedusuite create job --from=cronjob/openedusuite-predictive-agent-watchdog watchdog-manual-$(date +%s)` and read its logs |

Design notes: `/ready` 503 does **not** restart the pod (liveness is `/healthz`)
— that gap is covered by the degraded-streak mail and the watchdog. The
watchdog runs as a least-privilege CronJob *outside* the agent precisely so it
still works when the agent is dead.
