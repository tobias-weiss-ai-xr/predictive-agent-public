# Best-Practice Audit — predictive-agent v4.0.8

**Date:** 2026-11-07 · **Scope:** predictive health-monitoring operator (Kubernetes + Docker), Kalman/Markov trend forecasting, Bayesian risk scoring, POMDP decision layer, LLM analysis, autonomous remediation.

This document maps current industry best practice for this class of system onto
the actual implementation. Every status is verified against the code (anchors
included), not assumed. The audit covers six domains:

| Domain | Sources (citations at end) |
|---|---|
| A. Safe autonomous remediation | derkonline guardrails · OWASP APTS Safety Controls · Fallbrook · Metasphere · zof.ai · AI-Journal remediation governor · Nik Kale circuit breakers · Enteroid |
| B. Predictive / anomaly-detection discipline | Netverge · ONES FP-acceptance · FDAI · zop.dev closed-loop · TechnoLynx · AWS WA DevOps · Emre Bener |
| C. LLM-in-the-loop reliability | AWS GenAI lens · GO4 reliability patterns · OWASP LLM01 · linus-teklenburg synthesis · Google Cloud SRE for LLMs · QITP microservices |
| D. Alerting | AWS EKS · Prometheus Alertmanager + practices · Grafana · Icinga · NeutralBlock · incident.io |
| E. Kubernetes-operator hygiene | Operator SDK · Java Operator SDK · Red Hat · containers.codeguides · kube.rs · tejo.dev |
| F. Prognostics / uncertainty | Celaya & Goebel · Sankararaman · NASA DASHlink PHM metrics |

**Legend:** ✅ followed · 🟡 partial (mechanism exists, not fully wired) · ❌ missing

---

## A. Safe autonomous remediation (the critical class)

| Best practice | Status | Evidence |
|---|---|---|
| Safe default posture: remediation off, dry-run on | ✅ | `config.py:94-95` — `REMEDIATION_ENABLED=false`, `REMEDIATION_DRY_RUN=true` |
| Dry-run / plan preview before executing | ✅ | `ActionContext.dry_run` (`remediator.py:90`), `GET /plan`, `/shadow` |
| Approval gates for high-consequence actions | 🟡 | Review tier + `GET/POST /review` approve/reject, TTL 900 s (`remediator.py:330-347`, `server.py:771-803`). Review tier defaults to `none` → all autonomous unless operator configures it (`config.py:113`) |
| Classify-before-act (action risk taxonomy) | ✅ | `action.risk_tier` (destructive → extra logging, `remediator.py:505-516`); POMDP `DEFAULT_ACTION_COSTS` = blast-radius proxy (`pomdp.py:53-56`) |
| Rate limits + cooldowns + change windows | 🟡 | ✅ 5/min, 50/h, 300 s cooldown, protected namespaces (`config.py:96-98,106`). ❌ no maintenance / change windows |
| Blast-radius limits | 🟡 | ✅ per-cycle breaker (10), per-node (2), per-min/hour, `MAX_PODS_PER_CYCLE=3`, min-confidence gate (`config.py:140-147`). ❌ no fleet-relative caps (e.g. ≤10–15% of a Deployment) |
| Idempotency of every action | ✅ | Idempotency keys + dedup window + stored `desired_state` for convergence (`remediator.py:354-410`) |
| Escalation after N futile attempts | ✅ | `note_remediation_outcome` → `remediation_exhausted` notify; calibration-drift breaker (`reconcile_phases.py:912-947`) |
| Audit trail + decision observability | ✅ | `audit_trail`, `/remediation-log`, `Remediation decision: … outcome=… reason=…` logs, Prometheus metrics |
| **Verify-after-act** (health-gated confirmation, multi-signal, settling window, escalation flag) | ❌ | **Biggest gap.** No post-action verification against recorded resolution criteria. The loop only re-observes next cycle; the closed-loop Detect→Decide→Act→**Verify** pattern (with a hold-not-rerun on failed verify) is unimplemented |
| Canary remediation (small slice first, then widen) | ❌ | Not present |
| Rollback / defined undo per action | ❌ | No per-action undo path (scale-back is not guaranteed, no improvement check). Only *planner*-level rollback on calibration drift |
| Progressive authority (scope grows with demonstrated reliability) | 🟡 | Config-driven review tiers exist; no automatic performance-based scope expansion/contraction |
| Segregation of duties | 🟡 | Approval via separate `/review` endpoint, but no identity separation — anyone with HTTP access can approve |

## B. Predictive / anomaly-detection discipline

| Best practice | Status | Evidence |
|---|---|---|
| Calibrate scores, not sensitivity | ✅ | `calibration.py`: Brier + over-rate, drift → circuit-breaker rollback to reactive (`main.py:208`) |
| Base-rate anchoring | ✅ | `PREDICTION_BASE_RISK` (~15–20% prior, `risk.py:39`) |
| Replay / backtest + shadow validation | ✅ | `backtester.py`, `benchmark.py` (4-strategy), `shadow.py`, `GET /backtest`, `GET /shadow` |
| Formal FP/confidence acceptance criteria before autonomous enablement | 🟡 | `MIN_ACTION_CONFIDENCE` exists but defaults to 0 (gate off, `config.py:146`). No agreed FP-rate limits (per the SRE acceptance-criteria playbooks) before going beyond dry-run |
| Human label feedback → retraining loop | 🟡 | Alert history exists; no operator "confirm/dismiss" labeling feeding calibration / continual learning |
| Multi-signal corroboration / quorum | 🟡 | 6-signal Bayesian fusion ✅; no explicit quorum requirement (a single strong signal can alarm) |
| Flapping prevention / hysteresis | 🟡 | Kalman smoothing + cooldowns soften it; no explicit pending-window / hysteresis on the alert edge |

## C. LLM-in-the-loop reliability

| Best practice | Status | Evidence |
|---|---|---|
| Structured output + schema validation | ✅ | `parse_response` w/ fenced-JSON extraction + typed fallback (`llm.py:188-213`) |
| Typed failure propagation (per failure class) | ✅ | HTTP-reject vs timeout vs connectivity vs generic — distinct degraded dicts, surfaced at WARNING (`llm.py:151-186`) |
| Retry with recovery prompt | ❌ | Single attempt per analyze; graceful degrade instead of retry-with-recovery |
| Circuit breaker per provider + fallback path | ❌ | No LLM-layer breaker/backoff; fallback is a "Monitor" action |
| Semantic cache (TTL, eviction) + call dedup | ✅ | `l1_cache.py` (TTL 300 s, size eviction); `deduplicate_llm_call` (`llm_batch.py:220`) |
| Batching / bulkhead | 🟡 | Batch processor ✅; no explicit concurrency quotas protecting the core loop |
| Bounded loops / tokens / cost | 🟡 | `max_tokens=4096`, timeout, per-cycle pod caps ✅; token/cost counts are rough (`len/4`), not real usage |
| Prompt-injection defense | 🟡 | Input is internal metrics (low exposure); no untrusted-content segregation (low priority) |

## D. Alerting

| Best practice | Status | Evidence |
|---|---|---|
| Dedup, rate limiting, grouping by incident | 🟡 | Plan/autonomy notifies throttled per-pod (300 s / 600 s, `reconcile_phases.py:620-633,795-807`); self-health per-kind cooldown + streak detection (`self_health.py`). **Remediation notify is unthrottled** — bounded only indirectly by action rate limits |
| Severity routing / pages-vs-tickets | ❌ | Single email+webhook channel; no severity → channel mapping |
| Maintenance-window suppression | ❌ | Not present |
| Alert tuning / review KPI workflow | ❌ | No FP / ack-rate tracking or review cadence |

## E. Kubernetes-operator / reconcile hygiene

| Best practice | Status | Evidence |
|---|---|---|
| Idempotent, level-triggered reconcile | ✅ | Poll loop, idempotent actions, exception-safe (`main.py:311-328`) |
| Backoff on failure | 🟡 | Fixed 60 s reschedule — no exponential backoff on DEGRADED collector streaks |
| RBAC least privilege | 🟡 | Resource-scoped, but cluster-wide `delete`/`patch` (`k8s/rbac.yaml`) — could be namespace-scoped to `WATCH_NAMESPACES` |
| Leader election / HA | 🟡 | `replicas: 1`, no leader election (acceptable as designed) |
| Monitor the operator itself | ✅ | Exceptional: self-health alerting + degraded-cycle streaks + external watchdog CronJob + `/healthz` `/ready` + `/metrics` (see README "Self-Health Alerting") |
| Own-goal alerting (alert on alerting) | 🟡 | Self-health covers persistence/collector; no alert on notifier failure |

## F. Prognostics / uncertainty

| Best practice | Status | Evidence |
|---|---|---|
| Uncertainty on predictions (not point estimates) | ✅ | Kalman covariance, Markov `predictive_interval`, `state_support`/`is_confident`, TTF with confidence + 43 200 min cap (`predictor.py`) |
| Forecast-quality metrics (prognostic horizon, α-λ) | 🟡 | Brier / backtester approximate; no per-horizon PH / α-λ metrics |

## Hygiene note (test suite)

`python3.12 -m pytest tests/` → **1763 passed, 4 failed, 3 skipped**. The four
failures are environmental: the dev host lacks `kubectl` on PATH
(`kubectl binary not found on PATH — kubernetes-mode collection cannot run`),
so kubernetes-mode integration tests fail locally while the image now ships
kubectl (commit `648621c`). Pre-existing; not caused by any audit finding.
Consider `pytest.importorskip("kubectl")`-style guard or adding kubectl to the
dev environment.

---

## Prioritized recommendations (highest-value gaps)

> **Status legend:** ✅ implemented (commit `verify-after-act-reversible-remediation`),
> 🟡 partial, ■ backlog. Implemented items are described in README
> ("Closed-Loop Remediation") and `docs/OPERATOR_RUNBOOK.md`
> ("Verify-after-act & reversibility").

1. **Verify-after-act loop** ✅ — after an executed action, the manager re-samples
   ≥ `VERIFY_MIN_SIGNALS` (2) independent signals after `VERIFY_SETTLING_S` (90 s),
   confirms sustained improvement, and on failure **holds + escalates**
   (`VERIFY_HOLD_S` 600 s, no blind re-fire; `remediation_exhausted` via the
   existing futile channel). Most evidence-backed missing practice for
   closed-loop AIOps. *(zop.dev; AI-Journal; Enteroid; Metasphere)*
2. **Per-action reversibility metadata** ✅ — `reversible: bool` + `revert(target,
   prev_state)` hook per action (scale restores recorded `prev_replicas`);
   auto-revert on failed verification (`REMEDITATION_REVERT_ENABLED`), with an
   operator-approval path for irreversible actions. *(derkonline; zof.ai "armed rollback")*
3. **Canary for fleet-consequence actions** ✅ — `canary_fraction` + two-leg
   execution (bounded leg → verify → widen), clamped by `CANARY_MAX_FRACTION`
   (0.15). Enabled on `scale`; `rollout_restart` documented operator-enable at 0.0.
   *(derkonline; Metasphere)*
4. **Fleet-relative blast radius** ■ — cap per-action impact as a % of a
   Deployment's pods, not just absolute counts. *(Metasphere 10–15% rule; OWASP APTS)*
5. **LLM resilience** ■ — retry-with-recovery-prompt + per-provider circuit
   breaker (currently single-shot graceful degradation). *(GO4 V19/V20; AWS GenAI lens; Google SRE)*
6. **Alerting discipline** 🟡 — verification/outcome alerts throttled per target +
   kind; formal dedup window + incident grouping + severity routing remain
   backlog. *(Prometheus/Alertmanager; Grafana; ONES; Netverge)*
7. **Operator feedback loop** ■ — "confirm / dismiss" labeling endpoint feeding
   calibration, plus reconcile backoff on degraded collector streaks.
   *(Netverge HITL; AWS WA O.CM.10; TechnoLynx)*

## Decision: which gaps become a change

A single coherent capability change covers recommendations **1–3** — they are
one closed-loop unit: *act → verify → confirm / hold / revert / canary*. That
is the top cluster and the natural first implementation. Items 4–7 remain
backlog for follow-up changes.

---

## Sources

**Autonomous remediation:** derkonline.com/blog/self-healing-infrastructure-guardrails ·
owasp.github.io/APTS/standard/2_Safety_Controls/ · fallbrookresearch.com/2025/11/15/autonomous-devops-self-healing-infrastructure ·
msphere.io/insights/automated-remediation-self-healing/ · zof.ai/blog/when-should-an-agent-defer-confidence-scoring-and-human-authorization-for-remediation ·
aijourn.com/kubernetes-reliability-in-the-age-of-automation · nikkale.substack.com/p/the-blast-radius-problem-designing ·
enterprisesingularity.com/resources/blogs/aiops-safe-autonomy-is-architecture
**Anomaly detection / AIOps:** netverge.com/blog/anomaly-detection-tuning · ones.com/blog/how-sre-teams-define-and-verify-false-alert-acceptance-criteria ·
dotnetpower.github.io/fdai/sre/observability-detection-and-forecasting/ · zop.dev/resources/blogs/closed-loop-aiops-detect-decide-act-verify ·
technolynx.com/post/reliability-engineering-for-anomaly-detection-systems · docs.aws.amazon.com/wellarchitected/latest/devops-guidance/o.cm.10-proactively-detect-issues-using-aiml.html ·
emrebener.com/blogs/infrastructure-and-operations/reliability-engineering/anomaly-detection-and-aiops
**LLM in ops:** docs.aws.amazon.com/wellarchitected/latest/generative-ai-lens/genrel03-bp01.html ·
jlldavies.github.io/go4-llm-design-patterns/patterns/RELIABILITY.html · genai.owasp.org/llmrisk/llm01-prompt-injection/ ·
linus-teklenburg.de/reliability · medium.com/google-cloud/building-bulletproof-llm-applications ·
qitpress.com/articles/QITP-IJAIDLRD/VOLUME_6_ISSUE_2/QITP-IJAIDLRD_06_02_002.pdf
**Alerting:** docs.aws.amazon.com/prescriptive-guidance/latest/amazon-eks-observability-best-practices/alerting-best-practices.html ·
prometheus.io/docs/alerting/latest/alertmanager/ · prometheus.io/docs/practices/alerting/ · grafana.com/docs/grafana/latest/alerting/guides/best-practices/ ·
icinga.com/blog/alert-fatigue-monitoring/ · neutralblock.com/learn/sre/sre-chapter-2/alerting-philosophy-and-best-practices ·
incident.io/blog/2025-guide-to-preventing-alert-fatigue-for-modern-on-call-teams
**Operators:** sdk.operatorframework.io/docs/best-practices/best-practices/ · javaoperatorsdk.io/docs/getting-started/patterns-best-practices/ ·
redhat.com/en/blog/kubernetes-operators-best-practices · containers.codeguides.io/crds-operators/best-practices/ ·
kube.rs/controllers/reconciler/ · tejo.dev/articles/writing-kubernetes-controllers/
**Prognostics / PHM:** papers.phmsociety.org/index.php/phmconf/article/view/2110 (Celaya & Goebel) ·
doi.org/10.1016/j.ymssp.2014.05.029 (Sankararaman) · c3.ndc.nasa.gov/dashlink/static/media/publication/2010_IEEEAerospace_MetricsWUncertainty.pdf
