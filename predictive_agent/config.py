"""Configuration for predictive-agent v4.0."""

import json
import os

# ─── Core ────────────────────────────────────────────────────────────────────
OPERATOR_NAME = os.environ.get("OPERATOR_NAME", "predictive-agent")
OPERATOR_NAMESPACE = os.environ.get("OPERATOR_NAMESPACE", "predictive-agent")
OPERATOR_VERSION = os.environ.get("OPERATOR_VERSION", "4.1.0")
WATCH_NAMESPACES = os.environ.get(
    "OPERATOR_WATCH_NAMESPACES", "openedusuite,default,llm"
).split(",")
SKIP_NAMESPACES = {"predictive-agent"}

# ─── LLM Backend ─────────────────────────────────────────────────────────────
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")  # ollama, saia, openai
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama.llm.svc.cluster.local:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-30b-a3b:latest")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "300"))

SAIA_API_URL = os.environ.get("SAIA_API_URL", "https://api.saia.ai/v1")
SAIA_API_KEY = os.environ.get("SAIA_API_KEY", "")
SAIA_MODEL = os.environ.get("SAIA_MODEL", "qwen3.5-35b-a3b")

OPENAI_API_URL = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

TUD_API_URL = os.environ.get("TUD_API_URL", "https://api.tud-llm.de/v1")
TUD_API_KEY = os.environ.get("TUD_API_KEY", "")
TUD_MODEL = os.environ.get("TUD_MODEL", "GLM-5.2-AWQ-INT4")

# ─── Reconcile ───────────────────────────────────────────────────────────────
RECONCILE_INTERVAL = int(os.environ.get("RECONCILE_INTERVAL", "60"))
ANALYSIS_TTL = int(os.environ.get("ANALYSIS_TTL", "300"))
ANALYSIS_TTL_MAX = int(os.environ.get("ANALYSIS_TTL_MAX", "1200"))
MAX_PODS_PER_CYCLE = int(os.environ.get("MAX_PODS_PER_CYCLE", "3"))
LOG_VERBOSITY = os.environ.get("LOG_VERBOSITY", "info")

# LLM analysis: only analyze pods with risk >= this threshold
LLM_ANALYSIS_THRESHOLD = float(os.environ.get("LLM_ANALYSIS_THRESHOLD", "0.7"))
# LLM analysis TTL: re-analyze a pod only after this many seconds
LLM_ANALYSIS_TTL = int(os.environ.get("LLM_ANALYSIS_TTL", "300"))

# ─── Prediction ──────────────────────────────────────────────────────────────
PREDICTION_ENABLED = os.environ.get("PREDICTION_ENABLED", "true").lower() == "true"
PREDICTION_RISK_THRESHOLD = float(os.environ.get("PREDICTION_RISK_THRESHOLD", "0.5"))
PREDICTION_BASE_RISK = float(os.environ.get("PREDICTION_BASE_RISK", "0.15"))

# ─── Kalman Filter ───────────────────────────────────────────────────────────
KALMAN_PROCESS_NOISE = float(os.environ.get("KALMAN_PROCESS_NOISE", "1.0"))
KALMAN_MEASUREMENT_NOISE = float(os.environ.get("KALMAN_MEASUREMENT_NOISE", "100.0"))
# Per-metric Kalman parameters (override defaults for finer control)
KALMAN_MEMORY_PROCESS_NOISE = float(os.environ.get("KALMAN_MEMORY_PROCESS_NOISE", "0.5"))
KALMAN_MEMORY_MEASUREMENT_NOISE = float(os.environ.get("KALMAN_MEMORY_MEASUREMENT_NOISE", "50.0"))
KALMAN_CPU_PROCESS_NOISE = float(os.environ.get("KALMAN_CPU_PROCESS_NOISE", "5.0"))
KALMAN_CPU_MEASUREMENT_NOISE = float(os.environ.get("KALMAN_CPU_MEASUREMENT_NOISE", "200.0"))

# ─── Persistence ─────────────────────────────────────────────────────────────
STATE_MODEL_FILE = os.environ.get("STATE_MODEL_FILE", "/var/lib/predictive-agent/state-model.json")
PREDICTIONS_FILE = os.environ.get("PREDICTIONS_FILE", "/var/lib/predictive-agent/predictions.json")
HISTORY_FILE = os.environ.get("HISTORY_FILE", "/var/lib/predictive-agent/analysis-history.json")
HISTORY_MAX = int(os.environ.get("HISTORY_MAX", "100"))

# ─── HTTP ────────────────────────────────────────────────────────────────────
HEALTH_PORT = int(
    os.environ.get("OPERATOR_HEALTH_PROBE_BIND_ADDRESS", "0.0.0.0:8081").split(":")[-1]
)
METRICS_PORT = int(
    os.environ.get("OPERATOR_METRICS_BIND_ADDRESS", "0.0.0.0:8080").split(":")[-1]
)

# ─── Collector Mode (Docker backend support) ────────────────────────────────
COLLECTOR_MODE = os.environ.get("COLLECTOR_MODE", "kubectl")  # kubectl, docker
DOCKER_SOCKET = os.environ.get("DOCKER_SOCKET", "/var/run/docker.sock")
DOCKER_HOST_PRESSURE_THRESHOLD = float(os.environ.get("DOCKER_HOST_PRESSURE_THRESHOLD", "0.9"))

# ─── Pod exclusions ───────────────────────────────────────────────────────────
# Pods matching any configured label or annotation selector are excluded from
# monitoring entirely (prediction, at-risk, LLM, remediation, escalation). Keys
# are evaluated against pod metadata.labels / metadata.annotations; a bare key
# (no "=") matches any value, a "key=value" matches exactly.
#
# Default ships with the KubeVirt launcher exclusion so VM pods are never auto-
# remediated: virt-launcher memory/CPU reflect a guest OS (Windows uses ~all of
# its allocation by design) and restarting a launcher pod is a disruptive VM
# reboot that yields no durable P(FAILED) reduction.
EXCLUDED_POD_LABELS: dict = {}
_excluded_labels_raw = os.environ.get("PREDICTIVE_AGENT_EXCLUDED_POD_LABELS", "").strip()
for _part in (p.strip() for p in _excluded_labels_raw.split(",") if p.strip()):
    if "=" in _part:
        _k, _v = _part.split("=", 1)
        EXCLUDED_POD_LABELS[_k.strip()] = _v.strip()
    else:
        EXCLUDED_POD_LABELS[_part] = None  # any value matches

EXCLUDED_POD_ANNOTATIONS: dict = {}
_excluded_anno_raw = os.environ.get("PREDICTIVE_AGENT_EXCLUDED_POD_ANNOTATIONS", "").strip()
for _part in (p.strip() for p in _excluded_anno_raw.split(",") if p.strip()):
    if "=" in _part:
        _k, _v = _part.split("=", 1)
        EXCLUDED_POD_ANNOTATIONS[_k.strip()] = _v.strip()
    else:
        EXCLUDED_POD_ANNOTATIONS[_part] = None  # any value matches

# ─── Unhealthy pod statuses ──────────────────────────────────────────────────
UNHEALTHY_STATUSES = {
    "CrashLoopBackOff", "Error", "OOMKilled", "ImagePullBackOff", "ErrImagePull",
    "ContainerCreating", "PodInitializing", "CreateContainerError",
    "CreateContainerConfigError", "RunContainerError", "InvalidImageName",
    "RegistryUnavailable", "Evicted", "Pending", "Failed", "Unknown",
}

SKIP_LOGS_STATUSES = {
    "ImagePullBackOff", "ErrImagePull", "ContainerCreating",
    "PodInitializing", "CreateContainerError", "CreateContainerConfigError", "Pending",
    # Succeeded pods are skipped from evaluation entirely — never pay a
    # kubectl logs call for them (observed 2026-11-24: 225 Succeeded
    # leftovers triggered per-cycle log fetches that were discarded).
    "Succeeded",
}

# Only fetch container logs when the last (re)start is newer than this
# window (minutes). Stale restarts (restartCount > 0 from days ago) are
# already captured by restart_rate; re-fetching logs for them every cycle
# was the dominant reconcile cost (343 candidates, reconcile 75s > 60s).
LOG_FETCH_WINDOW_MINUTES = int(os.environ.get("LOG_FETCH_WINDOW_MINUTES", "120"))

# ─── Markov States ───────────────────────────────────────────────────────────
MARKOV_STATES = ["HEALTHY", "DEGRADED", "STRESSED", "CRITICAL", "FAILED", "RECOVERED"]

# ─── Remediation ────────────────────────────────────────────────────────────
REMEDIATION_ENABLED = os.environ.get("REMEDIATION_ENABLED", "false")
REMEDIATION_DRY_RUN = os.environ.get("REMEDIATION_DRY_RUN", "true")
REMEDIATION_MAX_PER_MIN = os.environ.get("REMEDIATION_MAX_PER_MIN", "5")
REMEDIATION_MAX_PER_HOUR = os.environ.get("REMEDIATION_MAX_PER_HOUR", "50")
REMEDIATION_COOLDOWN_S = os.environ.get("REMEDIATION_COOLDOWN_S", "300")
# Idempotent remediation (P3): dedup window in seconds. When an action was
# executed for a namespace/target within this window, a re-run with the same
# idempotency key is skipped (logged as '[op] dedup: action already executed')
# so a crash mid-remediation followed by a reconcile loop cannot double-scale,
# double-restart, or double-tune. Default 300s.
REMEDIATION_DEDUP_WINDOW = int(os.environ.get("REMEDIATION_DEDUP_WINDOW", "300"))
REMEDIATION_RISK_THRESHOLD = float(os.environ.get("REMEDIATION_RISK_THRESHOLD", "0.7"))

# ─── Risk alert lifecycle (auto-resolve damping) ──────────────────────────
# The auto-resolve lifecycle closes a firing "plan" alert when a pod's score
# drops. Pods hovering at the threshold flap fire/resolve every reconcile
# cycle, flooding the Matrix room. Resolve is damped by:
#   RESOLVE_HYSTERESIS: require score < threshold - hysteresis (dead band).
#   RESOLVE_MIN_FIRING_S: require the incident to have been firing this long
#                        (ignore single-cycle blips).
#   RESOLVE_COOLDOWN_S: after a resolve, suppress re-firing notifications for
#                       this window (a genuine re-escalation after cooldown
#                       still alerts).
RESOLVE_HYSTERESIS = float(os.environ.get("RESOLVE_HYSTERESIS", "0.05"))
RESOLVE_MIN_FIRING_S = int(os.environ.get("RESOLVE_MIN_FIRING_S", "90"))
RESOLVE_COOLDOWN_S = int(os.environ.get("RESOLVE_COOLDOWN_S", "300"))
# Sustained-recovery dwell: a pod must sit clearly below the threshold for
# RESOLVE_DWELL_CYCLES consecutive reconcile cycles before the incident is
# closed. Oscillating pods around the boundary therefore do not emit a
# 'resolved' every dip.
RESOLVE_DWELL_CYCLES = int(os.environ.get("RESOLVE_DWELL_CYCLES", "3"))
REMEDIATION_PROTECTED_NS = os.environ.get(
    "REMEDIATION_PROTECTED_NS", "kube-system,predictive-agent"
)

# ── P6: bounded remediation autonomy ────────────────────────────────────
# Risk tier at or above which an action requires human approval before it
# executes ("none" disables the gate; "safe" gates everything).
REMEDIATION_REVIEW_TIER = os.environ.get("REMEDIATION_REVIEW_TIER", "none").lower()
# Per-action execution checkpoint: an action that does not finish within
# REMEDIATION_TIMEOUT_S is aborted (stop waiting) and logged as failed.
REMEDIATION_TIMEOUT_S = float(os.environ.get("REMEDIATION_TIMEOUT_S", "120"))
# How long an approval stays valid before the same action needs review again.
REMEDIATION_REVIEW_TTL_S = float(os.environ.get("REMEDIATION_REVIEW_TTL_S", "900"))

# Risk tiers ordered from least to most invasive (P6).
RISK_TIERS = {"safe": 0, "moderate": 1, "destructive": 2}

# Crash-loop handling: when the planner repeatedly concludes that no action is
# beneficial for an at-risk pod (e.g. a deterministic crash-loop where a restart
# yields no durable P(FAILED) reduction), the agent must NOT go silent. After
# CRASHLOOP_ESCALATE_AFTER consecutive at-risk reconcile cycles with no
# beneficial action it escalates (alert + metric). Set CRASHLOOP_POLICY=
# persist_restart to instead keep restarting on every cooldown (useful for
# demos / operators who prefer visible churn over silence).
CRASHLOOP_POLICY = os.environ.get("CRASHLOOP_POLICY", "escalate").lower()
CRASHLOOP_ESCALATE_AFTER = int(os.environ.get("CRASHLOOP_ESCALATE_AFTER", "3"))

# ─── POMDP decision planning ────────────────────────────────────────────────#
# When enabled, the reconcile loop uses the POMDP planner (plan_and_execute) as
# the default remediation decision path instead of the per-action reactive
# evaluate(). Both paths still respect SafetyPolicy, dry_run, and (now) each
# action's own should_execute() guard.
PLANNING_ENABLED = os.environ.get("PLANNING_ENABLED", "true").lower() in ("true", "1", "yes")
PLANNING_HORIZON = int(os.environ.get("PLANNING_HORIZON", "3"))

# ─── Decision-layer safety guardrails & calibration ────────────────────────#
# Circuit breaker: cap the number of remediation actions executed in a single
# reconcile cycle (blast-radius control). 0 means uncapped.
MAX_ACTIONS_PER_RECONCILE = int(os.environ.get("MAX_ACTIONS_PER_RECONCILE", "10"))
# Dec-POMDP / multi-agent coordination (#9): max simultaneous remediation
# actions permitted on a single node within one coordination wave.
MAX_ACTIONS_PER_NODE = int(os.environ.get("MAX_ACTIONS_PER_NODE", "2"))

# ─── Verify-after-act & reversible remediation (closed loop) ────────────────#
# Post-action verification: when enabled, every executed action is verified
# against the signal snapshot recorded at execution time after a settling
# window. Requires ``VERIFY_MIN_SIGNALS`` independent signals to show sustained
# improvement; otherwise the remediation HOLDs and escalates (it never re-fires
# the same action immediately).
VERIFY_ENABLED = os.environ.get("VERIFY_ENABLED", "true").lower() in ("true", "1", "yes")
# Seconds between an action's execution and its first verification re-sample.
VERIFY_SETTLING_S = float(os.environ.get("VERIFY_SETTLING_S", "90"))
# Minimum number of independent signals that must show sustained improvement
# for a remediation to count as verified.
VERIFY_MIN_SIGNALS = int(os.environ.get("VERIFY_MIN_SIGNALS", "2"))
# Hold (seconds): while held, a failed/unverified action is not re-executed.
VERIFY_HOLD_S = float(os.environ.get("VERIFY_HOLD_S", "600"))
# Consecutive failed verifications of one key before escalation to operators.
VERIFY_ESCALATE_AFTER = int(os.environ.get("VERIFY_ESCALATE_AFTER", "2"))
# Canary: max fraction of a workload's capacity a single leg may affect
# (clamped per action; 0 = canary disabled). Conservative 15% per guidance.
CANARY_MAX_FRACTION = float(os.environ.get("CANARY_MAX_FRACTION", "0.15"))
# Auto-revert on failed verification (only for actions declaring reversible=True).
REMEDITATION_REVERT_ENABLED = os.environ.get(
    "REMEDITATION_REVERT_ENABLED", "true"
).lower() in ("true", "1", "yes")
# Confidence gate: never let the planner act when the predictor's confidence is
# below this threshold (0.0 = gate disabled).
MIN_ACTION_CONFIDENCE = float(os.environ.get("MIN_ACTION_CONFIDENCE", "0.0"))
# Calibration monitor rolling window + drift thresholds.
CALIBRATION_WINDOW = int(os.environ.get("CALIBRATION_WINDOW", "100"))
CALIBRATION_BRIER_ALERT = float(os.environ.get("CALIBRATION_BRIER_ALERT", "0.25"))
CALIBRATION_OVER_RATE_ALERT = float(os.environ.get("CALIBRATION_OVER_RATE_ALERT", "0.5"))
CALIBRATION_MIN_SAMPLES = int(os.environ.get("CALIBRATION_MIN_SAMPLES", "20"))
# When True, a detected calibration drift flips the planner into a safe reactive
# fallback (and raises an alert) until calibration recovers.
AUTO_ROLLBACK_ON_DRIFT = os.environ.get("AUTO_ROLLBACK_ON_DRIFT", "false").lower() in (
    "true", "1", "yes"
)

# ─── Continual learning: forgetting & drift-aware reweighting (#10) ────────#
# Minimum observed transitions from a state before the planner is trusted to
# act from it. Below this, plan_and_execute falls back to conservative reactive
# remediation. 0 disables the gate (always trust the planner).
MIN_MODEL_SUPPORT = float(os.environ.get("MIN_MODEL_SUPPORT", "0.0"))
# Periodic exponential forgetting: decay learned counts every N reconcile
# cycles so stale transitions fade. 0 disables periodic decay.
MODEL_DECAY_INTERVAL = int(os.environ.get("MODEL_DECAY_INTERVAL", "0"))
# Forgetting factor per decay step (1.0 = no forgetting).
MODEL_DECAY_FACTOR = float(os.environ.get("MODEL_DECAY_FACTOR", "0.99"))
# Steeper forgetting applied when calibration drift is detected, to unlearn
# potentially stale/misleading transitions.
DRIFT_DECAY_FACTOR = float(os.environ.get("DRIFT_DECAY_FACTOR", "0.5"))

# ─── Cost-model calibration from incident / SLO data ────────────────────────#
# Used by predictive_agent.cost_model to derive the decision-theoretic costs
# from real operations data (option #5): a pod failure costs
# SLO_COST_PER_MINUTE * INCIDENT_TYPICAL_OUTAGE_MIN, and performing action `a`
# costs SLO_COST_PER_MINUTE * mean(downtime_minutes for a) *
# ACTION_DISRUPTION_FRACTION (a remediation disruption is typically far cheaper
# than a full outage).
SLO_COST_PER_MINUTE = float(os.environ.get("SLO_COST_PER_MINUTE", "1.0"))
INCIDENT_TYPICAL_OUTAGE_MIN = float(os.environ.get("INCIDENT_TYPICAL_OUTAGE_MIN", "30.0"))
ACTION_DISRUPTION_FRACTION = float(os.environ.get("ACTION_DISRUPTION_FRACTION", "0.1"))

# ─── Open Policy Agent (OPA) Integration ────────────────────────────────
# OPA integration provides policy-based gating for pod processing and LLM
# usage. It is OFF by default (fail-open) so deployments without an OPA
# sidecar are unaffected; enable by setting OPA_ENABLED=true and running the
# OPA sidecar from k8s/opa/opa-sidecar.yaml. The OPA client is stdlib-only.
OPA_ENABLED = os.environ.get("OPA_ENABLED", "false").lower() in ("true", "1", "yes")
OPA_HOST = os.environ.get("OPA_HOST", "localhost")
OPA_PORT = int(os.environ.get("OPA_PORT", "8181"))
OPA_USE_TLS = os.environ.get("OPA_USE_TLS", "false").lower() in ("true", "1", "yes")
OPA_AUTH_TOKEN = os.environ.get("OPA_AUTH_TOKEN", "")
OPA_TIMEOUT = float(os.environ.get("OPA_TIMEOUT", "5.0"))
OPA_MAX_RETRIES = int(os.environ.get("OPA_MAX_RETRIES", "3"))

# ─── Alerting / Notification ────────────────────────────────────────────────
# Defaults route issue-alerts via the Uni Marburg HRZ relay (port 25, no
# auth — 587 submission requires SMTP creds the operator does not have).
# From must be a resolvable uni address: HRZ 25 rejects other domains with
# 550 5.1.8 "Sender address rejected: Domain not found".
ALERT_EMAIL_TO = os.environ.get("ALERT_EMAIL_TO", "alerts@example.com")
ALERT_EMAIL_FROM = os.environ.get("ALERT_EMAIL_FROM", "alerts@example.com")
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.example.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "25"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_USE_TLS = os.environ.get("SMTP_USE_TLS", "true").lower() in ("true", "1", "yes")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")
WEBHOOK_TIMEOUT = int(os.environ.get("WEBHOOK_TIMEOUT", "10"))
# Default alert channel: "matrix" (webhook in Alertmanager v4 schema),
# or "email", "both", "none".
NOTIFY_CHANNEL = os.environ.get("NOTIFY_CHANNEL", "matrix").lower()

# ─── FixImage — GitOps-aware image-drift remediation ────────────────────
# Bounded-autonomy image fixes for "restart cannot help" crash loops (version/
# image drift, e.g. alembic migrations advanced shared state beyond the pinned
# image's migration chain). The agent NEVER invents an image: it only applies
# the operator-declared mappings below, and by default only when the pod's
# recent logs carry a migration/image-drift signature.
#
# FIX_IMAGE_TARGETS is a JSON list of maps:
#   [{"namespace": "llm", "deployment": "open-webui",
#     "container": "open-webui",
#     "image": "registry.local/openedusuite-nix/open-webui:v0.13.3-nix",
#     "mode": "patch"}]
# The deployed configuration uses mode="patch" -> kubectl set image (no
# GitOps writes; the HRZ-GitLab gitops-repo wiring was removed 2026-11-18).
# Targets may be pinned BY DIGEST (e.g. ...@sha256:4aa9306b…) so fixes
# converge to the deterministic in-cluster nix-builder artifact.
# mode: "patch"  -> kubectl set image (may be reverted by selfHeal ArgoCD)
#       "gitops" -> optional/legacy: edit the manifest in a local ArgoCD
#                    source checkout and git commit (+ push when
#                    FIX_IMAGE_GITOPS_PUSH=true); not configured in the
#                    deployed environment.
FIX_IMAGE_ENABLED = os.environ.get("FIX_IMAGE_ENABLED", "false").lower() in (
    "true", "1", "yes"
)
FIX_IMAGE_TARGETS = json.loads(os.environ.get("FIX_IMAGE_TARGETS", "[]"))
FIX_IMAGE_REQUIRE_LOG_SIGNATURE = os.environ.get(
    "FIX_IMAGE_REQUIRE_LOG_SIGNATURE", "true"
).lower() in ("true", "1", "yes")
FIX_IMAGE_LOG_PATTERNS = json.loads(
    os.environ.get(
        "FIX_IMAGE_LOG_PATTERNS",
        '["alembic", "can\'t locate revision", "migration failed", '
        '"running upgrade", "django.db.migrations", "prisma migrate", '
        '"knex", "sequelize", "digest mismatch", "manifest unknown", '
        '"no matching manifest", "image pull"]',
    )
)
FIX_IMAGE_LOG_LINES = int(os.environ.get("FIX_IMAGE_LOG_LINES", "60"))
FIX_IMAGE_RISK_THRESHOLD = float(os.environ.get("FIX_IMAGE_RISK_THRESHOLD", "0.7"))
# Local checkout of the ArgoCD/GitOps source repo for mode="gitops"
# (optional/legacy — the deployed environment uses mode="patch" only).
FIX_IMAGE_GITOPS_REPO_DIR = os.environ.get("FIX_IMAGE_GITOPS_REPO_DIR", "")
FIX_IMAGE_GITOPS_MANIFEST = os.environ.get(
    "FIX_IMAGE_GITOPS_MANIFEST", "k8s/llm/open-webui.yaml"
)
FIX_IMAGE_GITOPS_BRANCH = os.environ.get("FIX_IMAGE_GITOPS_BRANCH", "main")
FIX_IMAGE_GITOPS_PUSH = os.environ.get(
    "FIX_IMAGE_GITOPS_PUSH", "false"
).lower() in ("true", "1", "yes")
FIX_IMAGE_KUBECTL_TIMEOUT_S = float(os.environ.get("FIX_IMAGE_KUBECTL_TIMEOUT_S", "15"))

# ─── NodeGC — node disk-pressure remediation ─────────────────────────────
# Bounded-autonomy handling of node DiskPressure (evictions, tainted nodes,
# nodeSelector-pinned workloads that cannot reschedule).
#   method "kubectl": cordon the pressuring node + emit the exact prune
#                     instruction (agent has no host access in-cluster).
#   method "docker": execute `docker system prune` on the node — only valid
#                     when the agent runs on the node with the docker socket
#                     (standalone docker runtime); degrades to kubectl path.
#   method "none":   watch-only (never acts).
NODE_GC_ENABLED = os.environ.get("NODE_GC_ENABLED", "false").lower() in (
    "true", "1", "yes"
)
NODE_GC_METHOD = os.environ.get("NODE_GC_METHOD", "kubectl").lower()
NODE_GC_NODES = os.environ.get("NODE_GC_NODES", "")  # comma list; empty = all
NODE_GC_DOCKER_PRUNE_ARGS = os.environ.get(
    "NODE_GC_DOCKER_PRUNE_ARGS", "-af"
)  # never include --volumes
NODE_GC_TIMEOUT_S = float(os.environ.get("NODE_GC_TIMEOUT_S", "180"))
NODE_GC_RISK_THRESHOLD = float(os.environ.get("NODE_GC_RISK_THRESHOLD", "0.7"))

# ─── AlertWatch — cluster-alert / storage-health correlation ─────────────
# Optional watch layer: Alertmanager active alerts boost per-pod risk scores
# for namespaces with a firing alert; ceph_health_status is surfaced and a
# WARN/ERR transition produces an alerting event (drained to the notifier).
ALERT_FEED_ENABLED = os.environ.get("ALERT_FEED_ENABLED", "false").lower() in (
    "true", "1", "yes"
)
ALERTMANAGER_URL = os.environ.get("ALERTMANAGER_URL", "")  # http://alertmanager.monitoring:9093
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "")  # http://prometheus.monitoring:9090
PROMETHEUS_BASIC_AUTH_USER = os.environ.get("PROMETHEUS_BASIC_AUTH_USER", "")
PROMETHEUS_BASIC_AUTH_PASS = os.environ.get("PROMETHEUS_BASIC_AUTH_PASS", "")
ALERT_FEED_TIMEOUT_S = float(os.environ.get("ALERT_FEED_TIMEOUT_S", "10"))
ALERT_FEED_RISK_BOOST = float(os.environ.get("ALERT_FEED_RISK_BOOST", "0.15"))
CEPH_WATCH_ENABLED = os.environ.get("CEPH_WATCH_ENABLED", "false").lower() in (
    "true", "1", "yes"
)
