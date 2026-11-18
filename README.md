# Predictive Agent

**Predictive health monitor for Kubernetes and Docker — detects failure trends,
predicts crashes, and remediates before they happen.**

It runs a 60-second reconcile loop: it collects pod/container metrics,
estimates resource trends with a Kalman filter, models pod-state transitions
with a Markov chain, scores failure risk from six signals, and — when risk
crosses the threshold — plans and executes remediation through a POMDP-style
decision layer backed by optional LLM analysis. It works **without any LLM** in
pure predictive/planner mode.

- **Dual runtime** — monitors Kubernetes pods (kubectl) *or* Docker containers
  (socket), auto-detected
- **Predictive core** — Kalman(2D) trend estimation, Markov(6) state model,
  Bayesian risk scoring, time-to-failure
- **POMDP decision layer** — cost-aware planning (action cost vs. failure
  risk), what-if simulator, calibration guardrails, monotonic safety gates
- **Closed-loop remediation** — verify-after-act (hold / auto-revert / canary),
  cooldowns, dry-run mode, full audit trail
- **Self-health alerting** — emails its own failure modes (persistence errors,
  degraded-collector streaks), rate-limited per kind
- **Pluggable LLM analysis** (async, batched, cached) — Ollama, Saia, OpenAI;
  predictions carry uncertainty + time-to-failure
- **OPA policy gate (opt-in)** — Rego policy checks gate remediation
- **Bounded-autonomy remediations** — FixImage (GitOps-aware image/migration-drift
  fix via declared target mappings + optional log-signature gate) and NodeGC
  (node DiskPressure: cordon+instruct, or host prune where the agent runs on the
  node) — all opt-in, allowlisted, reversible where possible
- **AlertWatch feed (opt-in)** — correlates Alertmanager active alerts (risk
  boost per namespace) and Ceph health (`ceph_health_status`, WARN/ERR
  transition events), surfaced via `/alerts` and `/ceph-health`
- **State persistence** — models and prediction history survive restarts

## Architecture

```mermaid
flowchart LR
    subgraph AG["predictive-agent — one 60 s reconcile loop"]
        direction TB
        COL["Collector<br/>kubectl / docker · runtime detect"]
        SM["State Model<br/>Kalman(2D) · Markov(6)"]
        RS["Risk Scorer<br/>6 signals → risk · TTF"]
        PL["POMDP Planner<br/>cost-aware plan"]
        RD["RemediationManager<br/>safety gates · cooldown · dedup<br/>verify-after-act: hold / revert / canary"]
        NF["Notifier<br/>SMTP + webhook"]
        COL --> SM --> RS --> PL --> RD --> NF
    end

    KP[("Kubernetes<br/>pods")] --> COL
    DC[("Docker<br/>containers")] --> COL
    RD -->|"read signals / act"| KP
    RD -->|"read signals / act"| DC
    RS -.->|"async, risk ≥ 0.7"| LA["LLM Analyzer<br/>batched · cached"]
    LA --> LL[("LiteLLM proxy")]
    NF --> AL[("SMTP / webhook")]
    RD -.->|"verify-after-act:<br/>re-sample on later cycles"| COL
```

The loop: **collect → estimate trends → score risk → (if risky) plan with the
POMDP planner → execute with safety checks → verify-after-act → notify →
persist state.**

## Getting started

```bash
# Run the full test suite (hermetic — no cluster, no docker daemon needed)
python -m pytest tests/

# Local spin-up of the decision layer / reconciler against real docker
docker compose up                     # Docker-native mode, dry-run remediation

# Build the reproducible Nix OCI image
./scripts/build-nix-image.sh          # -> predictive-agent:v4.0.8-nix
```

Running modes at a glance:

| Mode | What it monitors | How |
|------|------------------|-----|
| **Kubernetes** | pods via kubectl | `kubectl apply -k k8s/` (kustomize; namespace `openedusuite`) |
| **Docker-native** | containers via socket | `docker compose up` or `docker run` with the socket mounted |
| **Plain package** | same loop, no container | `python -m predictive_agent.main` |

Full build + run instructions (incl. the Docker-native mode, reproducible
chaos/remediation demo, and every image env var) are in
[docs/DEPLOY_NIX.md](docs/DEPLOY_NIX.md).

## Documentation

| Doc | What's in it |
|-----|--------------|
| **Architecture** — [docs/arc42.md](docs/arc42.md) | arc42 template, 12 sections: runtime view, decision layer, deployment view, risks |
| **Operator runbook** — [docs/OPERATOR_RUNBOOK.md](docs/OPERATOR_RUNBOOK.md) | endpoints quick reference, key env knobs, operational scenarios, alert response |
| **Deploy (Nix image)** — [docs/DEPLOY_NIX.md](docs/DEPLOY_NIX.md) | image build, Docker-native + K8s run, env var reference |
| **ADRs** — [docs/adr](docs/adr) | architecture decision records |

## Configuration

100% environment-driven (`predictive_agent/config.py`). Highlights:

```env
# Runtime (auto-detected if unset)
OPERATOR_RUNTIME=kubernetes        # docker | kubernetes
OPERATOR_WATCH_NAMESPACES=default  # k8s namespaces, or compose-project names in docker mode

# LLM (optional; `none` gives pure predictive/planner mode)
LLM_BACKEND=none                   # ollama | saia | openai | none
OLLAMA_URL=http://ollama.llm.svc.cluster.local:11434

# Remediation safety
REMEDIATION_DRY_RUN=true           # SAFE default — set false to actually act
REMEDIATION_RISK_THRESHOLD=0.7
RECONCILE_INTERVAL=60

# Alerting (self-health + remediation notifications)
ALERT_EMAIL_TO=alerts@example.com
SMTP_HOST=smtp.example.com
SMTP_PORT=25
SMTP_USE_TLS=true
```

The full knob list (behaviour, safety, verify-after-act, calibration, cost
model) is in the **Operator runbook §3** and `predictive_agent/config.py`.

New in this release (`FIX_IMAGE_*`, `NODE_GC_*`, `ALERT_FEED_*` / `CEPH_WATCH_*`):

```env
# FixImage — image/migration-drift crash loops (opt-in, pre-declared targets)
FIX_IMAGE_ENABLED=false
FIX_IMAGE_TARGETS='[{"namespace":"llm","deployment":"open-webui","container":"open-webui","image":"registry.example.com:5001/.../app:v2","mode":"patch"}]'
FIX_IMAGE_REQUIRE_LOG_SIGNATURE=true      # act only on a migration/drift log signature
# mode "gitops" additionally needs a local GitOps checkout:
# FIX_IMAGE_GITOPS_REPO_DIR, FIX_IMAGE_GITOPS_MANIFEST, FIX_IMAGE_GITOPS_PUSH

# NodeGC — node disk pressure (opt-in)
NODE_GC_ENABLED=false
NODE_GC_METHOD=kubectl                    # kubectl (cordon+instruct) | docker | none
NODE_GC_NODES=                            # comma allowlist; empty = all nodes
# docker method: NODE_GC_DOCKER_PRUNE_ARGS=-af  (never --volumes)

# AlertWatch — Alertmanager/Prometheus correlation (opt-in)
ALERT_FEED_ENABLED=false
ALERTMANAGER_URL=
PROMETHEUS_URL=
CEPH_WATCH_ENABLED=false
# PROMETHEUS_BASIC_AUTH_USER / PROMETHEUS_BASIC_AUTH_PASS for v4 auth'd scrapes
```

All of these default to **off / empty allowlist** — nothing is invented by the
agent; it only applies operator-declared mappings and falls back to cordon +
precise operator instruction when it cannot safely act.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check predictive_agent/ tests/ scripts/   # lint
mypy predictive_agent/                          # types
bandit -r predictive_agent -lll                 # SAST (HIGH only)
python -m pytest tests/                         # tests (coverage-gated ≥90%)
```

CI (`.github/workflows/ci.yml`) runs ruff, mypy, bandit, pip-audit, the full
coverage-gated test suite on Python 3.11 + 3.12, and a nightly mutation
campaign.

## License

MIT — see [LICENSE](LICENSE).
