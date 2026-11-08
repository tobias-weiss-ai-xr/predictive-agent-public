## Context

The reconcile loop in `main.py` already drives collection → state model (Kalman + Markov) → Bayesian risk → HTTP endpoints, and that pipeline is runtime-agnostic: it operates on whatever workloads are *tracked*. The bug is purely at the discovery boundary. In Docker mode `collector.py` currently filters by `OPERATOR_WATCH_NAMESPACES` — Kubernetes namespace names that do not exist on a plain Docker host — so discovery yields zero containers and the agent is inert (observed live: "0 containers tracked" on example.org).

The in-progress `get_pod_status_signals()` in `collector.py` (covered by the new `tests/test_pod_status.py`) is the natural base to generalize into container-status signal extraction, since the risk model already consumes that signal shape (wait state, terminated, restart count, …).

## Goals / Non-Goals

**Goals:**
- Docker-native workload discovery via label / compose-project / name selectors.
- Extract container status signals and feed the existing Kalman/Markov/risk pipeline unchanged.
- Make discovery/status observable and tested (`pytest`).

**Non-Goals:**
- No change to the Kubernetes (`kubectl`) path.
- No new HTTP endpoints, LLM backends, or auto-remediation.
- No cAdvisor / cgroup-v2 work.

## Decisions

1. **Discovery mechanism — `docker` CLI via `subprocess`** (same pattern the k8s path uses for `kubectl`), not a third-party SDK.
   *Rationale*: keeps the package stdlib-only (a hard project convention); consistent with existing code. *Alternative considered*: `docker` Python SDK — rejected (adds a dependency).

2. **Config — additive `OPERATOR_WATCH_*` env vars.** Add `OPERATOR_WATCH_LABELS`, `OPERATOR_WATCH_COMPOSE_PROJECTS`, `OPERATOR_WATCH_NAMES` for Docker mode; keep `OPERATOR_WATCH_NAMESPACES` for k8s. Runtime is auto-detected (`DOCKER_SOCKET` present vs `KUBECONFIG`) or forced via `OPERATOR_RUNTIME=docker|k8s`.
   *Rationale*: no breaking change; k8s deployments keep working untouched.

3. **Signal extraction — generalize `get_pod_status_signals` → container variant** emitting the same fields the risk model already consumes, so the 6-signal Bayesian scorer needs no change.

4. **Tracking store** — keyed by container id; on disappearance, finalize/clean the state-model entry (no leak across restarts of the same service).

## Risks / Trade-offs

- **[Risk] `docker` CLI absent from the minimal nix image** → *Mitigation*: ensure `docker` is included in `predictive-agent.nix` (the k8s path already ships `kubectl`); document the mount otherwise.
- **[Risk] Discovery storms on busy hosts** → *Mitigation*: cache the container list per reconcile; only `inspect` containers whose state changed.
- **[Risk] Selector semantics differ per deployment** → *Mitigation*: document defaults; no selector ⇒ track all (safe default).
- **[Trade-off]** subprocess CLI is slower per call than an in-process socket client, but at a 60 s `RECONCILE_INTERVAL` this is negligible.

## Migration Plan

- Additive env vars; default Docker behavior (no selector) = track all running containers, so existing deployments keep working without config changes.
- Deploy via the `monitoring_agents` Ansible role (compose env); rebuild the nix image only if `docker` CLI must be added.
- Rollback = revert env / image tag. No API or spec break; endpoints unchanged.

## Open Questions

- Should `OPERATOR_WATCH_COMPOSE_PROJECTS` be the *primary* selector on the monitoring hosts (their stacks are compose projects)? Defaulting to "all" is safe; can refine later without spec change.
- Exact container-signal field names to keep the risk-model input contract stable (resolved during implementation; no spec impact).
