## Why

predictive-agent is deployed on Docker hosts (example.org, edge-node) but currently tracks **0 containers**: its `OPERATOR_WATCH_NAMESPACES` are Kubernetes namespace names, which do not exist on a plain Docker host, so the reconcile loop runs but monitors nothing. The agent is healthy yet inert in production. Docker-native monitoring makes the already-deployed agent actually fulfill its predictive purpose on the live fleet.

## What Changes

- Add Docker-aware workload discovery that finds containers by label / compose-project / name instead of requiring Kubernetes namespaces.
- Route discovered container metrics (CPU, memory, restarts, logs, status) through the existing Kalman + Markov + Bayesian risk pipeline — no rearchitecture.
- Repurpose the in-progress `get_pod_status_signals` work for container status (waiting / terminated / restart-loop detection) and extend tests.
- Surface predictions/state via the existing `/predictions`, `/state`, `/metrics` endpoints (no new API surface required).
- Update config so Docker mode uses a discoverable target set (e.g. `OPERATOR_WATCH_LABELS` / compose-project scoping) rather than k8s namespaces.
- Add/extend tests for Docker discovery + status signals; verify with `pytest`.

## Capabilities

### New Capabilities
- (none)

### Modified Capabilities
- `predictive-agent`: modifies the "Metrics collection" requirement so Docker mode actually discovers and tracks containers, and adds a "Docker workload discovery" requirement; the existing risk-prediction and state-model requirements now apply to tracked Docker containers.

## Impact

- **Code**: `predictive_agent/collector.py` (discovery + container status signals), `predictive_agent/main.py` (reconcile wiring). `server.py` unchanged (reuses endpoints); `state_model.py` / `risk.py` only if signal shaping changes.
- **Config/Deploy**: `monitoring_agents` Ansible role compose (`OPERATOR_WATCH_*`), nix image build (unchanged).
- **Tests**: extend `tests/test_pod_status.py` into container-status coverage; add Docker discovery tests.
- **Dependencies**: none new — stays stdlib-only.

## Non-goals

- Not changing the Kubernetes (`kubectl`) code path — it stays as-is.
- Not adding new HTTP endpoints or LLM backends.
- Not addressing cAdvisor / cgroup-v2 on edge-node (separate concern).
- Not adding auto-remediation / self-healing — still predictive-only.
