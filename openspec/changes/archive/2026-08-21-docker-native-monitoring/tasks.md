## 1. Discovery config

- [ ] 1.1 Add Docker watch-selector env vars (`OPERATOR_WATCH_LABELS`, `OPERATOR_WATCH_COMPOSE_PROJECTS`, `OPERATOR_WATCH_NAMES`) and `OPERATOR_RUNTIME` auto-detect (Docker socket vs kubeconfig) in the config loader (`predictive_agent/collector.py` / `predictive_agent/main.py`).
- [ ] 1.2 Add unit tests for selector parsing: label match, compose-project match, name pattern, and default (no selector ⇒ all).

## 2. Container discovery

- [ ] 2.1 Implement `discover_docker_containers()` in `predictive_agent/collector.py` using `docker ps` / `docker inspect` (subprocess, same pattern as the kubectl path) filtered by the configured selectors; return a stable container list keyed by id.
- [ ] 2.2 Add tests for discovery with mocked `docker` output: by label, by compose project, no selector ⇒ all, and lifecycle removal when a container disappears.

## 3. Container status signals

- [ ] 3.1 Generalize `get_pod_status_signals()` into a container-status extractor (`get_container_status_signals`) in `predictive_agent/collector.py` emitting the same fields the risk model consumes (wait_state, terminated, restart_count, ready).
- [ ] 3.2 Extend `tests/test_pod_status.py` to cover container scenarios (waiting / terminated / restart-loop / ready) and rename it to reflect container-status scope.

## 4. Reconcile wiring

- [ ] 4.1 Route discovered Docker containers through the existing Kalman + Markov + Bayesian risk pipeline in `predictive_agent/main.py`; the state model and risk scorer are unchanged.
- [ ] 4.2 Finalize/clean a container's state-model entry when it stops or is removed; add a test for lifecycle finalization.

## 5. Verification & deploy

- [ ] 5.1 Run `pytest` and confirm all new and existing tests pass.
- [ ] 5.2 Update the `monitoring_agents` Ansible role compose env (`OPERATOR_WATCH_*`) for the monitoring hosts and `predictive-agent.nix` if the `docker` CLI must be added to the image; document defaults in `README.md`.
