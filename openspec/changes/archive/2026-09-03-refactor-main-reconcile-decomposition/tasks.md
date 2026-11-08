## Tasks

### 1. Extraction scaffolding (no behavior change)

- [x] 1.1 Create `predictive_agent/pod_collection.py`; move `_now_iso`, `_kubectl_available`, `_get_pods_json`, `_get_pod_restart_count`, `_get_pod_namespace`, `_get_pod_name`, `_get_pod_memory_limit`, `_should_skip_namespace`, `_get_node_pressure`, `_matches_docker_selectors`, `_collect_docker_container_metrics`, `_get_container_memory_limit` with identical bodies; main re-binds all names as module aliases
- [x] 1.2 Create `predictive_agent/reconcile_phases.py` with `ReconcileContext` frozen dataclass (component refs + cycle counters) and empty result dataclasses; no phase bodies yet
- [x] 1.3 Run full suite — must be 1290 passed with zero test edits (aliases verified)

### 2. Phase extraction (one commit-sized step per phase, suite green after each)

- [x] 2.1 Extract `collect_metrics` phase (runtime detection, docker/k8s collection, degraded-mode collector-state tracking, removal tracking) — reconcile() calls it via main namespace
- [x] 2.2 Extract `update_state_model` phase (pod upsert, degraded no-data handling, removed-pod finalization)
- [x] 2.3 Extract `evaluate_pods` phase (per-pod OPA gate, prediction, risk, decision planning, remediation) with per-pod accumulator intact
- [x] 2.4 Extract `apply_calibration_guardrail` phase (drift detection → circuit breaker/rollback)
- [x] 2.5 Extract `apply_continual_learning` phase (Markov update, periodic forgetting)
- [x] 2.6 Extract `persist_state` phase (state store write, `_state_saves` delta)
- [x] 2.7 Extract `record_backtest_predictions` phase
- [x] 2.8 Extract `finalize_summary` phase (remediation stats, server context sync) and shrink reconcile() to pipeline + globals ownership

### 3. Startup decomposition

- [x] 3.1 Create `predictive_agent/bootstrap.py` with `initialize_components()` returning `ComponentSet` (remediation, notifier, KG, backtester, LLM analyzer in current order); main() unpacks into existing globals
- [x] 3.2 Run full suite — zero test edits expected

### 4. New phase-level tests

- [x] 4.1 `tests/test_reconcile_phases.py`: direct phase invocations (collect_metrics degraded/error branches, update_state_model removal, evaluate_pods OPA-deny + plan-only, calibration guardrail, continual learning decay, persist, backtest, summary contract keys)
- [x] 4.2 `tests/test_main_facade.py`: facade names exist on main; patching `main._get_pods_json` / `_collect_docker_container_metrics` intercepts reconcile; summary keys unchanged
- [x] 4.3 Verify main.py coverage ≥ 85%, overall ≥ 91% (`--cov=predictive_agent --cov-report=term-missing`)

### 5. Gates and completion

- [x] 5.1 ruff clean, mypy clean (40+ source files), full suite green
- [x] 5.2 `openspec validate` passes; spec delta scenarios all demonstrable
- [x] 5.3 Commit (e2b2cb5), push origin/main; edge-node sync deferred (host-local runner)
