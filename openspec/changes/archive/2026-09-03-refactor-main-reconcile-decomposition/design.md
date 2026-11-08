## Design: Decomposing main.py's reconcile pipeline

### Constraints discovered by inspection (the WHY behind every choice)

1. **The main module IS the service locator.** `server.py` builds its
   `_ServerState` from main's module attributes and the test suite
   patches `predictive_agent.main._state_model`, `._get_pods_json`,
   `._collect_docker_container_metrics`, `._reconcile_count`,
   `reconcile` directly (verified: 20 patch sites on `_state_model`
   alone). Moving or copying these names would silently break patch
   targeting. → Every existing main-module name must survive, and
   `reconcile()` must resolve moved helpers through main's namespace
   (late binding) so `patch("predictive_agent.main.X")` still
   intercepts.
2. **Behavior is pinned by 1290 tests.** The refactor must be a
   mechanical extraction; any semantic drift surfaces as a red suite.
3. **Python 3.9 typing style** and stdlib-only apply to new modules.

### Target shape

```
main.py (orchestrator, ~350 lines)
├── globals + thin late-bound aliases          (contract, unchanged)
├── reconcile() = pipeline calling phases      (~120 lines)
├── ReconcileLoop / _setup_logging /
│   _start_http_server / main()                (unchanged, minus init body)

pod_collection.py   (new, ~220 lines)  — pure runtime-data functions
reconcile_phases.py (new, ~600 lines)  — phase functions + result dataclasses
bootstrap.py        (new, ~180 lines)  — component init, returns ComponentSet
```

### Key decisions

- **Phase functions are parameters-in / result-out**, not methods on a
  god-object. They receive a `ReconcileContext` dataclass holding the
  component references (state model, predictor, remediation manager,
  backtester, ...) plus cycle numbers. Counters (`_llm_calls`,
  `_llm_errors`, `_state_saves`) stay main-module globals; phases
  return deltas that reconcile() applies. Rationale: keeps the patch
  contract intact and phases pure enough to unit-test without
  importing main.
- **Collection helpers move wholesale** to `pod_collection.py`; main
  re-binds them (`_get_pods_json = pod_collection.get_pods_json`).
  Because reconcile() calls them via main's globals, patch sites keep
  working. Trade-off: an indirection alias instead of a plain import —
  accepted for contract stability.
- **`ReconcileContext` is a frozen dataclass** built once per cycle
  from the globals; phases never import main. This breaks the
  main↔phases import cycle (phases import config/models, main imports
  phases).
- **bootstrap.py mirrors the current init order exactly** (remediation
  → notifications → KG → backtester → LLM analyzer) and returns a
  `ComponentSet`; `main()` unpacks it into the existing globals. Init
  order is load-bearing (warnings, fallback wiring), so no reordering.
- **The per-pod loop (616–991) is extracted as one phase**
  (`evaluate_pods`) rather than being split further: its OPA gate,
  prediction, planning, and remediation steps share loop-local state
  (tracker, risk, decision plan). Splitting it would require a
  per-pod accumulator object that adds indirection without a second
  consumer. The calibration guardrail, continual learning, persistence,
  and backtest blocks become separate phases (they are sequential,
  independent, and already communicate through the summary dict).

### Test strategy

1. Existing suite must pass untouched after every extraction step
   (run after each phase, not just at the end).
2. New `tests/test_reconcile_phases.py` calls each phase function
   directly with constructed inputs — this is where main.py coverage
   rises (phases were unreachable without full reconcile setup).
3. New facade test `tests/test_main_facade.py`: asserts the contract
   names exist on main and that patching `main._get_pods_json` changes
   reconcile behavior (guards against future regressions to
   copy-style imports).
4. Coverage gate: `main.py` ≥ 85% (from 73%); overall stays ≥ 91%.
