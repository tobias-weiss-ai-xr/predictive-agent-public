## Proposal: Decompose main.py's reconcile pipeline into testable phases

### Why

`predictive_agent/main.py` is 1440 lines with a single ~870-line
`reconcile()` function covering nine responsibilities (metric collection,
state-model update, OPA gating, prediction, remediation, calibration
guardrail, continual learning, persistence, backtest recording) plus the
startup/initialization sequence for six component singletons. At 73%
coverage it is the last major untested surface; the function is only
exercised end-to-end through patched globals, and every new behavior
lands inside one untestable body.

### What changes

Extract behavior-preserving phase modules and have `main.reconcile()`
become a thin pipeline that owns the module globals and delegates:

- `predictive_agent/pod_collection.py` — runtime detection, kubectl/Docker
  metric gathering helpers (`_get_pods_json`,
  `_collect_docker_container_metrics`, `_matches_docker_selectors`, pod
  field getters).
- `predictive_agent/reconcile_phases.py` — one function per reconcile
  phase, each taking an explicit `ReconcileContext` (dataclass) plus
  phase inputs and returning a phase result object.
- `predictive_agent/bootstrap.py` — component initialization
  (remediation manager, notifier, KG client, backtester, LLM analyzer)
  returning a `ComponentSet`.

**Frozen public contract** (verified by grep): `server.py` reads main's
module globals through its `_ServerState`, and tests patch
`predictive_agent.main._state_model`, `._get_pods_json`,
`._collect_docker_container_metrics`, `.reconcile`, `_reconcile_count`,
etc. `main.py` therefore keeps every existing module-level name; moved
implementations are re-bound as late-bound aliases and `reconcile()`
continues to resolve them through main's namespace so `patch()` targets
keep working.

### Non-goals

- No behavioral changes: same logging, same summary dict keys, same
  ENV-driven config, same collector/degraded-mode protocol.
- No renaming or removal of any public or `main`-module attribute.
- No new dependencies (package stays stdlib-only).
- No changes to `collector.py`, `server.py`, or any other module's API.
- Not rewriting the pod loop into a class-based strategy (deferred).

### Affected capability specs

- `openspec/specs/predictive-agent/spec.md` (reconcile loop & observability
  requirements; structure-only deltas, no requirement changes)

### Success measure

`main.py` shrinks to ~350 lines of orchestration; suite stays green
(1290 tests); `main.py` coverage rises from 73% toward ≥85% via new
phase-level tests; ruff + mypy stay clean.
