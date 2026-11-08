## ADDED Requirements

### Requirement: Reconcile pipeline facade stability
The main module SHALL preserve its public facade while its internal
structure changes: the `reconcile()` entry point, the module-level
component and statistics attributes (`_state_model`, `_predictor`,
`_state_store`, `_remediation_manager`, `_notifier`, `_backtester`,
`_policy_engine`, `_llm_analyzer`, `_kg_client`, `_reconcile_count`,
`_last_reconcile_time`, `_last_reconcile_duration`), and the collection
hooks (`_get_pods_json`, `_collect_docker_container_metrics`) SHALL
remain importable and patchable at the module path
`predictive_agent.main`. The HTTP server and the test suite rely on
resolving these names on the main module at call time, not on copies.

#### Scenario: Server keeps reading live state through the facade
- **WHEN** the HTTP server reads component references or reconcile
  statistics from the main module
- **THEN** it observes the values updated by the most recent reconcile
  cycle, without importing any new internal module

#### Scenario: Collection hooks stay overridable
- **WHEN** a test patches a collection hook on the main module
- **THEN** the next reconcile cycle resolves and uses the patched hook,
  and no unpatched duplicate implementation is invoked instead

#### Scenario: Reconcile summary contract unchanged
- **WHEN** one reconcile cycle completes
- **THEN** it returns the same summary keys as before the
  decomposition (predictions, state, timestamp, pods_tracked,
  at_risk_count, reconcile_count, and the remediation/backtest
  statistics blocks)
