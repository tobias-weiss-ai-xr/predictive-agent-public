## Tasks

### 1. Setup
- [x] 1.1 Add `mutmut` to `[dev]` extras in `pyproject.toml`
- [x] 1.2 Add `[tool.mutmut]` config: `source_paths` = 3 phase modules,
  full-suite selection minus env-flaky files (`test_server`,
  `test_decision_layer_integration`, `test_opa_integration_simple`,
  `test_kalman_params` — order-sensitive under mutmut's runner)
- [x] 1.3 Verify `python -m mutmut run` executes (config keys modernized
  for mutmut 3.x: `source_paths`, `pytest_add_cli_args[_test_selection]`)

### 2. Campaign (local Mac; edge-node saturated — load avg 99-165 from
### external workloads, campaign moved off it)
- [x] 2.1 Full campaign: **2521 mutants, 1211 killed (48.0%), 1310
  survived** (fresh run incl. harvest tests; first pass was 42.1%)
- [x] 2.2 Survivors triaged by sampling: ~92% SEMANTIC (argv/timeout
  contracts, dict-key contracts, gate polarity, getattr defaults),
  ~8% trivial (log-format strings)
- [x] 2.3 Harvest round 1: `tests/test_mutation_harvest.py` — 12
  family-killing tests (exact kubectl argv+timeout contracts, owner-ref
  Job/CronJob filter, log-gate polarity incl. SKIP_LOGS_STATUSES,
  inclusive risk>=0.7 warning, persist payload key set, autonomy-notify
  getattr defaults, removed-pod FAILED-casing finalize, node-pressure
  condition keys, docker stats line parse) → **+150 mutants killed**
- [x] 2.4 Re-run confirmed incremental kill (48.0% total)

### 3. Campaign — bootstrap (edge-node)
- [x] 3.1 bootstrap.py included in the same campaign (240 survivors
  there are dominated by logger-string + fallback-branch mutants)
- [x] 3.2 Covered by harvest tests (k8s/docker action registration
  paths already contract-pinned in tests/test_bootstrap.py)

### 4. Completion
- [x] 4.1 CI integration DEFERRED by design: per-push mutmut on the
  shared self-hosted runner (edge-node) would add ~15 min of heavy load per
  push; campaign stays on-demand (see spec scenario 1)
- [x] 4.2 Full suite green: 1409 passed; ruff + mypy + bandit clean
- [x] 4.3 `openspec validate` passes; commit + push + sync edge-node (117 green on-node)
