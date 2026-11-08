## Proposal: Tier-2 mutation testing (mutmut) on phase modules

### Why

The main.py decomposition extracted 8 phase functions into
`reconcile_phases.py`, 12 collection helpers into `pod_collection.py`,
and startup logic into `bootstrap.py`. Phase coverage is now **95%/89%/84%**
respectively, but line coverage cannot detect *mutated* conditions — a
swapped `and`/`or`, a `>=` → `>`, or a deleted `not` in a fail-open gate
would pass the suite yet silently weaken safety. These extracted modules
are pure, deterministic, and small (~1100 lines combined), making them
the ideal mutation-testing surface before they accrete untested branches.

### What changes

1. Add `mutmut` to dev dependencies (`[dev]` extras in pyproject.toml).
2. Add a `[tool.mutmut]` config scoping the campaign to the three new
   modules (exclude `__main__` / `server` / `demo_optimizations`).
3. Run `mutmut run` + `mutmut apply` (or harvest survivors as new tests)
   against `reconcile_phases.py` + `pod_collection.py` first, then
   `bootstrap.py`.
4. Add the surviving-mutant regression tests to
   `tests/test_reconcile_phases.py` / `tests/test_pod_collection.py`,
   following the TaskFleet round-3 fixture-reuse pattern
   (documented in `mem_mtiroa5z_mtkfoyn7`).
5. Gate: ≥ 90% killed mutation score before flipping the job to required.

### Out of scope

- Mutating the full legacy modules (collector, state_model, predictor)
  — separate change; focused pass on the freshly-extracted surface.
- CI integration of the mutmut job (that lives in the tier-2 CI gates
  change; here we run it ad hoc on edge-node first).

### Risk

- **Timebox**: mutmut on ~1100 lines is ~50–90 min on edge-node (192.0.2.42)
  at the same scale as TaskFleet round 3. We seed the cache so re-runs
  skip covered mutants.
- **False positives**: fail-open `except: pass` / `# nosec`-style blocks may
  yield "killed by timeout" — reviewed manually, not auto-applied.
