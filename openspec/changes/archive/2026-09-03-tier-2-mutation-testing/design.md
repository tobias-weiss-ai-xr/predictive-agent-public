## Design: Mutation testing campaign

### Tooling
- `mutmut` (dev dependency) — Python-native, works with the existing pytest
  backend (no extra runtime deps since the package stays stdlib-only at runtime).

### Scope & config
`[tool.mutmut]` in `pyproject.toml`:
```toml
[tool.mutmut]
paths_to_mutate = "predictive_agent/reconcile_phases.py,predictive_agent/pod_collection.py,predictive_agent/bootstrap.py"
tests_dir = "tests/"
runner = "python -m pytest"
# Pre-seed the cache so re-runs don't regenerate covered mutants.
```
Start with `reconcile_phases.py` + `pod_collection.py` (the safety-critical
gate/decay/throttle logic), then expand to `bootstrap.py`.

### Actual workflow (learned)
- Campaign ran on the LOCAL Mac: edge-node was saturated by external
  workloads (load avg 99-165; it hosts GitHub Actions runners).
- mutmut 3.x config: `source_paths` (not `paths_to_mutate`),
  `pytest_add_cli_args` / `pytest_add_cli_args_test_selection` (not
  `runner` / `tests_dir`).
- Cache: `.mutmut-cache` serves classifications even after tests change —
  `rm -rf .mutmut-cache mutants` before a post-harvest re-run.
- mutmut's own full-suite "clean tests" run trips on order/state-sensitive
  files that pass standalone (test_kalman_params) — deselect them.
- ~15 min wall per full campaign at `--max-children 4` (2521 mutants,
  5.5 mutations/s).

### Original workflow sketch (on edge-node `user@192.0.2.2`, ~75m wall)
1. `python -m mutmut run` — generates + tests mutants, writes `.mutmut-cache`.
2. `python -m mutmut results` — list survivors.
3. For each survivor:
   - Is it a real defect (e.g. `>=` → `>` in `markov_p_critical` threshold)?
     → add a test in `tests/test_reconcile_phases.py` following the
     "skim the edges file + reuse fixture" pattern (`mem_mtiroa5z_mtkfoyn7`).
   - Is it an accepted false positive (e.g. `except: pass` fail-open)?
     → justify inline with `# mutmut: allow`.
4. `python -m mutmut run` again → confirm ≥ 90% killed.
5. Add `mutmut` to CI `quality` job (report-only first).

### Boundaries
- Mutate **only** the three extracted modules; legacy modules are a
  separate change to keep the campaign timeboxed.
- Do NOT mutate `server.py` / `__main__.py` / `demo_optimizations.py`
  (already out of coverage scope).
