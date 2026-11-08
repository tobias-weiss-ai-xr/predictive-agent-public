## Design: Tier-2 quality gates

### Approach

Three independent, non-blocking-for-now CI additions + one test file,
all driven by config the project already uses (env var configuration).

### 1. bandit

- Install: `pip install bandit` (dev dependency; stdlib-only package is
  unaffected at runtime — bandit is a lint-only tool).
- Command: `bandit -r predictive_agent -lll` (HIGH/CRITICAL severity only;
  exits 0 on current main — 0 HIGH findings).
- Pre-existing MEDIUM findings (`B310` urlopen in collector/server metrics
  scraping, `B104` bind to `0.0.0.0` HTTP server) exist by design; they are
  deferred tech-debt (scheme allow-list for urlopen, bind HTTP server to
  localhost) addressed in a follow-up.
- The reconciler's `except Exception: # noqa: BLE001` blocks are
    `B110 try_except_pass`-style — bandit does NOT flag broad `except
    Exception` by default (that's ruff BLE001), so no new findings expected.
- Make it **non-blocking first**: `continue-on-error: true` until the first
  green run, then remove the flag.

### 2. pip-audit

- Install: `pip install pip-audit` (dev only).
- Command: `pip-audit -r requirements.txt` if `requirements.txt` exists,
  else `pip-audit -p .` (build a wheel from pyproject and audit).
- Since `predictive_agent` is stdlib-only, the runtime dep set is small
  (pytest/ruff/mypy for dev). pip-audit will catch transitive vulns in
  dev tooling if any.

### 3. Coverage ratchet

- `pyproject.toml`: `fail_under = 90`.
- Exclude the known-low files from the coverage denominator via a
  `[tool.coverage.run]` `omit` list (NOT `exclude_lines`, which is for
  partial lines):
  ```toml
  omit = [
      "predictive_agent/demo_optimizations.py",
      "predictive_agent/__main__.py",
  ]
  ```
  `tests/` is already excluded from the cov target by CI's `--cov`.
- The `__main__.py` smoke test (item 4) covers the entrypoint; once it's
  above 50% we can drop it from `omit` and let the ratchet protect it.

### 4. `__main__.py` smoke test

- `tests/test_cli_entrypoint.py`: run
  `python -m predictive_agent --help` via `subprocess` (or
  `runpy.run_module("predictive_agent", run_name="__main__")`) and assert
  it raises `SystemExit(0)` or returns without ImportError. Keeps it at a
  real invocation rather than just import.

### Component boundaries (no production changes)

- `pyproject.toml` — `fail_under`, `[tool.coverage.run] omit`.
- `.github/workflows/ci.yml` — two new steps in the existing Python-matrix
  job (bandit, pip-audit), both report-only first.
- `tests/test_cli_entrypoint.py` — new test file.
- No changes to `predictive_agent/` package source.
