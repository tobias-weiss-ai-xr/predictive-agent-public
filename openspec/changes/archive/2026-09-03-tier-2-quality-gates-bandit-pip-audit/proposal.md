## Proposal: Tier-2 quality gates — bandit (SAST) + pip-audit + coverage ratchet

### Why

CI currently enforces **Tier 1 gates**: Ruff (lint), Mypy (types), and
coverage-gated tests (`fail_under = 75`). After the main.py decomposition
the project sits at **93% coverage (1391 tests)**, so the 75% floor is
dead weight — it does not protect the gains and would only trip on a
single bad PR years from now. A KRITIS-aware deployment additionally needs
dependency vulnerability scanning (pip-audit) and static taint analysis
(bandit) on the reconciler path, which contains `run_cmd` shellouts,
Kubernetes API calls, and OPA fail-open `try/except` blocks.

### What changes

1. **Ratchet the coverage gate** in `pyproject.toml` from `fail_under = 75`
   to `fail_under = 90`, with explicit `--cov-exclude` for the three
   intentionally-low-coverage files (`demo_optimizations.py`,
   `__main__.py`, `tests/` already excluded by mypy config) so the floor
   is meaningful rather than blunt.
2. **Add `bandit`** as a CI step: `bandit -r predictive_agent -ll -iii -ll`
   (medium-and-high severity, low-and-high confidence). The reconciler's
   fail-open `except Exception` blocks are already tagged `# noqa: BLE001`
   under ruff; bandit's `try_except_pass`/`B110` should remain clean.
3. **Add `pip-audit`** as a CI step: `pip-audit -r requirements.txt`
   (or `pip-audit -p .` against the pyproject build) to scan transitive
   dependencies (urllib3, pyyaml, requests, httpx, kubernetes).
4. **Add a `__main__.py` smoke test** (`tests/test_cli_entrypoint.py`) so
   the 0%-coverage entrypoint is covered (removes the last gap file).

### Out of scope

- No new production code paths; these are CI/config/test-only changes.
- Mutation testing (`mutmut`) is a separate follow-up change.
- Hypothesis property tests are a separate follow-up change.

### Risk

- **Low.** bandit/pip-audit are non-blocking report-only steps for the
  first green run; the coverage ratchet only bites if coverage drops
  below 90% (currently 93%, so safe). The `__main__` smoke test is
  ~5 lines.
- If `pip-audit` flags an existing vulnerable transitive dep, the gate
  is added as `continue-on-error` until the dep is remediated, so CI
  never blocks on a pre-existing issue.
