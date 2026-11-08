## ADDED Requirements

### Requirement: Tier-2 static analysis gate

The CI pipeline SHALL run a static analysis tool over the `predictive_agent`
package that detects common security issues (hardcoded secrets, shell
injection via `run_cmd`, use of `eval`/`exec`, weak crypto, weak TLS, and
overly-broad `except` blocks that swallow errors silently).

#### Scenario: bandit finds no high/critical issues on main
- **WHEN** CI runs `bandit -r predictive_agent -ll -iii -ll`
- **THEN** the job reports zero findings at severity `HIGH` (B3) or `CRITICAL` (B4)
  and the reconciler's deliberately-broad `except Exception: # noqa: BLE001`
  fail-open blocks are not flagged at the configured severity/confidence level
- **AND** the job exits 0

#### Scenario: a new high-severity bandit finding blocks merge
- **WHEN** a PR introduces a `HIGH` or `CRITICAL` bandit finding
- **THEN** the bandit step fails and the PR is blocked from merging
  until the finding is resolved or explicitly justified with a
  `# nosec` annotation reviewed by a maintainer

### Requirement: Dependency vulnerability scanning gate

The CI pipeline SHALL scan project dependencies (runtime transitive deps
declared in `requirements.txt` / `pyproject.toml` build) for known
vulnerabilities and report versions with no fixed upstream patch.

#### Scenario: pip-audit clean on main
- **WHEN** CI runs `pip-audit` against the resolved dependency set
- **THEN** the job reports zero known vulnerabilities (or only
  `continue-on-error`-flagged advisory items with no upstream fix)
- **AND** the job exits 0

### Requirement: Coverage ratchet (non-blunt)

The coverage gate SHALL enforce `fail_under = 90` for the project while
explicitly excluding known-low-coverage entrypoint/demo files so the
ratchet cannot be defeated by padding, nor trip on intentionally-skipped
code.

#### Scenario: coverage floor holds after a feature PR
- **WHEN** CI computes aggregate coverage with
  `pytest --cov=predictive_agent --cov-report=term-missing`
- **THEN** the job fails if coverage drops below 90%
- **AND** `demo_optimizations.py` (demo script), `__main__.py` (3-line
  entrypoint), and `tests/` are excluded from the gate denominator

### Requirement: CLI entrypoint test coverage

The `predictive_agent/__main__.py` entrypoint SHALL have a regression
test so its 0% coverage is removed and the `python -m predictive_agent`
invocation path is protected against regressions.

#### Scenario: entrypoint imports and reports usage
- **WHEN** a test runs `python -m predictive_agent --help` (via subprocess
  or `runpy.run_module`)
- **THEN** it exits 0 (or exits cleanly with a usage message) and the test
  asserts the module is importable without error
