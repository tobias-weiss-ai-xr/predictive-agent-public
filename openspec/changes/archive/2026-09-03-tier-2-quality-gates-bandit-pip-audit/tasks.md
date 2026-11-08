## Tasks

### 1. Coverage ratchet (config)
- [x] 1.1 Set `fail_under = 90` in `[tool.coverage]` in `pyproject.toml`
- [x] 1.2 Add `[tool.coverage.run] omit = [...]` for `demo_optimizations.py` and `__main__.py`
- [x] 1.3 Verify local `pytest --cov` still passes ≥90% before committing

### 2. bandit gate (CI)
- [x] 2.1 Add `pip install bandit` to CI dev-tooling matrix
- [x] 2.2 Add `bandit -r predictive_agent -ll -iii -ll` step to `.github/workflows/ci.yml` (continue-on-error first run)
- [x] 2.3 Resolve or justify any findings; flip step to required on green

### 3. pip-audit gate (CI)
- [x] 3.1 Add `pip install pip-audit`
- [x] 3.2 Add `pip-audit -r requirements.txt` (fallback `pip-audit -p .`) step
- [ ] 3.3 Make non-blocking; verify clean on main before requiring

### 4. CLI entrypoint test
- [x] 4.1 Create `tests/test_cli_entrypoint.py` covering `python -m predictive_agent --help`
- [x] 4.2 Verify it passes; remove `__main__.py` from coverage omit once >50%

### 5. Completion
- [x] 5.1 Full suite green at `fail_under = 90`; ruff + mypy clean
- [x] 5.2 `openspec validate` passes for the change
- [x] 5.3 Commit, push, sync edge-node (`192.0.2.42`)
