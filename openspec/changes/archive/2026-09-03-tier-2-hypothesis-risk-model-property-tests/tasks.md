## Tasks

### 1. Setup
- [x] 1.1 Add `hypothesis` to `[dev]` extras in `pyproject.toml`
- [x] 1.2 Confirm `from hypothesis import given, strategies as st, settings` available

### 2. Property tests
- [x] 2.1 Create `tests/test_risk_model_properties.py` with
  `pod_metrics_strategy` composite
- [x] 2.2 `test_predict_risk_score_in_unit_interval` (no raise, risk ∈ [0,1])
- [x] 2.3 `test_classify_state_always_canonical` (state ∈ 5-state set)
- [x] 2.4 `test_state_model_update_state_remains_canonical`
- [x] 2.5 `test_predict_restart_rate_bounded` (finite, non-negative)
- [x] 2.6 Determinism: `derandomize=True` profile (stronger than a pinned
  seed — no DB dependence). FINDING: ttf_minutes==0 is a valid contract
  outcome (remaining headroom <= 0 ⇒ imminent failure, predictor.py:135).

### 3. CI & completion
- [x] 3.1 Add Hypothesis step to `.github/workflows/ci.yml` `quality` job (report-only)
- [x] 3.2 Full suite green (1393 + new); ruff + mypy clean
- [x] 3.3 `openspec validate` passes; commit + push + sync edge-node (117 green on-node)
