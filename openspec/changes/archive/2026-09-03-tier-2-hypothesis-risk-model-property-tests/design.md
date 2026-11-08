## Design: Hypothesis property tests

### Strategy
A `pod_metrics_strategy` composite strategy bounded to valid domains:
- `memory_pct`: `floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False)`
- `memory_mib`: `integers(min_value=0, max_value=65536)`
- `memory_limit_mib`: `integers(min_value=1, max_value=262144)` (≥1 to avoid
  div-by-zero in `memory_pct = mem / limit * 100` — mirrors the guard in
  `PodTracker.update`)
- `cpu_m`: `integers(min_value=0, max_value=8000)`
- `restart_count`: `integers(min_value=0, max_value=1000)`
- `log_errors`: `integers(min_value=0, max_value=10000)`
- `creationTimestamp`: sampled from valid ISO-8601 strings + a couple of
  malformed ones (to exercise the `ValueError → pod_age_s = 3600` fallback).

### Properties (tests/test_risk_model_properties.py)
1. `test_predict_risk_score_in_unit_interval` — `predict(...)` risk ∈ [0,1], no raise.
2. `test_classify_state_always_canonical` — `classify_state(...)` in the 5-state set.
3. `test_state_model_update_state_remains_canonical` — `update_pod` keeps
   tracker.state in the Markov set.
4. `test_predict_restart_rate_bounded` — restart_rate_per_hr =
   restart_count / max(age_seconds/3600, 1.0) is finite & non-negative
   (catches the divide-by-zone and NaN-propagation edge cases).

### Budget & CI
- Decorator `@settings(max_examples=100, deadline=None)` — deterministic
  via `--hypothesis-seed=0`; `.hypothesis/` DB committed for shrinkage
  replay.
- Placed in the `quality` CI job (report-only first run), not the fast
  test job, to keep the main matrix snappy.

### Boundaries
- Does NOT touch docker/kubectl/LLM paths (non-deterministic I/O).
- Uses the SAME `StateModel`/`Predictor`/`classify_state` APIs the edge
  tests already use — zero production changes.
