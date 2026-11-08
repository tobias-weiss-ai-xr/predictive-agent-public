## Proposal: Tier-2 Hypothesis property tests for risk model + state machine

### Why

The fleet rounds 1–3 found **6 real bugs** via edge-case tests (rl.py
`update()` KeyError, collector KeyError escape, L1Cache key asymmetry,
`_background_processor` dropping the wake-up request, opa_client mypy).
All were boundary or invariant violations that line/edge-example tests
only catch *after* someone writes the specific input.

The newly-decomposed `reconcile_phases.py` (95%) + `state_model.py` +
`predictor.py` expose deterministic, numeric boundaries that are
*ideal* for property-based fuzzing:
- `risk_score ∈ [0, 1]` for all valid pod inputs
- `predictor.predict` never raises on valid-but-extreme metrics
- `StateModel.update_pod` Markov transitions are monotonic in restart_rate
- `classify_state` returns one of the 5 canonical states

A single Hypothesis strategy over pod metrics would have caught the
`markov_p_critical >= 0.7` WARNING-log threshold bug *and* the
restart-rate divide-by-zero at the pod-age boundary — both are exactly
the class of defect the round-3 contract facts (`markov_p_critical`,
`markov_p_failed`, `RECONCILE` states) target.

### What changes

1. Add `hypothesis` to `[dev]` extras.
2. Add `tests/test_risk_model_properties.py` with `@given`-driven strategies:
   - `pod_metrics_strategy` — floats in valid ranges, `creationTimestamp` variants
   - property: `risk_score` always in `[0, 1]`
   - property: `predict` always returns a `PredictionResult` (no exceptions)
   - property: `StateModel.update_pod` keeps `state` in the valid Markov set
3. Add a Hypothesis strategy for `classify_state` boundary fuzz around the
   70/85/95 `% memory` and restart-rate thresholds (catches off-by-one in the
   classifier that the existing `HEALTHY/DEGRADED/STRESSED/CRITICAL` matrix
   doesn't pin).
4. Gate: `hypothesis --hypothesis-seed=0` deterministic in CI; budget the
   runs (default 100 examples, `--hypothesis-seed` pinned).

### Out of scope
- Hypothesis for non-deterministic / I/O-bound code (docker logs, kubectl,
  LLM background thread) — those keep the existing edge-example style.
- Property tests for the legacy collector/server (separate change).

### Risk
- **False positives**: Hypothesis may shrink to exotic inputs that the
  production path handles "by luck" (e.g. NaN, infinity). Use `assume()`
  to bound the strategy to physically-valid metrics.
- **Runtime**: 5–8 new `@given` tests × ~100 examples → ~30s added to CI.
  Acceptable for a non-blocking `quality` job.
