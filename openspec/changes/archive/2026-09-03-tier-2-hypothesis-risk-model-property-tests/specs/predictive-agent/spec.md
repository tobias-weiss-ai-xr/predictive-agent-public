## ADDED Requirements

### Requirement: Bounded-metric property tests for the risk model
A Hypothesis strategy over physically-valid pod metrics SHALL be used to
verify invariants of the prediction + state-classification pipeline.

#### Scenario: risk_score is always normalized to [0, 1]
- **WHEN** `Predictor.predict` is invoked with any combination of
  memory_pct ∈ [0, 100], cpu_pct ∈ [0, 100], restart_rate_per_hr ∈ [0, ∞),
  and log_error_rate_per_min ∈ [0, ∞)
- **THEN** `result.risk_score` is a float in the closed interval [0, 1]
- **AND** `predict` does not raise for any such input

#### Scenario: state classifier always yields a canonical Markov state
- **WHEN** `classify_state` is invoked with any combination of the metric
  arguments within their valid domains
- **THEN** the returned state is one of `HEALTHY`, `DEGRADED`, `STRESSED`,
  `CRITICAL` (never `None`, never a novel string)

#### Scenario: StateModel.update_pod keeps state within the Markov set
- **WHEN** a pod tracker is updated via `StateModel.update_pod` with
  random-but-valid metrics (memory ≥ 0, cpu ≥ 0, restart_count ≥ 0)
- **THEN** `tracker.state` and `tracker.prev_state` remain in the canonical
Markov set after the update

### Requirement: Hypothesis determinism in CI
The CI property-test job SHALL pin the Hypothesis random seed for
reproducible shrinkage.

#### Scenario: CI uses a pinned seed for reproducibility
- **WHEN** the Hypothesis job runs in CI
- **THEN** it passes `--hypothesis-seed=0` so a failing example shrinks
  to the same minimal input on every run, and `.hypothesis/` failures
  are committed for replay
