"""Test MarkovChain class for state transitions."""
import pytest

from predictive_agent.markov import MarkovChain


def test_markov_chain_initialization():
    """Test that MarkovChain initializes correctly."""
    mc = MarkovChain()
    assert mc.STATES == ["HEALTHY", "DEGRADED", "STRESSED", "CRITICAL", "FAILED", "RECOVERED"]
    assert mc.total_transitions == 0


def test_markov_chain_record_transition():
    """Test recording a state transition."""
    mc = MarkovChain()
    mc.record_transition("HEALTHY", "DEGRADED")
    assert mc.total_transitions == 1


def test_markov_chain_transition_matrix():
    """Test transition matrix calculation."""
    mc = MarkovChain()
    mc.record_transition("HEALTHY", "DEGRADED")
    mc.record_transition("DEGRADED", "STRESSED")

    matrix = mc.transition_matrix()
    assert len(matrix) == 6
    assert len(matrix[0]) == 6
    # Each row should sum to ~1.0
    for row in matrix:
        assert abs(sum(row) - 1.0) < 0.01


def test_markov_chain_predict():
    """Test state prediction."""
    mc = MarkovChain()
    mc.record_transition("HEALTHY", "DEGRADED")
    mc.record_transition("DEGRADED", "STRESSED")

    prediction = mc.predict("HEALTHY", steps=1)
    assert "DEGRADED" in prediction
    assert prediction["DEGRADED"] > 0


def test_markov_chain_persistence():
    """Test saving and loading state."""
    mc1 = MarkovChain()
    mc1.record_transition("HEALTHY", "DEGRADED")

    data = mc1.to_dict()
    assert "counts" in data
    assert data["total_transitions"] == 1

    mc2 = MarkovChain.from_dict(data)
    assert mc2.total_transitions == 1


def test_markov_chain_most_likely_next():
    """Test most likely next state."""
    mc = MarkovChain()
    # Record many HEALTHY→DEGRADED transitions
    for _ in range(100):
        mc.record_transition("HEALTHY", "DEGRADED")

    state, prob = mc.most_likely_next("HEALTHY")
    assert state == "DEGRADED"
    assert prob > 0.5


def test_markov_chain_multi_step():
    """Test multi-step prediction."""
    mc = MarkovChain()
    for _ in range(50):
        mc.record_transition("HEALTHY", "DEGRADED")
    for _ in range(50):
        mc.record_transition("DEGRADED", "STRESSED")

    pred_1 = mc.predict("HEALTHY", steps=1)
    pred_5 = mc.predict("HEALTHY", steps=5)
    # Multi-step should spread probability more
    assert pred_5["STRESSED"] > pred_1["STRESSED"]


def test_markov_chain_unknown_state():
    """Test that unknown state defaults to index 0."""
    mc = MarkovChain()
    mc.record_transition("UNKNOWN", "HEALTHY")
    assert mc.total_transitions == 1


def test_markov_chain_prior_counts():
    """Test that prior counts are present initially."""
    mc = MarkovChain()
    matrix = mc.transition_matrix()
    # HEALTHY should mostly stay HEALTHY with prior
    assert matrix[0][0] > 0.8  # 9.5/10


# ─── Mutation-test killer battery ────────────────────────────────────────
# Pin the contracts that a prior campaign let slip: normalization guard,
# uniform fallback, prior-matrix values, observation learning, and the
# last_updated timestamp.


def test_markov_chain_zero_row_uniform_fallback():
    """A counts row summing to zero must yield the uniform 1/6 row, not
    divide by zero (the total > 0 guard is load-bearing)."""
    mc = MarkovChain()
    mc.counts[2] = [0.0] * 6  # simulate a state with no prior at all
    matrix = mc.transition_matrix()
    assert all(abs(p - 1 / 6) < 1e-12 for p in matrix[2])
    # other rows keep their normalized priors
    assert abs(sum(matrix[0]) - 1.0) < 1e-9


def test_markov_chain_prior_matrix_values():
    """Fresh chain serves the normalized prior; key entries are calibrated."""
    mc = MarkovChain()
    matrix = mc.transition_matrix()
    assert abs(matrix[0][0] - (9.5 / 10.0)) < 1e-9   # HEALTHY→HEALTHY 0.95
    assert abs(matrix[4][4] - (8 / 10.0)) < 1e-9     # FAILED→FAILED 0.8
    assert abs(matrix[4][5] - (2 / 10.0)) < 1e-9     # FAILED→RECOVERED 0.2
    for row in matrix:
        assert abs(sum(row) - 1.0) < 1e-9  # rows are stochastic


def test_markov_chain_observations_dominate_priors():
    """Repeated observed transitions must shift probability mass."""
    mc = MarkovChain()
    for _ in range(10):
        mc.record_transition("HEALTHY", "FAILED")
    matrix = mc.transition_matrix()
    # prior P(H→F)=0.0; after 10 observations: 10/20
    assert abs(matrix[0][4] - 0.5) < 1e-9
    assert mc.total_transitions == 10


def test_markov_chain_last_updated_contract():
    """last_updated starts at 0.0 (unset) and is a positive timestamp
    after any recorded transition (staleness checks rely on this)."""
    mc = MarkovChain()
    assert mc.last_updated == 0.0
    mc.record_transition("HEALTHY", "DEGRADED")
    assert mc.last_updated > 0.0


def test_markov_chain_multi_step_distribution_sums_to_one():
    """Multi-step prediction is a stochastic vector regardless of steps."""
    mc = MarkovChain()
    for steps in (1, 2, 5):
        probs = mc.predict("STRESSED", steps=steps)
        assert abs(sum(probs.values()) - 1.0) < 1e-9
        assert set(probs) == set(MarkovChain.STATES)


# ═══════════════════════════════════════════════════════════════════════════
# Mutation-survivor killers: unknown-state fallbacks, dict-key contract,
# zero-row normalization boundary, serialization round-trip shape.
# ═══════════════════════════════════════════════════════════════════════════

import pytest as _pytest


def test_record_transition_unknown_from_state_lands_on_healthy_row():
    """Unknown from-state falls back to index 0 (HEALTHY), not 1/None."""
    chain = MarkovChain()
    before = chain.counts[0][MarkovChain.STATE_IDX["FAILED"]]
    chain.record_transition("BOGUS_STATE", "FAILED")
    assert chain.counts[0][MarkovChain.STATE_IDX["FAILED"]] == before + 1
    assert chain.total_transitions == 1
    assert chain.counts[1][MarkovChain.STATE_IDX["FAILED"]] == MarkovChain.PRIOR_COUNTS[1][MarkovChain.STATE_IDX["FAILED"]]


def test_record_transition_unknown_to_state_lands_on_healthy_column():
    """Unknown to-state falls back to column 0 (HEALTHY)."""
    chain = MarkovChain()
    before = chain.counts[MarkovChain.STATE_IDX["STRESSED"]][0]
    chain.record_transition("STRESSED", "BOGUS_STATE")
    assert chain.counts[MarkovChain.STATE_IDX["STRESSED"]][0] == before + 1


def test_transition_matrix_zero_row_boundary_is_exact():
    """total > 0 (not > 1): a row summing to exactly 1 normalizes to identity."""
    chain = MarkovChain()
    chain.counts[2] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    matrix = chain.transition_matrix()
    assert matrix[2] == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def test_transition_matrix_shape_contract():
    chain = MarkovChain()
    matrix = chain.transition_matrix()
    assert len(matrix) == 6 and all(len(row) == 6 for row in matrix)
    assert all(sum(row) == _pytest.approx(1.0) for row in matrix)


def test_predict_default_steps_is_one():
    chain = MarkovChain()
    assert chain.predict("HEALTHY") == chain.predict("HEALTHY", 1)


def test_predict_unknown_state_uses_healthy_fallback():
    chain = MarkovChain()
    assert chain.predict("BOGUS") == chain.predict("HEALTHY")


def test_predict_preserves_markov_property():
    """Rows sum to 1 at every step and predictions differ per source state."""
    chain = MarkovChain()
    for state in MarkovChain.STATES:
        for steps in (1, 2, 3):
            dist = chain.predict(state, steps)
            assert set(dist) == set(MarkovChain.STATES)
            assert sum(dist.values()) == _pytest.approx(1.0), (state, steps)
    assert chain.predict("DEGRADED", 1) != chain.predict("HEALTHY", 1)


def test_most_likely_next_passthrough_semantics():
    """most_likely_next == argmax of predict at the same step count."""
    chain = MarkovChain()
    for state in ("HEALTHY", "DEGRADED", "CRITICAL"):
        for steps in (1, 2):
            probs = chain.predict(state, steps)
            best = max(probs, key=probs.get)
            assert chain.most_likely_next(state, steps) == (best, probs[best])
    assert chain.most_likely_next("HEALTHY") == chain.most_likely_next("HEALTHY", 1)


def test_to_dict_exact_key_set():
    chain = MarkovChain()
    chain.record_transition("HEALTHY", "DEGRADED")
    data = chain.to_dict()
    assert set(data) == {"counts", "total_transitions", "last_updated"}
    assert data["total_transitions"] == 1


def test_from_dict_roundtrip_exact_values():
    chain = MarkovChain()
    chain.record_transition("HEALTHY", "DEGRADED")
    chain.record_transition("DEGRADED", "FAILED")
    restored = MarkovChain.from_dict(chain.to_dict())
    assert restored.counts == chain.counts
    assert restored.total_transitions == 2
    assert restored.last_updated == chain.last_updated


def test_from_dict_missing_keys_fall_back_exactly():
    """Partial dicts fall back to 0 / prior copies — never None or 1."""
    chain = MarkovChain.from_dict({"counts": [[1.0] * 6] * 6})
    assert chain.counts == [[1.0] * 6] * 6
    assert chain.total_transitions == 0
    assert chain.last_updated == 0
    priors = MarkovChain.from_dict({})
    assert priors.counts == [row[:] for row in MarkovChain.PRIOR_COUNTS]
    assert priors.total_transitions == 0
    assert priors.last_updated == 0
    # from_dict(None) also yields pristine prior state (falsy data guard)
    pristine = MarkovChain.from_dict(None)
    assert pristine.counts == [row[:] for row in MarkovChain.PRIOR_COUNTS]
    assert pristine.total_transitions == 0


def test_from_dict_truthy_payload_without_counts_uses_priors():
    """A truthy dict lacking 'counts' must fall back to PRIOR_COUNTS copies
    (default None would crash the first transition_matrix() call)."""
    chain = MarkovChain.from_dict({"anything": 1})
    assert chain.counts == [row[:] for row in MarkovChain.PRIOR_COUNTS]
    assert sum(chain.predict("HEALTHY", 1).values()) == pytest.approx(1.0)
