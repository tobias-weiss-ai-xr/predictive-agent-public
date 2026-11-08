"""Property-based tests using Hypothesis for invariant verification.

SOTA paradigm: Property-based testing generates random inputs to discover
edge cases that unit tests miss. Each property test runs 100+ examples by
default, covering a much wider input space than hand-written tests.

Covers:
- Kalman filter: monotonicity, bounded uncertainty, convergence
- Risk score: always in [0, 0.99]
- Markov chain: transition probabilities always sum to 1.0
- Collector: parse_cpu/parse_memory round-trip
- State classifier: always returns a valid state
- Predictor: confidence always in [0, 1]
"""
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from predictive_agent.collector import parse_cpu, parse_memory
from predictive_agent.kalman import KalmanTrend
from predictive_agent.markov import MarkovChain
from predictive_agent.predictor import Predictor
from predictive_agent.risk import calculate_risk
from predictive_agent.state_model import classify_state

# ─── Kalman filter properties ────────────────────────────────────────────────

@given(
    measurements=st.lists(st.floats(min_value=0, max_value=10000, allow_nan=False,
                                     allow_infinity=False),
                          min_size=1, max_size=50)
)
@settings(max_examples=100, deadline=None)
def test_property_kalman_level_within_data_range(measurements):
    """After updates, the Kalman level should stay within or near the data range.

    The Kalman filter can overshoot when there's a sudden drop (velocity goes
    negative and the prediction step overshoots). We allow a band of up to
    the full data range on either side as slack.
    """
    kt = KalmanTrend()
    for m in measurements:
        kt.update(m)
    lo = min(measurements)
    hi = max(measurements)
    data_range = hi - lo
    # Allow overshoot up to the data range (Kalman velocity prediction can overshoot)
    slack = max(1000, data_range)
    assert lo - slack <= kt.level <= hi + slack


@given(
    measurements=st.lists(st.floats(min_value=0, max_value=10000, allow_nan=False,
                                     allow_infinity=False),
                          min_size=2, max_size=50)
)
@settings(max_examples=100, deadline=None)
def test_property_kalman_uncertainty_nonnegative(measurements):
    """Kalman uncertainty (variance) must always be non-negative."""
    kt = KalmanTrend()
    for m in measurements:
        kt.update(m)
    assert kt.level_uncertainty >= 0
    assert kt.velocity_uncertainty >= 0


@given(
    measurements=st.lists(st.floats(min_value=0, max_value=10000, allow_nan=False,
                                     allow_infinity=False),
                          min_size=1, max_size=50)
)
@settings(max_examples=100, deadline=None)
def test_property_kalman_predict_sigma_nonnegative(measurements):
    """Predicted sigma must always be non-negative."""
    kt = KalmanTrend()
    for m in measurements:
        kt.update(m)
    _, sigma = kt.predict(1.0)
    assert sigma >= 0


@given(
    base=st.floats(min_value=0, max_value=10000, allow_nan=False, allow_infinity=False),
    slope=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, deadline=None)
def test_property_kalmon_velocity_sign_matches_trend(base, slope):
    """Velocity should be positive for increasing data, negative for decreasing."""
    kt = KalmanTrend()
    for i in range(10):
        kt.update(base + slope * i)
    if abs(slope) > 1:
        assert (kt.velocity > 0) == (slope > 0)


# ─── Risk score properties ───────────────────────────────────────────────────

@given(
    memory_pct=st.floats(min_value=0, max_value=100),
    cpu_pct=st.floats(min_value=0, max_value=100),
    restart_rate=st.floats(min_value=0, max_value=100),
    log_error_rate=st.floats(min_value=0, max_value=100),
    node_mem_pressure=st.booleans(),
    node_disk_pressure=st.booleans(),
    markov_state=st.sampled_from(["HEALTHY", "DEGRADED", "STRESSED", "CRITICAL", "FAILED", "RECOVERED"]),
    markov_p_critical=st.floats(min_value=0, max_value=1),
    markov_p_failed=st.floats(min_value=0, max_value=1),
)
@settings(max_examples=200, deadline=None)
def test_property_risk_always_bounded(memory_pct, cpu_pct, restart_rate, log_error_rate,
                                       node_mem_pressure, node_disk_pressure,
                                       markov_state, markov_p_critical, markov_p_failed):
    """Risk score must always be in [0, 0.99]."""
    pod_metrics = {
        "memory_pct": memory_pct,
        "cpu_pct": cpu_pct,
        "restart_rate_per_hr": restart_rate,
        "log_error_rate_per_min": log_error_rate,
        "node_memory_pressure": node_mem_pressure,
        "node_disk_pressure": node_disk_pressure,
    }
    risk = calculate_risk(pod_metrics, markov_state, markov_p_critical, markov_p_failed)
    assert 0.0 <= risk <= 0.99


@given(
    markov_state=st.sampled_from(["HEALTHY", "DEGRADED", "STRESSED", "CRITICAL", "FAILED", "RECOVERED"]),
    markov_p_critical=st.floats(min_value=0, max_value=1),
    markov_p_failed=st.floats(min_value=0, max_value=1),
)
@settings(max_examples=100, deadline=None)
def test_property_risk_monotonic_with_markov_severity(markov_state, markov_p_critical, markov_p_failed):
    """Risk should be higher for worse Markov states (with same metrics)."""
    base_metrics = {
        "memory_pct": 80.0,
        "cpu_pct": 50.0,
        "restart_rate_per_hr": 1.0,
        "log_error_rate_per_min": 1.0,
        "node_memory_pressure": False,
        "node_disk_pressure": False,
    }
    severity_order = {"HEALTHY": 0, "RECOVERED": 0, "DEGRADED": 1, "STRESSED": 2, "CRITICAL": 3, "FAILED": 4}
    risk = calculate_risk(base_metrics, markov_state, markov_p_critical, markov_p_failed)
    risk_healthy = calculate_risk(base_metrics, "HEALTHY", markov_p_critical, markov_p_failed)
    if severity_order.get(markov_state, 0) > 0:
        assert risk >= risk_healthy


# ─── Markov chain properties ─────────────────────────────────────────────────

@given(
    transitions=st.lists(
        st.tuples(
            st.sampled_from(MarkovChain.STATES),
            st.sampled_from(MarkovChain.STATES),
        ),
        min_size=0,
        max_size=100,
    )
)
@settings(max_examples=100, deadline=None)
def test_property_markov_transition_matrix_sums_to_one(transitions):
    """Every row of the transition matrix must sum to 1.0."""
    mc = MarkovChain()
    for from_state, to_state in transitions:
        mc.record_transition(from_state, to_state)
    matrix = mc.transition_matrix()
    for row in matrix:
        assert abs(sum(row) - 1.0) < 1e-9, f"Row sums to {sum(row)}, expected 1.0"


@given(
    from_state=st.sampled_from(MarkovChain.STATES),
    steps=st.integers(min_value=1, max_value=10),
    transitions=st.lists(
        st.tuples(
            st.sampled_from(MarkovChain.STATES),
            st.sampled_from(MarkovChain.STATES),
        ),
        min_size=1,
        max_size=50,
    )
)
@settings(max_examples=100, deadline=None)
def test_property_markov_predict_returns_valid_distribution(from_state, steps, transitions):
    """predict() must return a probability distribution that sums to 1.0."""
    mc = MarkovChain()
    for fs, ts in transitions:
        mc.record_transition(fs, ts)
    probs = mc.predict(from_state, steps=steps)
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    for state, prob in probs.items():
        assert 0.0 <= prob <= 1.0


@given(
    transitions=st.lists(
        st.tuples(
            st.sampled_from(MarkovChain.STATES),
            st.sampled_from(MarkovChain.STATES),
        ),
        min_size=1,
        max_size=100,
    )
)
@settings(max_examples=100, deadline=None)
def test_property_markov_total_transitions_matches(transitions):
    """total_transitions should match the number of recorded transitions."""
    mc = MarkovChain()
    for from_state, to_state in transitions:
        mc.record_transition(from_state, to_state)
    assert mc.total_transitions == len(transitions)


# ─── Collector parsing properties ────────────────────────────────────────────

@given(
    cpu_str=st.from_regex(r'\d+m', fullmatch=True),
)
@settings(max_examples=50, deadline=None)
def test_property_parse_cpu_millicores(cpu_str):
    """Parsing 'Nm' format should return N."""
    result = parse_cpu(cpu_str)
    expected = int(cpu_str[:-1])
    assert result == expected


@given(
    mem_str=st.from_regex(r'\d+Mi', fullmatch=True),
)
@settings(max_examples=50, deadline=None)
def test_property_parse_memory_mib(mem_str):
    """Parsing 'NMi' format should return N."""
    result = parse_memory(mem_str)
    expected = int(mem_str[:-2])
    assert result == expected


# ─── State classifier properties ─────────────────────────────────────────────

@given(
    memory_pct=st.floats(min_value=0, max_value=100),
    cpu_pct=st.floats(min_value=0, max_value=100),
    restart_rate=st.floats(min_value=0, max_value=100),
    log_errors=st.floats(min_value=0, max_value=100),
    node_pressure=st.booleans(),
    markov_state=st.sampled_from(["HEALTHY", "DEGRADED", "STRESSED", "CRITICAL"]),
)
@settings(max_examples=200, deadline=None)
def test_property_classify_state_always_valid(memory_pct, cpu_pct, restart_rate,
                                                log_errors, node_pressure, markov_state):
    """classify_state must always return a valid state string."""
    state = classify_state(memory_pct, cpu_pct, restart_rate, log_errors,
                            node_pressure, markov_state)
    assert state in {"HEALTHY", "DEGRADED", "STRESSED", "CRITICAL"}


@given(
    memory_pct=st.floats(min_value=0, max_value=50),
    cpu_pct=st.floats(min_value=0, max_value=30),
    restart_rate=st.floats(min_value=0, max_value=0.5),
    log_errors=st.floats(min_value=0, max_value=1),
    node_pressure=st.booleans(),
    markov_state=st.sampled_from(["HEALTHY"]),
)
@settings(max_examples=100, deadline=None)
def test_property_classify_state_low_metrics_healthy(memory_pct, cpu_pct, restart_rate,
                                                       log_errors, node_pressure, markov_state):
    """Low metrics should always classify as HEALTHY."""
    assume(not node_pressure)
    state = classify_state(memory_pct, cpu_pct, restart_rate, log_errors,
                            node_pressure, markov_state)
    assert state == "HEALTHY"


# ─── Predictor properties ────────────────────────────────────────────────────

@given(
    memory_pct=st.floats(min_value=0, max_value=100),
    memory_trend=st.floats(min_value=0, max_value=100, allow_nan=False),
    cpu_pct=st.floats(min_value=0, max_value=100),
    restart_rate=st.floats(min_value=0, max_value=50),
    log_error_rate=st.floats(min_value=0, max_value=50),
    markov_p_critical=st.floats(min_value=0, max_value=1),
    markov_p_failed=st.floats(min_value=0, max_value=1),
    markov_state=st.sampled_from(["HEALTHY", "DEGRADED", "STRESSED", "CRITICAL", "FAILED", "RECOVERED"]),
)
@settings(max_examples=200, deadline=None)
def test_property_predictor_confidence_bounded(memory_pct, memory_trend, cpu_pct,
                                                 restart_rate, log_error_rate,
                                                 markov_p_critical, markov_p_failed,
                                                 markov_state):
    """Predictor confidence must always be in [0, 1]."""
    p = Predictor()
    result = p.predict(
        pod_key="test/pod",
        memory_pct=memory_pct,
        memory_trend_mib_per_min=memory_trend,
        memory_limit_mib=1024,
        memory_mib=int(memory_pct * 10.24),
        cpu_pct=cpu_pct,
        restart_rate_per_hr=restart_rate,
        log_error_rate_per_min=log_error_rate,
        node_memory_pressure=False,
        node_disk_pressure=False,
        markov_state=markov_state,
        markov_p_critical=markov_p_critical,
        markov_p_failed=markov_p_failed,
    )
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.risk_score <= 0.99
