"""Property-based tests for the risk model + state machine (tier-2 change
``tier-2-hypothesis-risk-model-property-tests``).

Hypothesis strategies generate physically-valid pod metrics and verify
invariants that example-based tests only check after a human thinks of
the specific input:

- ``Predictor.predict`` never raises on valid input and always returns
  ``risk_score`` in the closed interval [0, 1].
- ``classify_state`` always yields a canonical Markov state string.
- ``StateModel.update_pod`` keeps the tracker state inside the canonical
  Markov set.
- Restart-rate derivation stays finite and non-negative.

Determinism: a ``derandomize=True`` profile pins the generation so CI
failures shrink reproducibly (stronger than a pinned seed: no database
dependence).
"""

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from predictive_agent.config import MARKOV_STATES
from predictive_agent.predictor import Predictor
from predictive_agent.state_model import StateModel, classify_state

# Deterministic profile: same examples on every run (CI + local).
settings.register_profile(
    "ci", max_examples=100, deadline=None, derandomize=True
)
settings.load_profile("ci")


# ── Strategies (bounded to physically-valid domains) ───────────────────────

memory_pcts = st.floats(min_value=0.0, max_value=100.0, allow_nan=False,
                        allow_infinity=False)
cpu_pcts = st.floats(min_value=0.0, max_value=100.0, allow_nan=False,
                     allow_infinity=False)
memory_mibs = st.integers(min_value=0, max_value=65536)
memory_limits = st.integers(min_value=1, max_value=262144)  # ≥1: no div-by-0
cpu_ms = st.integers(min_value=0, max_value=8000)
restart_counts = st.integers(min_value=0, max_value=1000)
restart_rates = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False,
                          allow_infinity=False)
log_errors = st.integers(min_value=0, max_value=10000)
log_rates = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False,
                      allow_infinity=False)
trends = st.floats(min_value=-500.0, max_value=500.0, allow_nan=False,
                   allow_infinity=False)  # memory freeing ⇒ negative trend
markov_states = st.sampled_from(MARKOV_STATES)
booleans = st.booleans()


# ── Properties ──────────────────────────────────────────────────────────────

@settings(max_examples=100, deadline=None, derandomize=True)
@given(pod_key=st.text(min_size=1, max_size=20),
       memory_pct=memory_pcts, memory_trend=trends,
       memory_limit_mib=memory_limits, memory_mib=memory_mibs,
       cpu_pct=cpu_pcts, restart_rate=restart_rates,
       log_rate=log_rates,
       node_mem=booleans, node_disk=booleans,
       markov_state=st.sampled_from(["HEALTHY", "DEGRADED", "STRESSED",
                                     "CRITICAL"]),
       p_crit=st.floats(min_value=0.0, max_value=1.0),
       p_fail=st.floats(min_value=0.0, max_value=1.0))
def test_predict_risk_score_in_unit_interval(pod_key, memory_pct,
                                             memory_trend, memory_limit_mib,
                                             memory_mib, cpu_pct,
                                             restart_rate, log_rate,
                                             node_mem, node_disk,
                                             markov_state, p_crit, p_fail):
    """predict() never raises on valid metrics; risk_score ∈ [0, 1]."""
    predictor = Predictor(risk_threshold=0.5)
    result = predictor.predict(
        pod_key=pod_key,
        memory_pct=memory_pct,
        memory_trend_mib_per_min=memory_trend,
        memory_limit_mib=memory_limit_mib,
        memory_mib=memory_mib,
        cpu_pct=cpu_pct,
        restart_rate_per_hr=restart_rate,
        log_error_rate_per_min=log_rate,
        node_memory_pressure=node_mem,
        node_disk_pressure=node_disk,
        markov_state=markov_state,
        markov_p_critical=p_crit,
        markov_p_failed=p_fail,
    )
    assert isinstance(result.risk_score, float)
    assert 0.0 <= result.risk_score <= 1.0
    assert math.isfinite(result.risk_score)


@settings(max_examples=100, deadline=None, derandomize=True)
@given(memory_pct=memory_pcts, cpu_pct=cpu_pcts,
       restart_rate=restart_rates, log_errors=log_errors,
       node_pressure=booleans, markov_state=markov_states)
def test_classify_state_always_canonical(memory_pct, cpu_pct, restart_rate,
                                         log_errors, node_pressure,
                                         markov_state):
    """classify_state() returns one of the canonical states, never None/new str."""
    state = classify_state(memory_pct, cpu_pct, restart_rate, log_errors,
                           node_pressure, markov_state)
    assert state in MARKOV_STATES
    assert isinstance(state, str)


@settings(max_examples=100, deadline=None, derandomize=True)
@given(namespace=st.text(min_size=1, max_size=10),
       name=st.text(min_size=1, max_size=10),
       memory_mib=memory_mibs, memory_limit_mib=memory_limits,
       cpu_m=cpu_ms, restart_count=restart_counts,
       log_errors=log_errors, node_pressure=booleans)
def test_state_model_update_state_remains_canonical(namespace, name,
                                                    memory_mib,
                                                    memory_limit_mib, cpu_m,
                                                    restart_count, log_errors,
                                                    node_pressure):
    """update_pod() keeps tracker.state within the canonical Markov set."""
    model = StateModel()
    tracker = model.update_pod(namespace, name, memory_mib, memory_limit_mib,
                               cpu_m, restart_count, log_errors,
                               node_pressure)
    assert tracker.state in MARKOV_STATES
    # A second update (transition path) must stay canonical as well.
    tracker2 = model.update_pod(namespace, name, memory_mib, memory_limit_mib,
                                cpu_m, restart_count + 1, log_errors,
                                node_pressure)
    assert tracker2.state in MARKOV_STATES


@settings(max_examples=100, deadline=None, derandomize=True)
@given(memory_pct=memory_pcts, memory_trend=trends,
       memory_limit_mib=memory_limits, memory_mib=memory_mibs,
       cpu_pct=cpu_pcts, restart_rate=restart_rates,
       markov_state=st.sampled_from(["HEALTHY", "DEGRADED", "STRESSED",
                                     "CRITICAL"]))
def test_predict_ttf_finite_or_none(memory_pct, memory_trend,
                                    memory_limit_mib, memory_mib, cpu_pct,
                                    restart_rate, markov_state):
    """TTF is None (no failure path) or finite and >= 0.

ttf_minutes == 0 is a *valid* outcome: it means remaining headroom <= 0
(pod at/over its memory limit) — failure is imminent. Verified against
predictor.py:135 (the `else: ttf_minutes = 0` branch)."""
    predictor = Predictor(risk_threshold=0.5)
    result = predictor.predict(
        pod_key="ns/p",
        memory_pct=memory_pct,
        memory_trend_mib_per_min=memory_trend,
        memory_limit_mib=memory_limit_mib,
        memory_mib=memory_mib,
        cpu_pct=cpu_pct,
        restart_rate_per_hr=restart_rate,
        log_error_rate_per_min=0.0,
        node_memory_pressure=False,
        node_disk_pressure=False,
        markov_state=markov_state,
        markov_p_critical=0.5,
        markov_p_failed=0.1,
    )
    if result.ttf_minutes is not None:
        assert math.isfinite(result.ttf_minutes)
        assert result.ttf_minutes >= 0  # 0 = at/over limit ⇒ imminent
        assert result.ttf_minutes <= 43200  # capped at 30 days
    assert math.isfinite(result.confidence)
    assert 0.0 <= result.confidence <= 1.0


# ── calculate_risk invariants (risk.py had no property coverage) ──────────

from predictive_agent.risk import calculate_risk

_base_metrics = {
    "memory_pct": 50.0,
    "memory_trend_mib_per_min": 0.5,
    "cpu_pct": 30.0,
    "restart_rate_per_hr": 0.0,
    "log_error_rate_per_min": 0.0,
    "node_memory_pressure": False,
    "node_disk_pressure": False,
}

_scalars = st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False)
_states = st.sampled_from(MARKOV_STATES)
_all_scalars = st.fixed_dictionaries({k: _scalars for k in _base_metrics})


@given(metrics=_all_scalars,
       markov_state=_states,
       p_c=st.floats(0, 1), p_f=st.floats(0, 1))
@settings(max_examples=200, derandomize=True)
def test_calculate_risk_stays_in_unit_interval(metrics, markov_state, p_c, p_f):
    risk = calculate_risk(metrics, markov_state, p_c, p_f)
    assert 0.0 <= risk <= 1.0


@given(extra=st.floats(min_value=0, max_value=49))
@settings(max_examples=100, derandomize=True)
def test_calculate_risk_monotonic_in_memory_pct(extra):
    """Raising any single signal must never lower the score."""
    low = calculate_risk({**_base_metrics, "memory_pct": 50.0}, "HEALTHY", 0.01, 0.001)
    high = calculate_risk({**_base_metrics, "memory_pct": 50.0 + extra}, "HEALTHY", 0.01, 0.001)
    assert high >= low


@given(markov_state=_states)
@settings(max_examples=20, derandomize=True)
def test_calculate_risk_accepts_every_markov_state(markov_state):
    assert isinstance(calculate_risk(_base_metrics, markov_state, 0.01, 0.001), float)
