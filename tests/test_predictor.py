"""Test prediction engine."""
from predictive_agent.predictor import PredictionResult, Predictor


def test_prediction_result_creation():
    """Test PredictionResult dataclass."""
    pr = PredictionResult(
        pod_key="opendesk/openldap-0",
        risk_score=0.75,
        ttf_minutes=15,
        confidence=0.85,
        markov_state="STRESSED",
        memory_trend=2.5,
        cpu_trend=10.0,
        memory_pct=85.0,
        cpu_pct=60.0,
    )
    assert pr.pod_key == "opendesk/openldap-0"
    assert pr.risk_score == 0.75
    assert pr.ttf_minutes == 15
    assert pr.confidence == 0.85


def test_predictor_creation():
    """Test Predictor creation."""
    p = Predictor(risk_threshold=0.5)
    assert p.risk_threshold == 0.5


def test_predictor_predict_low_risk():
    """Test prediction for low-risk pod."""
    p = Predictor(risk_threshold=0.5)
    result = p.predict(
        pod_key="ns/pod-0",
        memory_pct=40.0,
        memory_trend_mib_per_min=0.1,
        memory_limit_mib=1024,
        memory_mib=400,
        cpu_pct=20.0,
        restart_rate_per_hr=0.0,
        log_error_rate_per_min=0.0,
        node_memory_pressure=False,
        node_disk_pressure=False,
        markov_state="HEALTHY",
        markov_p_critical=0.01,
        markov_p_failed=0.001,
    )
    assert result.risk_score < 0.5
    assert result.markov_state == "HEALTHY"


def test_predictor_predict_high_risk():
    """Test prediction for high-risk pod."""
    p = Predictor(risk_threshold=0.5)
    result = p.predict(
        pod_key="ns/pod-0",
        memory_pct=95.0,
        memory_trend_mib_per_min=5.0,
        memory_limit_mib=1024,
        memory_mib=970,
        cpu_pct=90.0,
        restart_rate_per_hr=5.0,
        log_error_rate_per_min=10.0,
        node_memory_pressure=True,
        node_disk_pressure=False,
        markov_state="CRITICAL",
        markov_p_critical=0.5,
        markov_p_failed=0.2,
    )
    assert result.risk_score > 0.5
    assert result.ttf_minutes is not None
    assert result.ttf_minutes > 0


def test_predictor_predict_no_ttf():
    """Test prediction when no TTF (stable memory)."""
    p = Predictor(risk_threshold=0.5)
    result = p.predict(
        pod_key="ns/pod-0",
        memory_pct=50.0,
        memory_trend_mib_per_min=0.0,
        memory_limit_mib=1024,
        memory_mib=512,
        cpu_pct=30.0,
        restart_rate_per_hr=0.0,
        log_error_rate_per_min=0.0,
        node_memory_pressure=False,
        node_disk_pressure=False,
        markov_state="HEALTHY",
        markov_p_critical=0.0,
        markov_p_failed=0.0,
    )
    assert result.ttf_minutes is None or result.ttf_minutes == 0


def test_predictor_at_risk():
    """Test getting pods above risk threshold."""
    p = Predictor(risk_threshold=0.3)
    # Add some predictions
    p.add_prediction("ns/pod-1", PredictionResult(
        "ns/pod-1", 0.8, 10, 0.9, "CRITICAL", 3.0, 50.0, 90.0, 80.0
    ))
    p.add_prediction("ns/pod-2", PredictionResult(
        "ns/pod-2", 0.1, None, 0.5, "HEALTHY", 0.0, 10.0, 40.0, 20.0
    ))
    at_risk = p.get_at_risk()
    assert len(at_risk) == 1
    assert at_risk[0].pod_key == "ns/pod-1"


# ─── Mutation-test killer battery ────────────────────────────────────────
# Pin the TTF formula, confidence table, anomaly aggregation, and the
# at-risk boundary that a prior campaign let slip.

_KW = dict(
    memory_pct=40.0,
    memory_trend_mib_per_min=0.1,
    memory_limit_mib=1024,
    memory_mib=400,
    cpu_pct=20.0,
    restart_rate_per_hr=0.0,
    log_error_rate_per_min=0.0,
    node_memory_pressure=False,
    node_disk_pressure=False,
    markov_state="HEALTHY",
    markov_p_critical=0.01,
    markov_p_failed=0.001,
)


def _predict(p, **over):
    return p.predict(pod_key="ns/pod-ttf", **{**_KW, **over})


def test_ttf_minutes_exact_value():
    """TTF = int((limit - current) / trend) minutes."""
    p = Predictor()
    r = _predict(p, memory_trend_mib_per_min=10.0)
    assert r.ttf_minutes == 62  # (1024-400)/10 = 62.4


def test_ttf_capped_at_30_days():
    """Raw TTF beyond 43200 min (30 days) is capped, not overflowed."""
    p = Predictor()
    r = _predict(p, memory_trend_mib_per_min=0.01)  # 624/0.01 = 62400
    assert r.ttf_minutes == 43200


def test_ttf_small_but_meaningful_trend_still_predicted():
    """A 0.5 MiB/min trend is meaningful — only <1e-10 is filtered."""
    p = Predictor()
    r = _predict(p, memory_trend_mib_per_min=0.5)  # 624/0.5 = 1248
    assert r.ttf_minutes == 1248


def test_ttf_none_without_trend_or_limit():
    p = Predictor()
    assert _predict(p, memory_trend_mib_per_min=0.0).ttf_minutes is None
    r = _predict(p, memory_limit_mib=0, memory_trend_mib_per_min=10.0)
    assert r.ttf_minutes is None  # no limit → no TTF even with a trend


def test_ttf_zero_when_at_or_over_limit():
    p = Predictor()
    assert _predict(p, memory_mib=1024).ttf_minutes == 0
    assert _predict(p, memory_mib=1100).ttf_minutes == 0


def test_confidence_table_and_volatility():
    """Confidence: base by state × (1 - p_crit - p_fail), damped for
    volatile trends; unknown states fall back to 0.5."""
    stable = dict(memory_trend_mib_per_min=2.0, markov_p_critical=0.0, markov_p_failed=0.0)
    p = Predictor()
    assert _predict(p, markov_state="HEALTHY", **stable).confidence == 0.9
    assert _predict(p, markov_state="CRITICAL", **stable).confidence == 0.3
    assert _predict(p, markov_state="MYSTERY", **stable).confidence == 0.5
    volatile = dict(memory_trend_mib_per_min=12.0, markov_p_critical=0.0, markov_p_failed=0.0)
    assert _predict(p, markov_state="HEALTHY", **volatile).confidence == 0.9 * 0.7
    moderate = dict(memory_trend_mib_per_min=6.0, markov_p_critical=0.0, markov_p_failed=0.0)
    assert _predict(p, markov_state="HEALTHY", **moderate).confidence == 0.9 * 0.85


def test_anomaly_score_takes_max_of_both_signals():
    p = Predictor()
    assert _predict(p, memory_anomaly_score=1.0, cpu_anomaly_score=4.0).anomaly_score == 4.0
    assert _predict(p, memory_anomaly_score=4.0, cpu_anomaly_score=1.0).anomaly_score == 4.0


def test_get_at_risk_includes_pods_exactly_at_threshold():
    """'At risk' is risk_score >= threshold (boundary inclusive)."""
    p = Predictor(risk_threshold=0.5)
    r = _predict(p)
    p.risk_threshold = r.risk_score  # exactly at threshold
    assert r in p.get_at_risk()


# ═══════════════════════════════════════════════════════════════════════════
# Mutation-survivor killers: pod_metrics key contract, result-field wiring,
# ttf boundary paths, confidence table semantics.
# ═══════════════════════════════════════════════════════════════════════════

import pytest


def _pred(**over):
    """Baseline quiet pod; kwargs override predict() parameters directly."""
    args = dict(
        pod_key="ns/pod",
        memory_pct=50.0,
        memory_trend_mib_per_min=0.0,
        memory_limit_mib=1024,
        memory_mib=512,
        cpu_pct=30.0,
        restart_rate_per_hr=0.0,
        log_error_rate_per_min=0.0,
        node_memory_pressure=False,
        node_disk_pressure=False,
        markov_state="HEALTHY",
        markov_p_critical=0.01,
        markov_p_failed=0.001,
    )
    args.update(over)
    return Predictor().predict(**args)


# (predict kwargs, expected effect): each pod_metrics key must reach
# calculate_risk — kills XXkeyXX / UPPER-KEY mutants in the dict literal.
_KEY_ESCALATIONS = [
    {"memory_pct": 96.0},
    {"memory_trend_mib_per_min": 200.0},  # ttf ~2.5 min
    {"cpu_pct": 96.0},
    {"cpu_trend_m_per_min": 100.0},
    {"restart_rate_per_hr": 6.0},
    {"log_error_rate_per_min": 12.0},
    {"node_memory_pressure": True},
    {"node_disk_pressure": True},
    {"memory_anomaly_score": 3.5},
    {"cpu_anomaly_score": 3.5},
    {"blast_radius": 25},
    {"pod_phase": "Failed"},
    {"container_ready": False},
    {"wait_state": "CrashLoopBackOff"},
    {"terminated": True, "terminated_reason": "OOMKilled"},
    {"pod_scheduled": False},
]


@pytest.mark.parametrize("over", _KEY_ESCALATIONS)
def test_every_signal_key_reaches_risk(over):
    base = _pred().risk_score
    escalated = _pred(**over).risk_score
    assert escalated > base, f"signal dropped in pod_metrics: {over}"


def test_result_field_wiring_exact():
    result = _pred(
        pod_key="ns/x", memory_pct=77.0, cpu_pct=44.0,
        memory_trend_mib_per_min=3.25, cpu_trend_m_per_min=9.5,
        memory_anomaly_score=1.0, cpu_anomaly_score=2.0, blast_radius=13,
        markov_state="STRESSED", markov_p_critical=0.2, markov_p_failed=0.1,
    )
    assert result.pod_key == "ns/x"
    assert result.memory_pct == 77.0
    assert result.cpu_pct == 44.0
    assert result.memory_trend == 3.25
    assert result.cpu_trend == 9.5
    assert result.anomaly_score == 2.0  # max of both anomaly signals
    assert result.blast_radius == 13
    assert result.markov_state == "STRESSED"
    assert result.risk_score > 0.15  # stressed markov state raises risk


def test_ttf_branch_contracts():
    # int truncation: remaining 50 / trend 3 = 16.67 -> 16
    assert _pred(memory_trend_mib_per_min=3.0, memory_limit_mib=62,
                    memory_mib=12).ttf_minutes == 16
    # cap: raw 50000 -> exactly 43200
    assert _pred(memory_trend_mib_per_min=1.0, memory_limit_mib=60000,
                    memory_mib=10000).ttf_minutes == 43200
    # limit == 1 boundary: guard is limit > 0, so ttf is still produced
    assert _pred(memory_trend_mib_per_min=1.0, memory_limit_mib=1,
                    memory_mib=0).ttf_minutes == 1
    # remaining == 1: ttf = 1/trend, int-truncated (guard must not be > 1)
    assert _pred(memory_trend_mib_per_min=0.8, memory_limit_mib=100,
                 memory_mib=99).ttf_minutes == 1
    # trend == 1e-10: not below the near-zero guard -> capped ttf, not None
    assert _pred(memory_trend_mib_per_min=1e-10, memory_limit_mib=100,
                    memory_mib=0).ttf_minutes == 43200
    # at/over limit -> 0; no trend or no limit -> None
    assert _pred(memory_trend_mib_per_min=5.0, memory_limit_mib=100,
                    memory_mib=100).ttf_minutes == 0
    assert _pred(memory_trend_mib_per_min=0.0).ttf_minutes is None
    assert _pred(memory_limit_mib=0).ttf_minutes is None


def test_confidence_state_table_ordering():
    p = Predictor()
    def conf(state):
        return p._calculate_confidence(0.0, state, 0.0, 0.0)
    chain = [conf(s) for s in ("HEALTHY", "DEGRADED", "STRESSED", "CRITICAL", "FAILED")]
    assert chain == sorted(chain, reverse=True)
    assert conf("BOGUS") == pytest.approx(conf("STRESSED"))
    assert chain[-1] < conf("BOGUS")


def test_confidence_transition_probability_dampening():
    p = Predictor()
    base = p._calculate_confidence(0.0, "HEALTHY", 0.0, 0.0)
    damped = p._calculate_confidence(0.0, "HEALTHY", 0.5, 0.0)
    assert damped < base
    # pc + pf >= 1.0 clamps the factor to zero confidence
    assert p._calculate_confidence(0.0, "HEALTHY", 0.6, 0.6) == 0.0
    # pf alone dampens as much as pc alone (additive, not one-sided)
    assert p._calculate_confidence(0.0, "HEALTHY", 0.0, 0.5) == pytest.approx(damped)


def test_confidence_volatility_tiers_and_boundaries():
    p = Predictor()
    def conf(trend):
        return p._calculate_confidence(trend, "HEALTHY", 0.0, 0.0)
    assert conf(10.0) > conf(11.0)  # boundary: >10 (not >=10)
    assert conf(5.0) > conf(6.0)  # boundary: >5
    assert conf(11.0) < conf(6.0)  # strongest tier is the most penalized
    assert conf(-11.0) == pytest.approx(conf(11.0))  # abs() semantics
    assert conf(0.0) == pytest.approx(conf(5.0))  # both in the no-penalty band


def test_default_risk_threshold_is_half():
    assert Predictor().risk_threshold == 0.5
    assert Predictor(risk_threshold=0.9).risk_threshold == 0.9


def test_confidence_state_table_exact_values():
    """The state->confidence table is a documented contract: exact values."""
    p = Predictor()
    conf = p._calculate_confidence
    assert conf(0.0, "HEALTHY", 0.0, 0.0) == pytest.approx(0.9)
    assert conf(0.0, "DEGRADED", 0.0, 0.0) == pytest.approx(0.7)
    assert conf(0.0, "STRESSED", 0.0, 0.0) == pytest.approx(0.5)
    assert conf(0.0, "CRITICAL", 0.0, 0.0) == pytest.approx(0.3)
    assert conf(0.0, "FAILED", 0.0, 0.0) == pytest.approx(0.1)


def test_predict_markov_state_reaches_risk():
    """markov_state must flow into the risk computation, not be dropped."""
    Predictor()
    critical = _pred(markov_state="CRITICAL")
    healthy = _pred(markov_state="HEALTHY")
    assert critical.risk_score > healthy.risk_score
    assert critical.markov_state == "CRITICAL"


def test_predict_uses_memory_mib_not_just_limit():
    """ttf derives from remaining = limit - memory_mib (dict key contract)."""
    p = Predictor()
    r = p.predict(
        pod_key="ns/pod",
        memory_pct=50.0,
        memory_trend_mib_per_min=1.0,
        memory_limit_mib=50,
        memory_mib=34,
        cpu_pct=30.0,
        restart_rate_per_hr=0.0,
        log_error_rate_per_min=0.0,
        node_memory_pressure=False,
        node_disk_pressure=False,
        markov_state="HEALTHY",
        markov_p_critical=0.0,
        markov_p_failed=0.0,
    )
    assert r.ttf_minutes == 16  # (50 - 34) / 1, truncated


def test_predict_ttf_exact_cap_for_minuscule_trend():
    """Trend below the epsilon guard caps ttf at exactly 43200 minutes."""
    r = _pred(memory_trend_mib_per_min=1e-10)
    assert r.ttf_minutes == 43200


def test_predict_zero_trend_yields_none_not_sentinel_string():
    r = _pred(memory_trend_mib_per_min=0.0)
    assert r.ttf_minutes is None


def test_optional_predict_defaults_match_explicit_falsy():
    """Omitting optional kwargs must equal passing the documented defaults —
    a True/nonzero default mutant changes the risk."""
    p = Predictor()
    common = dict(
        pod_key="ns/pod",
        memory_pct=50.0,
        memory_trend_mib_per_min=0.0,
        memory_limit_mib=1024,
        memory_mib=512,
        cpu_pct=30.0,
        restart_rate_per_hr=0.0,
        log_error_rate_per_min=0.0,
        node_memory_pressure=False,
        node_disk_pressure=False,
        markov_state="HEALTHY",
        markov_p_critical=0.0,
        markov_p_failed=0.0,
    )
    omitted = p.predict(**common)
    explicit = p.predict(
        **common,
        cpu_trend_m_per_min=0.0,
        memory_anomaly_score=0.0,
        cpu_anomaly_score=0.0,
        blast_radius=0,
        terminated=False,
    )
    assert omitted.risk_score == pytest.approx(explicit.risk_score)
    assert omitted.ttf_minutes == explicit.ttf_minutes
    assert omitted.confidence == pytest.approx(explicit.confidence)


def test_memory_mib_flips_risk_ttf_tier():
    """The metrics dict must carry memory_mib under its contract name:
    remaining at the ttf<5 tier vs no-mib remaining at the ttf<60 tier."""
    a = _pred(memory_limit_mib=50, memory_mib=48, memory_trend_mib_per_min=1.0)
    b = _pred(memory_limit_mib=50, memory_mib=0, memory_trend_mib_per_min=1.0)
    assert a.ttf_minutes == 2 and b.ttf_minutes == 50
    assert a.risk_score > b.risk_score


def test_predict_sub_epsilon_trend_yields_none_sentinel():
    """Trend in (0, 1e-10) takes the None sentinel branch, not a value."""
    r = _pred(memory_trend_mib_per_min=1e-11)
    assert r.ttf_minutes is None
