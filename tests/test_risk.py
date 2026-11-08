"""Test Bayesian risk scoring."""
from predictive_agent.config import PREDICTION_BASE_RISK
from predictive_agent.risk import calculate_risk


def test_risk_calculation_low_risk():
    """Test low risk calculation."""
    pod_metrics = {
        "memory_pct": 50.0,
        "memory_trend_mib_per_min": 0.5,
        "cpu_pct": 30.0,
        "restart_rate_per_hr": 0.0,
        "log_error_rate_per_min": 0.0,
        "node_memory_pressure": False,
        "node_disk_pressure": False,
    }

    risk = calculate_risk(pod_metrics, "HEALTHY", 0.01, 0.001)
    assert 0 <= risk <= 0.2  # Low risk (higher with new prior of 0.15)


def test_risk_calculation_high_risk():
    """Test high risk calculation."""
    pod_metrics = {
        "memory_pct": 95.0,
        "memory_trend_mib_per_min": 5.0,
        "cpu_pct": 90.0,
        "restart_rate_per_hr": 5.0,
        "log_error_rate_per_min": 10.0,
        "node_memory_pressure": True,
        "node_disk_pressure": False,
    }

    risk = calculate_risk(pod_metrics, "CRITICAL", 0.5, 0.2)
    assert risk > 0.8  # High risk


def test_risk_calculation_with_memory_ttf():
    """Test risk calculation with memory time-to-failure."""
    pod_metrics = {
        "memory_pct": 85.0,
        "memory_trend_mib_per_min": 2.0,
        "memory_limit_mib": 1024,
        "memory_mib": 870,
        "cpu_pct": 40.0,
        "restart_rate_per_hr": 1.0,
        "log_error_rate_per_min": 2.0,
        "node_memory_pressure": False,
        "node_disk_pressure": False,
    }

    risk = calculate_risk(pod_metrics, "STRESSED", 0.1, 0.05)
    assert risk > 0.1  # Medium risk due to TTF


def test_risk_calculation_no_metrics():
    """Test risk with empty metrics."""
    risk = calculate_risk({}, "HEALTHY", 0.0, 0.0)
    assert 0.14 <= risk <= 0.16  # Should be near prior of 0.15


def test_risk_calculation_boundary():
    """Test risk is bounded 0-1."""
    pod_metrics = {
        "memory_pct": 100.0,
        "memory_trend_mib_per_min": 100.0,
        "cpu_pct": 100.0,
        "restart_rate_per_hr": 100.0,
        "log_error_rate_per_min": 100.0,
        "node_memory_pressure": True,
        "node_disk_pressure": True,
    }
    risk = calculate_risk(pod_metrics, "FAILED", 1.0, 1.0)
    assert risk <= 0.99
    assert risk >= 0.0


def test_risk_calculation_markov_impact():
    """Test that Markov state affects risk."""
    base_metrics = {
        "memory_pct": 80.0,
        "cpu_pct": 50.0,
        "restart_rate_per_hr": 1.0,
        "log_error_rate_per_min": 1.0,
    }
    risk_healthy = calculate_risk(base_metrics, "HEALTHY", 0.0, 0.0)
    risk_critical = calculate_risk(base_metrics, "CRITICAL", 0.3, 0.1)
    assert risk_critical > risk_healthy


def test_risk_single_restart_increases_risk():
    """A single restart (rate=1) should increase risk via the >0 threshold."""
    base = {
        "memory_pct": 50.0,
        "cpu_pct": 30.0,
        "restart_rate_per_hr": 0.0,
        "log_error_rate_per_min": 0.0,
    }
    no_restart = calculate_risk(base, "HEALTHY", 0.0, 0.0)
    one_restart = calculate_risk(
        {**base, "restart_rate_per_hr": 1.0}, "HEALTHY", 0.0, 0.0
    )
    assert one_restart > no_restart


def test_risk_log_errors_any_increase():
    """Any log errors should increase risk (>0 threshold)."""
    base = {
        "memory_pct": 50.0,
        "cpu_pct": 30.0,
        "restart_rate_per_hr": 0.0,
        "log_error_rate_per_min": 0.0,
    }
    no_errors = calculate_risk(base, "HEALTHY", 0.0, 0.0)
    some_errors = calculate_risk(
        {**base, "log_error_rate_per_min": 1.0}, "HEALTHY", 0.0, 0.0
    )
    assert some_errors > no_errors


def test_risk_prior_is_0_15():
    """Base rate (no signals) should produce risk around 0.15."""
    risk = calculate_risk({}, "HEALTHY", 0.0, 0.0)
    assert 0.14 <= risk <= 0.16


def test_risk_stressed_with_restart_crosses_threshold():
    """High memory + restart + STRESSED state should cross 0.5 threshold."""
    metrics = {
        "memory_pct": 90.0,
        "cpu_pct": 50.0,
        "restart_rate_per_hr": 2.0,
        "log_error_rate_per_min": 0.0,
    }
    risk = calculate_risk(metrics, "STRESSED", 0.1, 0.0)
    assert risk > 0.5


def test_risk_crash_loop_back_off():
    """CrashLoopBackOff should significantly increase risk."""
    base = {
        "memory_pct": 50.0,
        "cpu_pct": 30.0,
        "restart_rate_per_hr": 0.0,
        "log_error_rate_per_min": 0.0,
        "pod_phase": "Running",
        "container_ready": False,
        "wait_state": "CrashLoopBackOff",
    }
    risk = calculate_risk(base, "HEALTHY", 0.0, 0.0)
 # Without CrashLoopBackOff
    base_no_crash = {**base, "wait_state": None, "container_ready": True}
    risk_no_crash = calculate_risk(base_no_crash, "HEALTHY", 0.0, 0.0)
    assert risk > risk_no_crash
    assert risk > 0.5  # Should be high risk


def test_risk_create_container_config_error():
    """CreateContainerConfigError should significantly increase risk."""
    metrics = {
        "memory_pct": 50.0,
        "cpu_pct": 30.0,
        "restart_rate_per_hr": 0.0,
        "log_error_rate_per_min": 0.0,
        "pod_phase": "Pending",
        "container_ready": False,
        "wait_state": "CreateContainerConfigError",
    }
    risk = calculate_risk(metrics, "HEALTHY", 0.0, 0.0)
    assert risk > 0.5


def test_risk_image_pull_back_off():
    """ImagePullBackOff should increase risk."""
    metrics = {
        "memory_pct": 50.0,
        "cpu_pct": 30.0,
        "restart_rate_per_hr": 0.0,
        "log_error_rate_per_min": 0.0,
        "pod_phase": "Pending",
        "container_ready": False,
        "wait_state": "ImagePullBackOff",
    }
    risk = calculate_risk(metrics, "HEALTHY", 0.0, 0.0)
    assert risk > 0.3


def test_risk_pending_phase():
    """Pending phase should increase risk."""
    base = {
        "memory_pct": 50.0,
        "cpu_pct": 30.0,
        "restart_rate_per_hr": 0.0,
        "log_error_rate_per_min": 0.0,
        "pod_phase": "Running",
    }
    risk_running = calculate_risk(base, "HEALTHY", 0.0, 0.0)
    base_pending = {**base, "pod_phase": "Pending"}
    risk_pending = calculate_risk(base_pending, "HEALTHY", 0.0, 0.0)
    assert risk_pending > risk_running


def test_risk_failed_phase():
    """Failed phase should produce very high risk."""
    metrics = {
        "memory_pct": 50.0,
        "cpu_pct": 30.0,
        "restart_rate_per_hr": 0.0,
        "log_error_rate_per_min": 0.0,
        "pod_phase": "Failed",
        "container_ready": False,
        "terminated": True,
        "terminated_reason": "Error",
    }
    risk = calculate_risk(metrics, "HEALTHY", 0.0, 0.0)
    assert risk > 0.8


def test_risk_succeeded_phase():
    """Succeeded phase should produce very low risk."""
    metrics = {
        "memory_pct": 50.0,
        "cpu_pct": 30.0,
        "restart_rate_per_hr": 0.0,
        "log_error_rate_per_min": 0.0,
        "pod_phase": "Succeeded",
        "terminated": True,
        "terminated_reason": "Completed",
    }
    risk = calculate_risk(metrics, "HEALTHY", 0.0, 0.0)
    assert risk < 0.05


def test_risk_oom_killed():
    """OOMKilled should significantly increase risk."""
    base = {
        "memory_pct": 50.0,
        "cpu_pct": 30.0,
        "restart_rate_per_hr": 0.0,
        "log_error_rate_per_min": 0.0,
        "pod_phase": "Running",
    }
    risk_base = calculate_risk(base, "HEALTHY", 0.0, 0.0)
    base_oom = {**base, "terminated": True, "terminated_reason": "OOMKilled"}
    risk_oom = calculate_risk(base_oom, "HEALTHY", 0.0, 0.0)
    assert risk_oom > risk_base
    assert risk_oom > 0.5


def test_risk_not_scheduled():
    """Unscheduled pod should have increased risk."""
    base = {
        "memory_pct": 50.0,
        "cpu_pct": 30.0,
        "restart_rate_per_hr": 0.0,
        "log_error_rate_per_min": 0.0,
        "pod_phase": "Running",
        "pod_scheduled": True,
    }
    risk_scheduled = calculate_risk(base, "HEALTHY", 0.0, 0.0)
    base_unscheduled = {**base, "pod_scheduled": False, "pod_phase": "Pending"}
    risk_unscheduled = calculate_risk(base_unscheduled, "HEALTHY", 0.0, 0.0)
    assert risk_unscheduled > risk_scheduled


def test_risk_container_not_ready():
    """Container not ready should increase risk."""
    base = {
        "memory_pct": 50.0,
        "cpu_pct": 30.0,
        "restart_rate_per_hr": 0.0,
        "log_error_rate_per_min": 0.0,
        "pod_phase": "Running",
        "container_ready": True,
    }
    risk_ready = calculate_risk(base, "HEALTHY", 0.0, 0.0)
    base_not_ready = {**base, "container_ready": False}
    risk_not_ready = calculate_risk(base_not_ready, "HEALTHY", 0.0, 0.0)
    assert risk_not_ready > risk_ready


def _ttf_risk(limit_mib, mem_mib, trend_mib_min, pct):
    """Helper: compute risk for a given memory/TTF scenario."""
    return calculate_risk(
        {
            "memory_pct": pct,
            "memory_trend_mib_per_min": trend_mib_min,
            "memory_limit_mib": limit_mib,
            "memory_mib": mem_mib,
            "cpu_pct": 30.0,
            "restart_rate_per_hr": 0.0,
            "log_error_rate_per_min": 0.0,
            "node_memory_pressure": False,
            "node_disk_pressure": False,
        },
        "HEALTHY", 0.0, 0.0,
    )


def test_risk_ttf_at_low_memory():
    """Regression: rapid memory growth must raise risk even when current
    memory % is low (old code only escalated TTF when mem_pct > 70)."""
    # Low memory % but very fast growth → OOM in ~24 min
    risk = _ttf_risk(limit_mib=1024, mem_mib=300, trend_mib_min=30.0, pct=29.0)
    baseline = _ttf_risk(limit_mib=1024, mem_mib=300, trend_mib_min=0.0, pct=29.0)
    assert risk > baseline
    assert risk > 0.2


def test_risk_ttf_severe_at_low_memory():
    """TTF under 5 min must escalate hard even below 70% memory."""
    risk = _ttf_risk(limit_mib=1024, mem_mib=900, trend_mib_min=30.0, pct=87.9)
    assert risk > 0.8


def test_risk_ttf_far_away_not_overescalated():
    """A trend that still has hours to OOM should not lift risk."""
    risk = _ttf_risk(limit_mib=2048, mem_mib=76, trend_mib_min=5.43, pct=3.7)
    assert 0.10 <= risk <= 0.30


# ─── Pod phase / container status / anomaly signal branches ────────────
# These multipliers are the strongest signals in the scorer (Failed ×20,
# CrashLoopBackOff ×15, OOMKilled ×12) — each branch must move the score
# in the right direction relative to a neutral-pod baseline.


def _metrics(**over):
    m = {
        "memory_pct": 50.0,
        "memory_trend_mib_per_min": 0.0,
        "cpu_pct": 30.0,
        "restart_rate_per_hr": 0.0,
        "log_error_rate_per_min": 0.0,
        "node_memory_pressure": False,
        "node_disk_pressure": False,
    }
    m.update(over)
    return m


def _baseline():
    return calculate_risk(_metrics(), "HEALTHY", 0.01, 0.001)


def test_pod_phase_failed_escalates_hard():
    risk = calculate_risk(_metrics(pod_phase="Failed"), "HEALTHY", 0.01, 0.001)
    assert risk > 0.7  # ×20 likelihood on the base rate


def test_pod_phase_ranking():
    unknown = calculate_risk(_metrics(pod_phase="Unknown"), "HEALTHY", 0.01, 0.001)
    pending = calculate_risk(_metrics(pod_phase="Pending"), "HEALTHY", 0.01, 0.001)
    succeeded = calculate_risk(_metrics(pod_phase="Succeeded"), "HEALTHY", 0.01, 0.001)
    assert unknown > pending > _baseline()
    assert succeeded < _baseline()  # completed pods damp risk


def test_not_scheduled_and_not_ready_escalate():
    unsched = calculate_risk(_metrics(pod_scheduled=False), "HEALTHY", 0.01, 0.001)
    not_ready = calculate_risk(_metrics(container_ready=False), "HEALTHY", 0.01, 0.001)
    assert unsched > _baseline()
    assert not_ready > _baseline()


def test_wait_state_ranking():
    crashloop = calculate_risk(_metrics(wait_state="CrashLoopBackOff"), "HEALTHY", 0.01, 0.001)
    image_pull = calculate_risk(_metrics(wait_state="ImagePullBackOff"), "HEALTHY", 0.01, 0.001)
    creating = calculate_risk(_metrics(wait_state="ContainerCreating"), "HEALTHY", 0.01, 0.001)
    unknown_wait = calculate_risk(_metrics(wait_state="SomeNewWaitReason"), "HEALTHY", 0.01, 0.001)
    assert crashloop > image_pull > creating
    assert unknown_wait > _baseline()  # unknown wait reasons still escalate
    assert creating > _baseline()  # even 'normal startup' state nudges risk


def test_termination_signals():
    oom = calculate_risk(_metrics(terminated=True, terminated_reason="OOMKilled"), "HEALTHY", 0.01, 0.001)
    error = calculate_risk(_metrics(terminated=True, terminated_reason="Error"), "HEALTHY", 0.01, 0.001)
    other = calculate_risk(_metrics(terminated=True, terminated_reason="Evicted"), "HEALTHY", 0.01, 0.001)
    completed = calculate_risk(_metrics(terminated=True, terminated_reason="Completed"), "HEALTHY", 0.01, 0.001)
    assert oom > error > other > _baseline()
    assert completed < _baseline()  # normal completion damps risk


def test_anomaly_score_multipliers():
    for key, mild, severe in (
        ("memory_anomaly_score", 2.5, 3.5),
        ("cpu_anomaly_score", 2.5, 3.5),
    ):
        base = calculate_risk(_metrics(), "HEALTHY", 0.01, 0.001)
        mild_risk = calculate_risk(_metrics(**{key: mild}), "HEALTHY", 0.01, 0.001)
        severe_risk = calculate_risk(_metrics(**{key: severe}), "HEALTHY", 0.01, 0.001)
        assert severe_risk > mild_risk > base, key


def test_wait_state_run_container_error_and_pod_initializing():
    """RunContainerError ×10 vs PodInitializing ×2 — distinct severity."""
    run_err = calculate_risk(_metrics(wait_state="RunContainerError"), "HEALTHY", 0.01, 0.001)
    init = calculate_risk(_metrics(wait_state="PodInitializing"), "HEALTHY", 0.01, 0.001)
    assert run_err > init > _baseline()


def test_blast_radius_moderate_tier():
    """KG blast radius > 5 (but below the heavy tiers) still escalates."""
    moderate = calculate_risk(_metrics(blast_radius=6), "HEALTHY", 0.01, 0.001)
    high = calculate_risk(_metrics(blast_radius=25), "HEALTHY", 0.01, 0.001)
    assert moderate > _baseline()
    assert high > moderate


def test_each_core_signal_strictly_monotonic():
    """Every scalar signal must raise risk when escalated — kills any
    mutant that drops a metric lookup (key typo / default swap)."""
    base = calculate_risk(_metrics(), "HEALTHY", 0.01, 0.001)
    for over in (
        {"memory_pct": 95.0},
        {"cpu_pct": 92.0},
        {"restart_rate_per_hr": 6.0},
        {"log_error_rate_per_min": 12.0},
        {"node_memory_pressure": True},
        {"node_disk_pressure": True},
        {"memory_trend_mib_per_min": 30.0, "memory_limit_mib": 1024, "memory_mib": 1010},
        {"cpu_trend_m_per_min": 500.0},
    ):
        escalated = calculate_risk(_metrics(**over), "HEALTHY", 0.01, 0.001)
        assert escalated > base, f"signal not escalated: {over}"


# ═══════════════════════════════════════════════════════════════════════════
# Mutation-survivor killers: tier boundaries, signal-key contract, LR
# multiplication semantics, guard interactions. Ordering/boundary asserts
# only — calibration constants are deliberately NOT pinned.
# ═══════════════════════════════════════════════════════════════════════════

import pytest


def _risk(**over):
    return calculate_risk(_metrics(**over), "HEALTHY", 0.0, 0.0)


# (kwargs, tier-below, threshold, tier-above): risk(below) == risk(at) <
# risk(above) — kills `>` -> `>=` boundary flips and elif->if leakage.
_TIER_BOUNDARIES = [
    ({"memory_pct": 70.0}, 69.0, 70.0, 71.0),
    ({"memory_pct": 85.0}, 84.0, 85.0, 86.0),
    ({"memory_pct": 95.0}, 94.0, 95.0, 96.0),
    ({"cpu_pct": 80.0}, 79.0, 80.0, 81.0),
    ({"cpu_pct": 95.0}, 94.0, 95.0, 96.0),
    ({"cpu_trend_m_per_min": 20.0}, 19.0, 20.0, 21.0),
    ({"cpu_trend_m_per_min": 50.0}, 49.0, 50.0, 51.0),
    ({"restart_rate_per_hr": 1.0}, 0.5, 1.0, 1.5),
    ({"restart_rate_per_hr": 3.0}, 2.5, 3.0, 3.5),
    ({"restart_rate_per_hr": 5.0}, 4.5, 5.0, 5.5),
    ({"log_error_rate_per_min": 5.0}, 4.5, 5.0, 5.5),
    ({"log_error_rate_per_min": 10.0}, 9.5, 10.0, 10.5),
    ({"memory_anomaly_score": 2.0}, 1.9, 2.0, 2.1),
    ({"memory_anomaly_score": 3.0}, 2.9, 3.0, 3.1),
    ({"cpu_anomaly_score": 2.0}, 1.9, 2.0, 2.1),
    ({"cpu_anomaly_score": 3.0}, 2.9, 3.0, 3.1),
    ({"blast_radius": 5}, 4, 5, 6),
    ({"blast_radius": 10}, 9, 10, 11),
    ({"blast_radius": 20}, 19, 20, 21),
]


@pytest.mark.parametrize("base_cfg,below,at,above", _TIER_BOUNDARIES)
def test_tier_boundary_semantics(base_cfg, below, at, above):
    """risk(threshold) must equal risk(just-below) and be < risk(just-above)."""
    key = next(iter(base_cfg))
    b = _risk(**{**base_cfg, key: below})
    a = _risk(**base_cfg)
    c = _risk(**{**base_cfg, key: above})
    assert a == pytest.approx(b), f"boundary leak: {base_cfg}"
    assert c > a, f"tier not crossed: {base_cfg}"


@pytest.mark.parametrize(
    "cfg",
    [
        {"ttf": 5.0, "mem_pct": 50.0},
        {"ttf": 10.0, "mem_pct": 50.0},
        {"ttf": 30.0, "mem_pct": 50.0},
        {"ttf": 60.0, "mem_pct": 50.0},
    ],
)
def test_ttf_boundary_semantics(cfg):
    """TTF tiers worsen as ttf shrinks: at-threshold equals just-above;
    just-below crosses into the stronger tier."""
    limit = 600.0  # exact integer ttf targets: 600/ttf is exact for all rows

    def r(ttf, pct):
        return _risk(
            memory_pct=pct,
            memory_trend_mib_per_min=limit / ttf,
            memory_limit_mib=limit,
            memory_mib=0,
        )

    at, below, above = cfg["ttf"], cfg["ttf"] - 0.5, cfg["ttf"] + 0.5
    assert r(at, cfg["mem_pct"]) == pytest.approx(r(above, cfg["mem_pct"])), cfg
    assert r(below, cfg["mem_pct"]) > r(at, cfg["mem_pct"]), cfg


# Signal-key contract: every pod_metrics key must influence the score —
# kills XXkeyXX / UPPER-KEY dict-key mutants inside calculate_risk.
_KEY_SWEEP = [
    {"memory_pct": 96.0},
    {"memory_pct": 80.0},
    {"memory_pct": 75.0},
    {"memory_trend_mib_per_min": 100.0, "memory_limit_mib": 1000, "memory_mib": 0},
    {"cpu_pct": 96.0},
    {"cpu_pct": 85.0},
    {"cpu_trend_m_per_min": 60.0},
    {"cpu_trend_m_per_min": 25.0},
    {"restart_rate_per_hr": 6.0},
    {"restart_rate_per_hr": 4.0},
    {"restart_rate_per_hr": 2.0},
    {"restart_rate_per_hr": 0.5},
    {"log_error_rate_per_min": 11.0},
    {"log_error_rate_per_min": 6.0},
    {"log_error_rate_per_min": 1.0},
    {"node_memory_pressure": True},
    {"node_disk_pressure": True},
    {"memory_anomaly_score": 3.5},
    {"memory_anomaly_score": 2.5},
    {"cpu_anomaly_score": 3.5},
    {"cpu_anomaly_score": 2.5},
    {"blast_radius": 25},
    {"blast_radius": 15},
    {"blast_radius": 7},
    {"pod_phase": "Failed"},
    {"pod_phase": "Pending"},
    {"pod_phase": "Unknown"},
    {"container_ready": False},
    {"wait_state": "CrashLoopBackOff"},
    {"wait_state": "ImagePullBackOff"},
    {"wait_state": "RunContainerError"},
    {"wait_state": "CreateContainerConfigError"},
    {"wait_state": "PodInitializing"},
    {"terminated": True, "terminated_reason": "OOMKilled"},
    {"terminated": True, "terminated_reason": "Error"},
    {"pod_scheduled": False},
]


@pytest.mark.parametrize("cfg", _KEY_SWEEP)
def test_signal_key_contract(cfg):
    """Setting each signal must raise risk above the all-quiet baseline."""
    assert _risk(**cfg) > _risk(), f"signal ignored: {cfg}"


def test_completed_termination_lowers_risk():
    """Completed termination de-escalates; bare terminated flag alone must not."""
    assert _risk(terminated=True, terminated_reason="Completed") < _risk()
    assert _risk(terminated=True, terminated_reason="Other") > _risk()
    assert _risk(terminated=False, terminated_reason="OOMKilled") == pytest.approx(_risk())


def test_wait_state_naming_contract():
    """Named wait states escalate in fixed order above the generic-else tier;
    the pre-start tier (ContainerCreating/PodInitializing) sits below it."""
    generic = _risk(wait_state="MysteryState")
    assert generic > _risk()
    for state in ("CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull",
                  "InvalidImageName", "RunContainerError",
                  "CreateContainerConfigError", "CreateContainerError"):
        assert _risk(wait_state=state) > generic, state
    for state in ("ContainerCreating", "PodInitializing"):
        assert _risk(wait_state=state) < generic, state
        assert _risk(wait_state=state) > _risk(), state
    assert _risk(wait_state="CrashLoopBackOff") > _risk(wait_state="ImagePullBackOff")
    assert _risk(wait_state=None) == pytest.approx(_risk())


def test_pod_phase_ordering_contract():
    """Failed > Unknown > Pending > Running == baseline; Succeeded below."""
    running = _risk(pod_phase="Running")
    assert _risk(pod_phase="Failed") > _risk(pod_phase="Unknown") > _risk(pod_phase="Pending") > running
    assert running == pytest.approx(_risk())
    assert _risk(pod_phase="Succeeded") < running


def test_markov_probability_factor_monotone():
    """Both Markov transition probabilities raise risk; jointly more than either."""
    base = calculate_risk(_metrics(), "HEALTHY", 0.0, 0.0)
    pc = calculate_risk(_metrics(), "HEALTHY", 0.2, 0.0)
    pf = calculate_risk(_metrics(), "HEALTHY", 0.0, 0.2)
    both = calculate_risk(_metrics(), "HEALTHY", 0.2, 0.2)
    assert pc > base and pf > base
    assert both > pc and both > pf


def test_lr_is_multiplicative_not_overwritten():
    """Combining a memory signal with a target signal must exceed the target
    alone — kills `lr = X` overwrites that discard earlier factors."""
    anchor = {"memory_pct": 96.0}  # fires first ladder rung
    anchors = [
        {"memory_pct": 96.0},
        {"restart_rate_per_hr": 6.0},
        {"pod_phase": "Failed"},
    ]
    targets = [
        {"memory_trend_mib_per_min": 100.0, "memory_limit_mib": 1000, "memory_mib": 0},
        {"cpu_pct": 96.0},
        {"cpu_trend_m_per_min": 60.0},
        {"log_error_rate_per_min": 11.0},
        {"node_memory_pressure": True},
        {"memory_anomaly_score": 3.5},
        {"blast_radius": 25},
        {"wait_state": "CrashLoopBackOff"},
        {"terminated": True, "terminated_reason": "OOMKilled"},
        {"pod_scheduled": False},
    ]
    for anchor in anchors:
        for target in targets:
            combo = _risk(**anchor, **target)
            solo = _risk(**target)
            assert combo > solo, f"anchor lost: {anchor} + {target}"


def test_ttf_guards_require_all_conditions():
    """Each TTF guard is a real AND: violating one side must land on baseline."""
    limit, trend = 1000, 30.0  # ttf ~33min otherwise
    assert _risk(memory_trend_mib_per_min=trend, memory_limit_mib=0, memory_mib=0) == pytest.approx(_risk())
    assert _risk(memory_trend_mib_per_min=0.0, memory_limit_mib=limit, memory_mib=0) == pytest.approx(_risk())
    # remaining == 1 (mem one MiB under limit): ttf = 1/trend — a true ttf,
    # not the at-limit ttf = 0 path; both trends land in the same <5 tier.
    rem1_fast = _risk(memory_pct=50.0, memory_trend_mib_per_min=10.0,
                      memory_limit_mib=1000, memory_mib=999)
    rem1_slow = _risk(memory_pct=50.0, memory_trend_mib_per_min=100.0,
                      memory_limit_mib=1000, memory_mib=999)
    assert rem1_fast == pytest.approx(rem1_slow)
    assert rem1_fast > _risk()
    # memory over limit: remaining < 0 -> at-limit ttf = 0 tier (x1.5),
    # NOT the x8 ttf < 5 tier that a remaining>=0 mutant would produce.
    over = _risk(memory_pct=50.0, memory_trend_mib_per_min=10.0,
                 memory_limit_mib=1000, memory_mib=1200)
    assert over < rem1_fast
    # ttf < 60 tier also demands memory_pct > 30
    below = _risk(memory_pct=20.0, memory_trend_mib_per_min=17.0, memory_limit_mib=1000, memory_mib=0)
    assert below == pytest.approx(_risk())  # ttf ~35 but pct 20
    crossed = _risk(memory_pct=31.0, memory_trend_mib_per_min=17.0, memory_limit_mib=1000, memory_mib=0)
    assert crossed > below


def test_posterior_clamped_at_99():
    """The posterior never exceeds 0.99 regardless of evidence strength."""
    extreme = _risk(
        memory_pct=96.0,
        restart_rate_per_hr=10.0,
        log_error_rate_per_min=20.0,
        node_memory_pressure=True,
        node_disk_pressure=True,
        wait_state="CrashLoopBackOff",
        pod_phase="Failed",
        terminated=True,
        terminated_reason="OOMKilled",
        blast_radius=30,
    )
    assert extreme == pytest.approx(0.99)


def _risk_state(state, **over):
    return calculate_risk(_metrics(**over), state, 0.0, 0.0)


# Every ladder rung AFTER the memory group must stack multiplicatively with
# an earlier factor: kills `lr = X` resets (a reset discards the memory
# factor, making combo == solo). Memory-group elifs fire alone and first,
# so their resets are equivalent — excluded by design.
_RUNGS = [
    ({"memory_trend_mib_per_min": 150.0, "memory_limit_mib": 512,
      "memory_mib": 0}, "HEALTHY"),  # ttf ~3.4 < 5
    ({"memory_trend_mib_per_min": 60.0, "memory_limit_mib": 512,
      "memory_mib": 0}, "HEALTHY"),  # ttf ~8.5 < 10
    ({"memory_trend_mib_per_min": 25.0, "memory_limit_mib": 512,
      "memory_mib": 0}, "HEALTHY"),  # ttf ~20.5 < 30
    ({"memory_trend_mib_per_min": 15.0, "memory_limit_mib": 512,
      "memory_mib": 0}, "HEALTHY"),  # ttf ~34 < 60, pct 50
    ({"cpu_pct": 96.0}, "HEALTHY"),
    ({"cpu_pct": 85.0}, "HEALTHY"),
    ({"cpu_trend_m_per_min": 60.0}, "HEALTHY"),
    ({"cpu_trend_m_per_min": 25.0}, "HEALTHY"),
    ({"restart_rate_per_hr": 6.0}, "HEALTHY"),
    ({"restart_rate_per_hr": 4.0}, "HEALTHY"),
    ({"restart_rate_per_hr": 2.0}, "HEALTHY"),
    ({"restart_rate_per_hr": 0.5}, "HEALTHY"),
    ({"log_error_rate_per_min": 11.0}, "HEALTHY"),
    ({"log_error_rate_per_min": 6.0}, "HEALTHY"),
    ({"log_error_rate_per_min": 1.0}, "HEALTHY"),
    ({"node_memory_pressure": True}, "HEALTHY"),
    ({"node_disk_pressure": True}, "HEALTHY"),
    ({}, "CRITICAL"),
    ({}, "STRESSED"),
    ({}, "DEGRADED"),
    ({"memory_anomaly_score": 3.5}, "HEALTHY"),
    ({"memory_anomaly_score": 2.5}, "HEALTHY"),
    ({"cpu_anomaly_score": 3.5}, "HEALTHY"),
    ({"cpu_anomaly_score": 2.5}, "HEALTHY"),
    ({"blast_radius": 25}, "HEALTHY"),
    ({"blast_radius": 15}, "HEALTHY"),
    ({"blast_radius": 7}, "HEALTHY"),
    ({"pod_phase": "Failed"}, "HEALTHY"),
    ({"pod_phase": "Unknown"}, "HEALTHY"),
    ({"pod_phase": "Pending"}, "HEALTHY"),
    ({"pod_phase": "Succeeded"}, "HEALTHY"),
    ({"pod_scheduled": False}, "HEALTHY"),
    ({"container_ready": False}, "HEALTHY"),
    ({"wait_state": "CrashLoopBackOff"}, "HEALTHY"),
    ({"wait_state": "ImagePullBackOff"}, "HEALTHY"),
    ({"wait_state": "CreateContainerConfigError"}, "HEALTHY"),
    ({"wait_state": "RunContainerError"}, "HEALTHY"),
    ({"wait_state": "MysteryState"}, "HEALTHY"),
    ({"wait_state": "ContainerCreating"}, "HEALTHY"),
    ({"terminated": True, "terminated_reason": "OOMKilled"}, "HEALTHY"),
    ({"terminated": True, "terminated_reason": "Failed"}, "HEALTHY"),
    ({"terminated": True, "terminated_reason": "Evicted"}, "HEALTHY"),
    ({"terminated": True, "terminated_reason": "Completed"}, "HEALTHY"),
    ({"terminated": True, "terminated_reason": None}, "HEALTHY"),
]


@pytest.mark.parametrize("cfg,state", _RUNGS)
def test_ladder_rung_stacks_with_memory_factor(cfg, state):
    """combo = memory anchor + rung must exceed rung alone."""
    anchor = {"memory_pct": 96.0}
    combo = _risk_state(state, **anchor, **cfg)
    solo = _risk_state(state, **cfg)
    assert combo > solo, f"memory factor lost for {cfg} / {state}"
    if state != "HEALTHY":
        assert solo > _risk_state("HEALTHY", **cfg), \
            f"state {state} did not escalate {cfg}"


def test_ttf_guard_edge_values():
    """Guard comparisons hold at their exact boundaries (no crash, no tier)."""
    # trend == 0 with positive limit: guard must skip (a >= 0 mutant divides by zero)
    assert _risk(memory_trend_mib_per_min=0.0, memory_limit_mib=512, memory_mib=0) == pytest.approx(_risk())
    # trend == 1.0: not above 1, guard must still pass -> tier fires
    assert _risk(memory_pct=50.0, memory_trend_mib_per_min=1.0, memory_limit_mib=59, memory_mib=0) > _risk()
    # trend 0.5 crosses no >1 mutant guard -> tier fires
    assert _risk(memory_pct=50.0, memory_trend_mib_per_min=0.5, memory_limit_mib=10, memory_mib=0) > _risk()
    # limit == 1: guard is limit > 0 -> ttf 1, tier fires
    assert _risk(memory_pct=50.0, memory_trend_mib_per_min=1.0, memory_limit_mib=1, memory_mib=0) > _risk()
    # remaining == 0 (mem exactly at limit): no ttf tier fires
    assert _risk(memory_pct=50.0, memory_trend_mib_per_min=5.0, memory_limit_mib=512, memory_mib=512) == pytest.approx(_risk())


def test_missing_metrics_keys_use_contract_defaults():
    """Sparse dicts must not crash and must not escalate (default 1/True flips)."""
    base = calculate_risk(
        {"memory_pct": 50.0, "cpu_pct": 30.0}, "HEALTHY", 0.0, 0.0)
    # trend missing: default 0 must fail the trend guard (default 1 would pass)
    assert calculate_risk(
        {"memory_pct": 50.0, "memory_limit_mib": 59}, "HEALTHY", 0.0, 0.0) == pytest.approx(base)
    # memory_mib missing: remaining = limit - 0, ttf 60 -> no tier; default 1
    # flips remaining to 59 -> x1.5 tier; default None crashes.
    assert calculate_risk(
        {"memory_pct": 50.0, "memory_trend_mib_per_min": 1.0,
         "memory_limit_mib": 60}, "HEALTHY", 0.0, 0.0) == pytest.approx(base)
    # terminated missing: default False must not fire the termination ladder
    # (default True + reason None would run the x2 startup tier)
    assert base == pytest.approx(PREDICTION_BASE_RISK)


def test_baseline_equals_prior_when_quiet():
    """Value anchor: a quiet pod sits exactly at the configured prior."""
    assert _risk() == pytest.approx(PREDICTION_BASE_RISK)


def test_ttf_tier_memory_pct_boundary_is_strict():
    """mem_pct == 30 is NOT above 30: the ttf<60 tier must not fire."""
    at = calculate_risk(_metrics(memory_pct=30.0, memory_trend_mib_per_min=15.0,
                                 memory_limit_mib=512, memory_mib=0),
                        "HEALTHY", 0.0, 0.0)
    below = calculate_risk(_metrics(memory_pct=29.0, memory_trend_mib_per_min=15.0,
                                    memory_limit_mib=512, memory_mib=0),
                           "HEALTHY", 0.0, 0.0)
    assert at == pytest.approx(below)


def test_markov_probability_amplification_contract():
    """p_critical carries weight x5 and p_failed x10 — a /-flip would
    diminish instead of amplify, and reorder the weights."""
    base = calculate_risk(_metrics(), "HEALTHY", 0.0, 0.0)
    pc = calculate_risk(_metrics(), "HEALTHY", 0.5, 0.0)
    calculate_risk(_metrics(), "HEALTHY", 0.0, 0.2)
    pc_ord = calculate_risk(_metrics(), "HEALTHY", 0.2, 0.0)
    pf_ord = calculate_risk(_metrics(), "HEALTHY", 0.0, 0.2)
    assert pc > 2.0 * base
    assert pf_ord > pc_ord
