"""Golden-table tests for risk.calculate_risk.

Every multiplier, boundary, guard and default in calculate_risk is pinned to
an exact posterior score, independently re-derived through the documented
Bayesian formula::

    posterior = (P * lr) / (P * lr + (1 - P))      P = PREDICTION_BASE_RISK

where ``lr`` is the product of the per-signal multipliers listed in each row.
The expected value is computed from the multiplier *specification*, not from
the implementation, so any multiplier-value, boundary or guard mutation in
the source shifts the float and is killed (this module exists because the
threshold-style assertions in test_risk.py let value mutations through).

Baselining: an empty signal set has lr = 1.0 → 0.15 / (0.15 + 0.85) = 0.15.
"""

import pytest

from predictive_agent.config import PREDICTION_BASE_RISK
from predictive_agent.risk import calculate_risk

P = PREDICTION_BASE_RISK


def _bayes(lr: float) -> float:
    return (P * lr) / (P * lr + (1 - P))


# (label, pod_metrics, markov_state, markov_p_critical, markov_p_failed, multipliers)
ROWS = [
    # ── Baseline / defaults ────────────────────────────────────────────────
    ("empty", {}, None, 0.0, 0.0, [1.0]),
    ("defaults_explicit", {"memory_pct": 0, "cpu_pct": 0, "restart_rate_per_hr": 0,
                           "log_error_rate_per_min": 0}, "HEALTHY", 0.0, 0.0, [1.0]),
    # ── Memory percent branches + boundaries ───────────────────────────────
    ("mem_96", {"memory_pct": 96}, None, 0.0, 0.0, [12.0]),
    ("mem_90", {"memory_pct": 90}, None, 0.0, 0.0, [6.0]),
    ("mem_75", {"memory_pct": 75}, None, 0.0, 0.0, [2.5]),
    ("mem_60_no_branch", {"memory_pct": 60}, None, 0.0, 0.0, [1.0]),
    ("mem_at_95_is_6x", {"memory_pct": 95}, None, 0.0, 0.0, [6.0]),
    ("mem_at_85_is_2_5x", {"memory_pct": 85}, None, 0.0, 0.0, [2.5]),
    ("mem_at_70_not_gt", {"memory_pct": 70}, None, 0.0, 0.0, [1.0]),
    # ── Time-to-failure (requires trend>0, limit>0, remaining>0) ───────────
    # memory_pct kept <=70 so rows isolate the TTF multiplier (no mem branch).
    ("ttf_4min", {"memory_pct": 50, "memory_trend_mib_per_min": 100,
                  "memory_limit_mib": 400, "memory_mib": 0}, None, 0.0, 0.0, [8.0]),
    ("ttf_8min", {"memory_pct": 50, "memory_trend_mib_per_min": 50,
                  "memory_limit_mib": 400, "memory_mib": 0}, None, 0.0, 0.0, [4.0]),
    ("ttf_20min", {"memory_pct": 50, "memory_trend_mib_per_min": 20,
                   "memory_limit_mib": 400, "memory_mib": 0}, None, 0.0, 0.0, [2.0]),
    ("ttf_45min_mem30", {"memory_pct": 40, "memory_trend_mib_per_min": 10,
                         "memory_limit_mib": 450, "memory_mib": 0}, None, 0.0, 0.0, [1.5]),
    ("ttf_45min_low_mem_no_boost", {"memory_pct": 20, "memory_trend_mib_per_min": 10,
                                    "memory_limit_mib": 450, "memory_mib": 0}, None, 0.0, 0.0, [1.0]),
    ("ttf_at_exactly_30", {"memory_pct": 50, "memory_trend_mib_per_min": 10,
                           "memory_limit_mib": 300, "memory_mib": 0}, None, 0.0, 0.0, [1.5]),
    ("ttf_at_exactly_5", {"memory_pct": 50, "memory_trend_mib_per_min": 80,
                          "memory_limit_mib": 400, "memory_mib": 0}, None, 0.0, 0.0, [4.0]),
    ("ttf_zero_trend_no_boost", {"memory_pct": 50, "memory_trend_mib_per_min": 0,
                                 "memory_limit_mib": 400, "memory_mib": 0}, None, 0.0, 0.0, [1.0]),
    ("ttf_no_limit_no_boost", {"memory_pct": 50, "memory_trend_mib_per_min": 100,
                               "memory_limit_mib": 0, "memory_mib": 0}, None, 0.0, 0.0, [1.0]),
    ("ttf_remaining_zero_no_boost", {"memory_pct": 50, "memory_trend_mib_per_min": 100,
                                     "memory_limit_mib": 400, "memory_mib": 400}, None, 0.0, 0.0, [1.0]),
    # ── CPU percent (only when cpu_limit_known) ────────────────────────────
    ("cpu_96", {"cpu_pct": 96}, None, 0.0, 0.0, [3.0]),
    ("cpu_90", {"cpu_pct": 90}, None, 0.0, 0.0, [1.5]),
    ("cpu_80_no_branch", {"cpu_pct": 80}, None, 0.0, 0.0, [1.0]),
    ("cpu_96_limit_unknown", {"cpu_pct": 96, "cpu_limit_known": False},
     None, 0.0, 0.0, [1.0]),
    # ── CPU trend (normalized against limit) ───────────────────────────────
    ("cpu_trend_strong", {"cpu_trend_m_per_min": 120, "cpu_limit_m": 2000},
     None, 0.0, 0.0, [2.0]),
    ("cpu_trend_mild", {"cpu_trend_m_per_min": 60, "cpu_limit_m": 2000},
     None, 0.0, 0.0, [1.3]),
    ("cpu_trend_at_50", {"cpu_trend_m_per_min": 50, "cpu_limit_m": 1000},
     None, 0.0, 0.0, [1.3]),
    ("cpu_trend_low", {"cpu_trend_m_per_min": 10, "cpu_limit_m": 1000},
     None, 0.0, 0.0, [1.0]),
    ("cpu_trend_limit_unknown", {"cpu_trend_m_per_min": 500, "cpu_limit_known": False},
     None, 0.0, 0.0, [1.0]),
    # ── Restart rate branches + floor ──────────────────────────────────────
    ("restart_6", {"restart_rate_per_hr": 6}, None, 0.0, 0.0, [15.0]),
    ("restart_4", {"restart_rate_per_hr": 4}, None, 0.0, 0.0, [10.0]),
    ("restart_2", {"restart_rate_per_hr": 2}, None, 0.0, 0.0, [6.0]),
    ("restart_0_5", {"restart_rate_per_hr": 0.5}, None, 0.0, 0.0, [3.0]),
    ("restart_at_0_1_no_branch", {"restart_rate_per_hr": 0.1}, None, 0.0, 0.0, [1.0]),
    ("restart_0_05_no_branch", {"restart_rate_per_hr": 0.05}, None, 0.0, 0.0, [1.0]),
    ("restart_zero_no_branch", {"restart_rate_per_hr": 0}, None, 0.0, 0.0, [1.0]),
    # ── Log error rate ─────────────────────────────────────────────────────
    ("log_11", {"log_error_rate_per_min": 11}, None, 0.0, 0.0, [4.0]),
    ("log_7", {"log_error_rate_per_min": 7}, None, 0.0, 0.0, [2.5]),
    ("log_1", {"log_error_rate_per_min": 1}, None, 0.0, 0.0, [1.5]),
    ("log_0_no_branch", {"log_error_rate_per_min": 0}, None, 0.0, 0.0, [1.0]),
    # ── Node pressure ──────────────────────────────────────────────────────
    ("node_mem_pressure", {"node_memory_pressure": True}, None, 0.0, 0.0, [4.0]),
    ("node_disk_pressure", {"node_disk_pressure": True}, None, 0.0, 0.0, [6.0]),
    ("node_both_pressures", {"node_memory_pressure": True, "node_disk_pressure": True},
     None, 0.0, 0.0, [24.0]),
    # ── Markov state ───────────────────────────────────────────────────────
    ("markov_critical", {}, "CRITICAL", 0.0, 0.0, [20.0]),
    ("markov_stressed", {}, "STRESSED", 0.0, 0.0, [3.0]),
    ("markov_degraded", {}, "DEGRADED", 0.0, 0.0, [1.5]),
    ("markov_healthy", {}, "HEALTHY", 0.0, 0.0, [1.0]),
    # ── Markov probabilities (always applied factor 1 + pc*5 + pf*10) ──────
    ("markov_pc_1", {}, None, 1.0, 0.0, [1.0 + 5.0]),
    ("markov_pf_1", {}, None, 0.0, 1.0, [1.0 + 10.0]),
    ("markov_pc_pf", {}, None, 0.2, 0.1, [1.0 + 1.0 + 1.0]),
    # ── Anomaly scores ─────────────────────────────────────────────────────
    ("mem_anomaly_strong", {"memory_anomaly_score": 4}, None, 0.0, 0.0, [3.0]),
    ("mem_anomaly_mild", {"memory_anomaly_score": 2.5}, None, 0.0, 0.0, [1.5]),
    ("mem_anomaly_at_2", {"memory_anomaly_score": 2}, None, 0.0, 0.0, [1.0]),
    ("cpu_anomaly_strong", {"cpu_anomaly_score": 4}, None, 0.0, 0.0, [2.0]),
    ("cpu_anomaly_mild", {"cpu_anomaly_score": 2.5}, None, 0.0, 0.0, [1.3]),
    ("cpu_anomaly_at_2", {"cpu_anomaly_score": 2}, None, 0.0, 0.0, [1.0]),
    # ── Blast radius ───────────────────────────────────────────────────────
    ("blast_60", {"blast_radius": 60}, None, 0.0, 0.0, [3.0]),
    ("blast_15", {"blast_radius": 15}, None, 0.0, 0.0, [2.0]),
    ("blast_8", {"blast_radius": 8}, None, 0.0, 0.0, [1.5]),
    ("blast_at_5", {"blast_radius": 5}, None, 0.0, 0.0, [1.0]),
    ("blast_zero", {"blast_radius": 0}, None, 0.0, 0.0, [1.0]),
    # ── Pod phase ──────────────────────────────────────────────────────────
    ("phase_failed", {"pod_phase": "Failed"}, None, 0.0, 0.0, [20.0]),
    ("phase_unknown", {"pod_phase": "Unknown"}, None, 0.0, 0.0, [10.0]),
    ("phase_pending", {"pod_phase": "Pending"}, None, 0.0, 0.0, [3.0]),
    ("phase_succeeded", {"pod_phase": "Succeeded"}, None, 0.0, 0.0, [0.1]),
    ("phase_running", {"pod_phase": "Running"}, None, 0.0, 0.0, [1.0]),
    # ── Scheduling / readiness ─────────────────────────────────────────────
    ("not_scheduled", {"pod_scheduled": False}, None, 0.0, 0.0, [5.0]),
    ("not_ready", {"container_ready": False}, None, 0.0, 0.0, [5.0]),
    ("not_scheduled_and_not_ready", {"pod_scheduled": False, "container_ready": False},
     None, 0.0, 0.0, [25.0]),
    # ── Wait states ────────────────────────────────────────────────────────
    ("wait_crashloop", {"wait_state": "CrashLoopBackOff"}, None, 0.0, 0.0, [15.0]),
    ("wait_ccce", {"wait_state": "CreateContainerConfigError"}, None, 0.0, 0.0, [10.0]),
    ("wait_cce", {"wait_state": "CreateContainerError"}, None, 0.0, 0.0, [10.0]),
    ("wait_imagepull", {"wait_state": "ImagePullBackOff"}, None, 0.0, 0.0, [8.0]),
    ("wait_errimagepull", {"wait_state": "ErrImagePull"}, None, 0.0, 0.0, [8.0]),
    ("wait_invalid_image", {"wait_state": "InvalidImageName"}, None, 0.0, 0.0, [8.0]),
    ("wait_runcontainer", {"wait_state": "RunContainerError"}, None, 0.0, 0.0, [10.0]),
    ("wait_containercreating", {"wait_state": "ContainerCreating"}, None, 0.0, 0.0, [2.0]),
    ("wait_podinitializing", {"wait_state": "PodInitializing"}, None, 0.0, 0.0, [2.0]),
    ("wait_unknown_state", {"wait_state": "SomeRandomState"}, None, 0.0, 0.0, [3.0]),
    ("no_wait_state_default", {}, None, 0.0, 0.0, [1.0]),
    # ── Termination ────────────────────────────────────────────────────────
    ("term_oom", {"terminated": True, "terminated_reason": "OOMKilled"},
     None, 0.0, 0.0, [12.0]),
    ("term_error", {"terminated": True, "terminated_reason": "Error"},
     None, 0.0, 0.0, [5.0]),
    ("term_completed", {"terminated": True, "terminated_reason": "Completed"},
     None, 0.0, 0.0, [0.1]),
    ("term_other", {"terminated": True, "terminated_reason": "DeadlineExceeded"},
     None, 0.0, 0.0, [3.0]),
    ("term_reason_none", {"terminated": True, "terminated_reason": None},
     None, 0.0, 0.0, [1.0]),
    ("not_terminated", {"terminated": False, "terminated_reason": "OOMKilled"},
     None, 0.0, 0.0, [1.0]),
    # ── Clamp at 0.99 (mults=None marks a clamp row → expect exactly 0.99) ─
    ("clamp_max", {"memory_pct": 99, "restart_rate_per_hr": 100,
                   "log_error_rate_per_min": 100, "node_memory_pressure": True,
                   "node_disk_pressure": True, "pod_phase": "Failed",
                   "container_ready": False, "wait_state": "CrashLoopBackOff",
                   "terminated": True, "terminated_reason": "OOMKilled",
                   "blast_radius": 100},
     "CRITICAL", 1.0, 1.0, None),
    # ── Compositions (verify multiplicative ordering holds) ────────────────
    ("mem_cpu_restart", {"memory_pct": 96, "cpu_pct": 96, "restart_rate_per_hr": 6},
     None, 0.0, 0.0, [12.0, 3.0, 15.0]),
    ("full_signals", {"memory_pct": 96, "cpu_pct": 96, "restart_rate_per_hr": 6,
                      "log_error_rate_per_min": 11, "node_memory_pressure": True,
                      "memory_anomaly_score": 4, "cpu_anomaly_score": 4,
                      "blast_radius": 60, "pod_scheduled": False,
                      "container_ready": False},
     "STRESSED", 0.2, 0.1, None),
]


@pytest.mark.parametrize("label,metrics,state,pc,pf,mults", ROWS,
                         ids=[r[0] for r in ROWS])
def test_calculate_risk_golden(label, metrics, state, pc, pf, mults):
    if mults is None:
        # Row composed to saturate every strong signal → clamp governs.
        assert calculate_risk(metrics, state, pc, pf) == pytest.approx(0.99, rel=1e-9, abs=1e-12)
        return
    lr = 1.0
    for m in mults:
        lr *= m
    expected = _bayes(lr)
    actual = calculate_risk(metrics, state, pc, pf)
    assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12), (
        f"score mismatch for '{label}': expected {expected:.9f}, got {actual:.9f}"
    )
