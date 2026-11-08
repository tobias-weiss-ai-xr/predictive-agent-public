"""Hysteresis tests: state transitions must be confirmed, not instantaneous.

Regression for the production rabbitmq flap — a pod hovering at the
DEGRADED/HEALTHY boundary reclassified every reconcile cycle (~10 min on
edge-node), flapping alerts. A committed state change now requires
`STATE_HYSTERESIS` consecutive agreeing classifications.
"""

from predictive_agent.state_model import PodTracker


def _update(tracker, memory_pct=50.0, restart_rate_bump=0, log_errors=0, cpu_pct=40.0):
    """Feed one update cycle at a chosen boundary level.

    71% memory of a 1024 MiB limit is score 2 (HEALTHY at rest);
    71% + 1 restart/hour crosses to 3 (DEGRADED boundary);
    96% memory + 11 log errors + 96% cpu = 12 (CRITICAL).
    """
    tracker.update(
        memory_mib=1024 * memory_pct / 100.0,
        memory_limit_mib=1024,
        cpu_m=cpu_pct * 10.0,
        restart_count=restart_rate_bump,
        log_errors=log_errors,
        node_pressure=False,
    )


def test_single_boundary_crossing_does_not_flip_state():
    """One degraded-looking cycle must not commit DEGRADED (jitter)."""
    t = PodTracker("ns", "rabbitmq-0")
    _update(t, memory_pct=40.0)  # baseline HEALTHY
    assert t.state == "HEALTHY"

    _update(t, memory_pct=71.0, restart_rate_bump=1)  # raw score 3
    assert t.state == "HEALTHY", "single crossing must be absorbed"

    _update(t, memory_pct=71.0, restart_rate_bump=1)  # sustained
    assert t.state == "DEGRADED", "confirmed after hysteresis cycles"


def test_alternating_jitter_never_flaps():
    """Alternating healthy/degraded cycles must hold the committed state."""
    t = PodTracker("ns", "rabbitmq-0")
    _update(t, memory_pct=40.0)
    committed = t.state
    for i in range(6):
        if i % 2 == 0:
            _update(t, memory_pct=71.0, restart_rate_bump=1)
        else:
            _update(t, memory_pct=40.0)
        assert t.state == committed, f"flap on cycle {i}"


def test_sustained_degradation_commits_quickly():
    """A genuinely degrading pod escalates after hysteresis, not never."""
    t = PodTracker("ns", "app-0")
    _update(t, memory_pct=40.0)
    for _ in range(2):
        _update(t, memory_pct=96.0, log_errors=11, cpu_pct=96.0)  # raw score 12
    assert t.state == "CRITICAL"


def test_recovery_also_needs_confirmation():
    """De-escalation is symmetric — one good cycle doesn't clear DEGRADED."""
    t = PodTracker("ns", "app-0")
    for _ in range(3):
        _update(t, memory_pct=96.0, log_errors=11, cpu_pct=96.0)
    assert t.state == "CRITICAL"

    _update(t, memory_pct=40.0)
    assert t.state == "CRITICAL", "single healthy cycle must not recover"
    # 1 more cycle confirms DEGRADED (markov stickiness), 2 more confirm HEALTHY
    for _ in range(4):
        _update(t, memory_pct=40.0)
    assert t.state == "HEALTHY"


def test_hysteresis_one_is_instant_backcompat():
    """hysteresis=1 restores the old immediate-classification behavior."""
    t = PodTracker("ns", "app-0", hysteresis=1)
    _update(t, memory_pct=71.0, restart_rate_bump=1)
    assert t.state == "DEGRADED"


def test_pending_progress_resets_on_conflicting_signal():
    """A→(pending)→B must restart the confirmation count for B, not A."""
    t = PodTracker("ns", "app-0")
    _update(t, memory_pct=71.0, restart_rate_bump=1)   # pending DEGRADED 1/2
    _update(t, memory_pct=96.0, log_errors=11)          # pending CRITICAL — reset
    _update(t, memory_pct=71.0, restart_rate_bump=1)   # pending DEGRADED 1/2
    assert t.state == "HEALTHY"
    _update(t, memory_pct=71.0, restart_rate_bump=1)   # DEGRADED confirmed
    assert t.state == "DEGRADED"
