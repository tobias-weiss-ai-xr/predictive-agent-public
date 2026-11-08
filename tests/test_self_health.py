"""Tests for predictive_agent.self_health — rate-limited self-issue alerts."""

import pytest

from predictive_agent.self_health import (
    DegradedCycleTracker,
    SelfHealthAlerter,
)


class FakeClock:
    """Deterministic monotonic clock."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


@pytest.fixture
def clock():
    return FakeClock()


class TestSelfHealthAlerter:
    def test_first_report_sends(self, clock):
        sent = []
        a = SelfHealthAlerter(
            notify_fn=lambda s, b: sent.append((s, b)) or True,
            cooldown_s=3600,
            clock=clock,
        )
        assert a.report("persistence", "state persist broken", "EACCES") is True
        assert len(sent) == 1
        assert "[self-health:persistence]" in sent[0][0]
        assert "EACCES" in sent[0][1]

    def test_repeat_within_cooldown_suppressed(self, clock):
        count = {"n": 0}
        a = SelfHealthAlerter(
            notify_fn=lambda s, b: count.__setitem__("n", count["n"] + 1) or True,
            cooldown_s=3600,
            clock=clock,
        )
        assert a.report("persistence", "down", "") is True
        clock.advance(60)
        assert a.report("persistence", "still down", "") is False
        clock.advance(60)
        assert a.report("persistence", "still down", "") is False
        assert count["n"] == 1  # one mail, not one per cycle

    def test_repeat_after_cooldown_sends_again(self, clock):
        count = {"n": 0}
        a = SelfHealthAlerter(
            notify_fn=lambda s, b: count.__setitem__("n", count["n"] + 1) or True,
            cooldown_s=3600,
            clock=clock,
        )
        a.report("persistence", "down", "")
        clock.advance(3600)
        assert a.report("persistence", "down again", "") is True
        assert count["n"] == 2

    def test_kinds_are_independent(self, clock):
        kinds = []
        a = SelfHealthAlerter(
            notify_fn=lambda s, b: kinds.append(s) or True,
            cooldown_s=3600,
            clock=clock,
        )
        assert a.report("persistence", "a", "") is True
        assert a.report("collector", "b", "") is True  # different kind, no cooldown

    def test_notify_none_is_noop(self, clock):
        a = SelfHealthAlerter(notify_fn=None, clock=clock)
        assert a.report("persistence", "down", "") is False
        # ...and it still records the cooldown (no exception, no spam on wire-up)
        assert a.report("persistence", "down", "") is False

    def test_never_raises_on_notify_failure(self, clock):
        def boom(subject, body):
            raise RuntimeError("smtp down")

        a = SelfHealthAlerter(notify_fn=boom, clock=clock)
        assert a.report("persistence", "x", "") is False

    def test_reset_clears_cooldown(self, clock):
        count = {"n": 0}
        a = SelfHealthAlerter(
            notify_fn=lambda s, b: count.__setitem__("n", count["n"] + 1) or True,
            clock=clock,
        )
        a.report("persistence", "down", "")
        a.reset()
        assert a.report("persistence", "down", "") is True
        assert count["n"] == 2

    def test_footer_identifies_agent(self, clock):
        sent = []
        a = SelfHealthAlerter(
            notify_fn=lambda s, b: sent.append(b) or True, clock=clock
        )
        a.report("persistence", "down", "detail here")
        assert "detail here" in sent[0]


class TestDegradedCycleTracker:
    def test_healthy_cycles_do_not_alert(self, clock):
        sent = []
        a = SelfHealthAlerter(
            notify_fn=lambda s, b: sent.append(s) or True, clock=clock
        )
        t = DegradedCycleTracker(a, threshold=5)
        for _ in range(10):
            assert t.cycle("HEALTHY") is False
        assert sent == []

    def test_streak_below_threshold_no_alert(self, clock):
        a = SelfHealthAlerter(notify_fn=lambda s, b: True, clock=clock)
        t = DegradedCycleTracker(a, threshold=5)
        for _ in range(4):
            assert t.cycle("DEGRADED") is False

    def test_streak_at_threshold_alerts_once(self, clock):
        sent = []
        a = SelfHealthAlerter(
            notify_fn=lambda s, b: sent.append(s) or True, clock=clock
        )
        t = DegradedCycleTracker(a, threshold=5)
        results = [t.cycle("DEGRADED") for _ in range(8)]
        assert results == [False, False, False, False, True, False, False, False]
        assert len(sent) == 1
        assert "[self-health:collector]" in sent[0]

    def test_recovery_resets_streak(self, clock):
        a = SelfHealthAlerter(notify_fn=lambda s, b: True, cooldown_s=0, clock=clock)
        t = DegradedCycleTracker(a, threshold=3)
        assert t.cycle("DEGRADED") is False
        assert t.cycle("DEGRADED") is False
        assert t.cycle("DEGRADED") is True
        assert t.cycle("HEALTHY") is False
        assert t.consecutive == 0
        # fresh streak must build up again before the next alert
        assert t.cycle("DEGRADED") is False
        assert t.cycle("DEGRADED") is False
        assert t.cycle("DEGRADED") is True

    def test_mixed_degraded_states_count_as_degraded(self, clock):
        a = SelfHealthAlerter(notify_fn=lambda s, b: True, clock=clock)
        t = DegradedCycleTracker(a, threshold=2)
        assert t.cycle("DEGRADED") is False
        # anything that is not HEALTHY keeps the streak alive
        assert t.cycle("UNKNOWN") is True
