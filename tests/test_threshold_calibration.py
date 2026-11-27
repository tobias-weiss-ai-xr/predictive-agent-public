"""A3 — adaptive action risk threshold driven by observed calibration.

after A1, the calibration monitor accumulates realized (predicted, actual)
outcomes even when nothing was done. Once enough samples exist, the effective
remediation risk threshold should move toward the Youden-optimal value implied
by the window: higher when the model over-predicts failure (false-positive
flood), lower when it under-predicts.
"""

from unittest import mock

import pytest

from predictive_agent import config
from predictive_agent.calibration import CalibrationMonitor

from .test_calibration_guardrails import _build_manager, _Pod, _Pred

# Window that makes the optimal threshold strictly ABOVE the 0.7 base:
# failures are predicted ~0.85, a band of false alarms sits just above base
# (0.72), and only a few failures are under-predicted.
RAISES = (
    [(0.85, True)] * 20
    + [(0.72, False)] * 30
    + [(0.2, False)] * 10
    + [(0.3, True)] * 5
)
# Window that makes the optimal threshold strictly BELOW the 0.7 base:
# correct alarms are predicted ~0.6, a band of false alarms sits at 0.55,
# some high false alarms live at 0.9, a few failures under-predicted.
LOWERS = (
    [(0.6, True)] * 20
    + [(0.55, False)] * 30
    + [(0.9, False)] * 10
    + [(0.3, True)] * 5
)


def _mon(pairs, **kw):
    mon = CalibrationMonitor(window=200, min_samples=20, **kw)
    for p, a in pairs:
        mon.record(p, a)
    return mon


# ─── CalibrationMonitor.suggested_threshold ────────────────────────────────
def test_suggested_threshold_none_below_min_samples():
    mon = _mon(LOWERS[:10])
    assert mon.suggested_threshold() is None


def test_suggested_threshold_raises_when_overpredicting():
    mon = _mon(RAISES)
    assert mon.suggested_threshold() > 0.7


def test_suggested_threshold_lowers_when_underpredicting():
    mon = _mon(LOWERS)
    assert mon.suggested_threshold() < 0.7


def test_suggested_threshold_healthy_fleet_keeps_base():
    mon = _mon([(0.1, False)] * 60)
    assert mon.suggested_threshold() == pytest.approx(0.7, abs=1e-9)


def test_suggested_threshold_clamped_to_high():
    # Every alarm is a false one -> optimal separation only above 0.99; the
    # suggestion must stay within the configurable band.
    mon = _mon([(0.99, False)] * 60)
    assert mon.suggested_threshold() == pytest.approx(0.85, abs=1e-9)


def test_suggested_threshold_respects_custom_bounds():
    mon = _mon([(0.99, False)] * 60)
    s = mon.suggested_threshold(low=0.5, high=0.75)
    assert 0.5 <= s <= 0.75


# ─── RemediationManager.recalibrate_threshold ──────────────────────────────
def test_recalibrate_no_monitor_no_change():
    rem = _build_manager()
    assert rem.calibration_monitor is None
    assert rem.recalibrate_threshold() == 0.7


def test_recalibrate_fake_monitor_without_method_no_change():
    rem = _build_manager()

    class _Fake:
        pass

    rem.calibration_monitor = _Fake()
    assert rem.recalibrate_threshold() == 0.7


def test_recalibrate_insufficient_samples_no_change():
    rem = _build_manager()
    rem.calibration_monitor = _mon(LOWERS[:10])
    assert rem.recalibrate_threshold() == 0.7


def test_recalibrate_disabled_no_change():
    rem = _build_manager()
    rem.calibration_monitor = _mon(LOWERS)
    with mock.patch.object(config, "THRESHOLD_CALIBRATION_ENABLED", False):
        assert rem.recalibrate_threshold() == 0.7


def test_recalibrate_lowers_toward_suggestion():
    rem = _build_manager()
    rem.calibration_monitor = _mon(LOWERS)
    eff = rem.recalibrate_threshold()
    assert 0.55 < eff < 0.7
    assert rem.effective_risk_threshold == eff
    assert rem.threshold_recalibrations == 1


def test_recalibrate_raises_toward_suggestion():
    rem = _build_manager()
    rem.calibration_monitor = _mon(RAISES)
    assert rem.recalibrate_threshold() > 0.7


def test_recalibrate_eventually_converges_and_stays_bounded():
    rem = _build_manager()
    rem.calibration_monitor = _mon(LOWERS)
    prev = 0.7
    for _ in range(20):
        cur = rem.recalibrate_threshold()
        assert config.THRESHOLD_CALIBRATION_MIN <= cur <= config.THRESHOLD_CALIBRATION_MAX
        assert cur <= prev + 1e-9  # monotone non-increasing for a lowering window
        prev = cur
    # Converged onto (or very close to) the bounded optimum.
    assert prev < 0.6


# ─── evaluate() gates on the EFFECTIVE threshold (reactive path) ──────────
def test_evaluate_uses_effective_threshold_above_base():
    """risk between effective and base must still be remediated."""
    rem = _build_manager()
    rem.effective_risk_threshold = 0.6
    results = rem.evaluate(_Pod("CRITICAL"), _Pred(risk_score=0.66, confidence=0.9), 0.66)
    assert results, "risk 0.66 >= effective 0.6 must pass the gate (base 0.7 would skip)"


def test_evaluate_uses_effective_threshold_below():
    """risk below effective must be skipped even though base is 0.7."""
    rem = _build_manager()
    rem.effective_risk_threshold = 0.6
    results = rem.evaluate(_Pod("CRITICAL"), _Pred(risk_score=0.55, confidence=0.9), 0.55)
    assert results == []


def test_recalibrate_updates_status():
    rem = _build_manager()
    rem.calibration_monitor = _mon(LOWERS)
    rem.recalibrate_threshold()
    stats = rem.get_stats()
    assert stats["effective_risk_threshold"] == rem.effective_risk_threshold
    assert stats["threshold_recalibrations"] == 1
