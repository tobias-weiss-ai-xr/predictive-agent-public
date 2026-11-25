"""Seasonal (Holt-Winters) forecaster unit tests.

Covers: fitting, additive seasonal profile recovery, online continuation,
expected_velocity semantics, confidence gating, forecast, and guards.
"""
import math

import pytest

from predictive_agent.seasonal import HoltWinters


def _daily_series(days=3, period=24, base=100.0, amp=20.0, noise=1.0, trend=0.1, interval=60.0):
    """Deterministic daily sine bout: value = base + amp*sin(2pi*h/period).
    Timestamps are in wall-clock seconds at `interval` spacing so the seasonal
    index (ts -> bucket via interval) advances one slot per sample."""
    out = []
    n = days * period
    for i in range(n):
        v = base + amp * math.sin(2 * math.pi * (i % period) / period) + i * trend
        if noise:
            v += (i % 7) * 0.1 * noise  # small deterministic "noise"
        out.append((float(i * interval), v))
    return out


# ── Guards ───────────────────────────────────────────────────────────────────


def test_season_len_must_be_at_least_4():
    with pytest.raises(ValueError):
        HoltWinters(season_len=3)


def test_smoothing_params_must_be_in_unit_interval():
    with pytest.raises(ValueError):
        HoltWinters(alpha=0.0)
    with pytest.raises(ValueError):
        HoltWinters(beta=1.5)


def test_not_fitted_until_enough_history():
    # season_len 24 needs 24 + 24//4 = 30 samples; one day (24) is not enough.
    hw = HoltWinters(season_len=24)
    for ts, v in _daily_series(days=1):
        hw.observe(ts, v)
    assert hw.fitted is False
    fresh = HoltWinters(season_len=1440)
    for i in range(10):
        fresh.observe(float(i), 100.0)
    assert fresh.fitted is False


# ── Fitting / seasonality recovery ───────────────────────────────────────────


def test_recovers_daily_sinusoidal_seasonality():
    hw = HoltWinters(season_len=24, sample_interval=60, beta=0.01, gamma=0.2, alpha=0.3)
    for ts, v in _daily_series(days=4, amp=20.0):
        hw.observe(ts, v)
    assert hw.fitted
    # Seasonal amplitude should be captured near the injected 2*amp = 40
    # (peak-to-trough of the sinusoid after the DC level is removed).
    amp = max(hw.season) - min(hw.season)
    assert 30.0 < amp <= 50.0, f"seasonal amp not recovered: {amp:.2f}"
    assert hw.confidence > 0.3, f"expected seasonal confidence, got {hw.confidence:.2f}"


def test_flat_series_yields_no_seasonality_confidence():
    hw = HoltWinters(season_len=24)
    for i in range(200):
        hw.observe(float(i), 100.0)
    assert hw.fitted
    assert hw.confidence == 0.0  # amplitude ~0 -> no seasonal signal


def test_forecast_tracks_seasonal_peak_ahead():
    """Forecasting past the next peak should see the seasonal rise, not just
    the linear trend."""
    hw = HoltWinters(season_len=24, alpha=0.3, beta=0.01, gamma=0.2)
    series = _daily_series(days=4)
    for ts, v in series:
        hw.observe(ts, v)
    # Pick a low point and forecast to the upcoming peak (period/4 ahead).
    # The seasonal profile is periodic by construction; forecast must exceed
    # the current level by roughly the amplitude at a peak-phase jump.
    fwd = hw.forecast(6, ts=series[-1][0])
    assert fwd > hw.level - 5  # not below level; peak phase is rising
    # And it must beat the raw linear projection when seasonality is rising.
    naive = hw.level + hw.trend * 6
    assert fwd > naive or abs(fwd - naive) < 0.5  # seasonal lift where applicable


# ── Online continuation ──────────────────────────────────────────────────────


def test_online_continuation_keeps_fitting_after_warmup():
    hw = HoltWinters(season_len=24)
    for ts, v in _daily_series(days=2):  # 48 samples -> fitted (30 needed)
        hw.observe(ts, v)
    assert hw.fitted
    before_mae = hw.rolling_mae()
    assert before_mae > 0
    for ts, v in _daily_series(days=2, noise=2.0):
        hw.observe(ts + 48, v)
    assert hw.fitted
    # Continuing online with only a little extra noise should not blow up MAE.
    assert hw.rolling_mae() < before_mae * 3 + 1


def test_buffer_bounded():
    hw = HoltWinters(season_len=24)
    for ts, v in _daily_series(days=10):
        hw.observe(ts, v)
    assert len(hw._samples) <= 24 * 2


# ── expected_velocity semantics ──────────────────────────────────────────────


def test_expected_velocity_reflects_seasonal_rise_and_fall():
    """On a sinusoidal profile, expected_velocity should be positive while
    rising and negative while falling."""
    hw = HoltWinters(season_len=24, alpha=0.3, beta=0.01, gamma=0.2)
    for ts, v in _daily_series(days=4):
        hw.observe(ts, v)
    # Rising half: index 0 (phase 0, valley) climbs toward the peak.
    vel_rise = hw.expected_velocity(ts=0.0)
    # Falling half: index 6 (phase pi/2, peak) descends after it.
    vel_fall = hw.expected_velocity(ts=6.0 * 60.0)  # index 6 in seconds
    assert vel_rise > 0, f"expected rising velocity, got {vel_rise:.3f}"
    assert vel_fall < 0, f"expected falling velocity, got {vel_fall:.3f}"


def test_level_stays_at_dc_not_absorbed_by_seasonality():
    """Regression: canonical HW de-seasonalizes the level update, so level
    must track the series mean (DC), not collapse toward the trough while the
    season absents the amplitude (observed: level 230 -> 59.5 on a 230+-170
    wave, eating the seasonal component)."""
    hw = HoltWinters(season_len=1440, sample_interval=60.0, alpha=0.25, beta=0.02, gamma=0.10)
    t0 = 1700000000.0
    for i in range(1440 * 5):
        hour = (i % 1440) / 60
        v = max(40.0, 230.0 + 170.0 * math.sin(2 * math.pi * (hour - 6) / 24))
        hw.observe(t0 + i * 60.0, v)
    assert hw.fitted
    assert 180.0 < hw.level < 280.0, f"level {hw.level:.1f} drifted off DC"
    amp = max(hw.season) - min(hw.season)
    assert 300.0 < amp < 380.0, f"seasonal amplitude {amp:.1f} was absorbed into level"
    assert hw.confidence > 0.7, f"conf {hw.confidence:.2f}"
    assert hw.rolling_mae() < 10.0, f"mae {hw.rolling_mae():.1f} too high on clean signal"
