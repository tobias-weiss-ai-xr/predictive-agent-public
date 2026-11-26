"""Holt-Winters triple exponential smoothing (additive seasonality).

Dependency-free seasonal forecaster for per-pod usage series. FIT/∂F are the
only calibration signals the operator needs from it:

    * ``expected_velocity()`` — what the seasonal pattern says usage is doing
      *right now* (units per sample-interval). This is the "seasonal trend"
      that complements Kalman's linear velocity: Kalman peaks late on a ramp,
      the seasonal profile knows e.g. that Monday 09:00 usage historically
      climbs fast.
    * ``one_step_error()`` / ``rolling_mae()`` — realized forecast error used
      for backtest-driven model selection (compare against Kalman's MAE on the
      same window and keep the trend source with the lower error).

The model is additive Holt-Winters (Holt 1957, Winters 1960) with a single
seasonal period (default 24h of 60s samples = 1440). Initial level/trend come
from a least-squares fit over the first full period; initial seasonal indices
from the period-mean profile.

Everything here is pure computation on plain timestamps/values — no IO, no
globals — so the core is trivially unit-testable.
"""

from __future__ import annotations

from collections import deque


class HoltWinters:
    """Triple exponential smoothing with additive seasonality.

    Args:
        season_len: Seasonal buckets per period (default 1440). With the
            default sample_interval of 60s this is the 24h day shape.
        sample_interval: Seconds covered by one sample/bucket (default 60.0).
            The wall-clock timestamp is mapped onto the seasonal index via
            ``((ts - t0) // interval) % season_len``, so a 60s cadence walks
            one bucket per observation. season_len*interval = period seconds.
        alpha: Level smoothing (0..1).
        beta: Trend smoothing (0..1).
        gamma: Seasonal smoothing (0..1).

    Usage:
        hw = HoltWinters(season_len=1440, sample_interval=60.0)
        for ts, value in samples:
            hw.observe(ts, value)
        when confident: hw.expected_velocity(ts)
    """

    def __init__(self, season_len: int = 1440, sample_interval: float = 60.0,
                 alpha: float = 0.25, beta: float = 0.02, gamma: float = 0.10):
        if season_len < 4:
            raise ValueError("season_len must be >= 4")
        if sample_interval <= 0:
            raise ValueError("sample_interval must be > 0")
        if not (0 < alpha < 1 and 0 < beta < 1 and 0 < gamma < 1):
            raise ValueError("smoothing params must be in (0, 1)")
        self.season_len = int(season_len)
        self.sample_interval = float(sample_interval)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.last_ts: float | None = None
        self.level: float | None = None
        self.trend: float = 0.0
        self.season: list[float] | None = None
        self.errors: deque[float] = deque(maxlen=season_len)  # one-step |err|
        self._samples: list[tuple[float, float]] = []
        self._fitted = False
        # Fraction of season indices seen during warm-up (0..1). A fit over
        # misaligned/gappy timing collapses most samples onto a few indices;
        # requiring broad coverage prevents trusting a degenerate profile.
        self.season_coverage: float = 0.0
        # Typical per-sample |change| over the trailing window. Used to bound
        # expected_velocity() so an index-collapse artifact (huge delta between
        # two adjacent profile rows that never actually occurred step-to-step)
        # cannot produce an absurd trend.
        self._typical_step: float = 0.0

    # ── Fit ────────────────────────────────────────────────────────────────
    def _index(self, ts: float) -> int:
        """Map a wall-clock timestamp onto a seasonal bucket index."""
        return int(((ts - self._t0) % (self.season_len * self.sample_interval))
                   // self.sample_interval)

    def _warmup_profile(self):
        """Initial seasonal indices from period-mean profile of the samples."""
        season = [0.0] * self.season_len
        counts = [0] * self.season_len
        for _, v in self._samples:
            idx = self._index(_)
            season[idx] += v
            counts[idx] += 1
        mean = sum(season) / max(1, sum(counts))
        covered = sum(1 for c in counts if c)
        self.season_coverage = covered / self.season_len
        for i in range(self.season_len):
            season[i] = (season[i] / max(1, counts[i])) - mean if counts[i] else 0.0
        return season

    def _measure_typical_step(self):
        """Mean absolute per-sample change across a trailing window."""
        n = max(1, self.season_len)
        tail = self._samples[-n:]
        if len(tail) < 2:
            self._typical_step = 0.0
            return
        diffs = [abs(tail[i][1] - tail[i - 1][1]) for i in range(1, len(tail))]
        self._typical_step = sum(diffs) / len(diffs)

    def _lstsq_trend(self):
        """Least-squares trend (units/sample) over the first full period."""
        n = max(1, self.season_len)
        xs = [(t - self._t0) for t, _ in self._samples[:n]]
        ys = [v for _, v in self._samples[:n]]
        xm = sum(xs) / len(xs) if xs else 0.0
        ym = sum(ys) / len(ys) if ys else 0.0
        num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
        den = sum((x - xm) ** 2 for x in xs)
        slope = num / den if den else 0.0
        return ym, slope  # ym = period mean (DC level), slope = per-sample trend

    def fit(self):
        """Fit the model over the buffered samples. Idempotent; refits cheaply."""
        if len(self._samples) < self.season_len + max(1, self.season_len // 4):
            return False  # not enough history for a stable seasonal profile
        self._t0 = self._samples[0][0]
        mean_level, trend = self._lstsq_trend()
        level = mean_level  # DC level = period mean (not the intercept): the
        # seasonal profile is centered on this, so level+season[idx] matches
        # the raw series phase (additive HW consistency).
        season = self._warmup_profile()
        self._measure_typical_step()
        # Iterate over all samples smoothing level/trend/season in place.
        # Canonical additive HW: the level update de-seasonalizes (v - season)
        # so level tracks the DC/trend component and the profile keeps the
        # periodic shape instead of being absorbed into the level.
        for ts, v in self._samples:
            idx = self._index(ts)
            pred = level + trend + season[idx]
            err = v - pred
            last_level = level
            level = self.alpha * (v - season[idx]) + (1 - self.alpha) * (level + trend)
            trend = self.beta * (level - last_level) + (1 - self.beta) * trend
            season[idx] = self.gamma * (v - level) + (1 - self.gamma) * season[idx]
            self.errors.append(abs(err))
        self.level = level
        self.trend = trend
        self.season = season
        self.last_ts = self._samples[-1][0]
        self._fitted = True
        return True

    # ── Online continuation ───────────────────────────────────────────────
    def observe(self, ts: float, value: float) -> None:
        """Append one sample. Buffers during warm-up, then continues online."""
        self._samples.append((ts, value))
        # Keep the buffer bounded to one period beyond the fit window.
        if len(self._samples) > self.season_len * 2:
            self._samples = self._samples[-(self.season_len + 1):]
        if self._fitted:
            self._step(ts, value)
        else:
            self.fit()
        self.last_ts = ts

    def _step(self, ts: float, value: float) -> None:
        if self.level is None or self.season is None:
            return
        idx = self._index(ts)
        pred = self.level + self.trend + self.season[idx]
        err = value - pred
        last_level = self.level
        self.level = self.alpha * (value - self.season[idx]) + (1 - self.alpha) * (
            self.level + self.trend
        )
        self.trend = self.beta * (self.level - last_level) + (1 - self.beta) * self.trend
        self.season[idx] = self.gamma * (value - self.level) + (
            1 - self.gamma
        ) * self.season[idx]
        self.errors.append(abs(err))

    # ── Queries ───────────────────────────────────────────────────────────
    @property
    def fitted(self) -> bool:
        return self._fitted

    # ── Warm-up observability ─────────────────────────────────────────────
    # A fit needs one full period plus a quarter-period of profiles. At the
    # default 60s cadence / 1440 buckets that is ~30h of uninterrupted
    # samples, so a fresh operator reports confidence 0.0 for a long time.
    # Exposing progress makes that an explainable "warming up" instead of a
    # silent 0.00 that reads like a bug.
    @property
    def samples_required(self) -> int:
        """Samples needed before the first fit."""
        return self.season_len + max(1, self.season_len // 4)

    @property
    def samples_seen(self) -> int:
        """Samples currently buffered toward the fit window."""
        return len(self._samples)

    @property
    def warmup_progress(self) -> float:
        """0..1 fraction of the required samples collected (1.0 when fitted)."""
        need = self.samples_required
        if not need:
            return 1.0
        return min(1.0, len(self._samples) / float(need))

    @property
    def confidence(self) -> float:
        """0..1 — how much to trust the seasonal signal.

        High when seasonal amplitude is meaningful relative to level and the
        realized one-step MAE is small relative to the signal.
        """
        if self.level is None or not self.errors:
            return 0.0
        if self.season_coverage < 0.5:
            return 0.0  # too few season indices seen -> degenerate profile
        amp = (max(self.season) - min(self.season)) if self.season else 0.0
        base = max(1e-9, abs(self.level))
        mae = sum(self.errors) / len(self.errors)
        if amp / base < 0.1:
            return 0.0  # no discernible seasonality
        # Error is judged against the signal's own scale (the larger of
        # seasonal amplitude and level): a low-base/hourly-swing series must
        # not be punished because its absolute MAE dwarfs its quiet level.
        scale = max(amp, base)
        err_ratio = mae / scale
        conf = min(1.0, (amp / base) / 0.5) * max(0.0, 1.0 - err_ratio * 2.0)
        return max(0.0, conf)

    def expected_velocity(self, ts: float | None = None) -> float:
        """Units per sample-interval the seasonal pattern implies at ``ts``.

        = trend component + (seasonal level change over one sample). Guarded:
        a degenerate/misaligned profile (huge adjacent-row delta that no real
        step-to-step sample ever shows) is clamped to trend + sign-capped
        seasonal delta, so an index-collapse artifact cannot emit an absurd
        velocity.
        """
        if self.level is None or self.season is None:
            return 0.0
        t = ts if ts is not None else self.last_ts
        if t is None:
            return 0.0
        idx = self._index(t)
        idx_next = (idx + 1) % self.season_len
        seasonal_delta = self.season[idx_next] - self.season[idx]
        bound = max(2.0 * self._typical_step, abs(self.trend), 1e-9)
        seasonal_delta = max(-bound, min(bound, seasonal_delta))
        return self.trend + seasonal_delta

    def forecast(self, steps: int, ts: float | None = None) -> float:
        """Forecast value ``steps`` samples ahead of ``ts`` (additive model)."""
        if self.level is None or self.season is None:
            return 0.0
        t = ts if ts is not None else self.last_ts
        if t is None:
            return 0.0
        idx = self._index(t)
        fwd = int(steps)
        return self.level + self.trend * fwd + self.season[(idx + fwd) % self.season_len]

    def rolling_mae(self) -> float:
        if not self.errors:
            return 0.0
        return sum(self.errors) / len(self.errors)
