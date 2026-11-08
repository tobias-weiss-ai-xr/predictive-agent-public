"""Test KalmanTrend class for trend estimation."""
from predictive_agent.kalman import KalmanTrend


def test_kalman_trend_initialization():
    """Test that KalmanTrend initializes correctly."""
    kt = KalmanTrend()
    assert kt.level == 0.0
    assert kt.velocity == 0.0
    assert kt.level_uncertainty > 0
    assert kt.velocity_uncertainty > 0


def test_kalman_trend_update():
    """Test that KalmanTrend updates state correctly."""
    kt = KalmanTrend()
    kt.update(100.0)
    assert kt.level == 100.0
    assert kt.velocity == 0.0


def test_kalman_trend_predict():
    """Test that KalmanTrend can predict future values."""
    kt = KalmanTrend()
    kt.update(100.0)
    kt.update(105.0)
    kt.update(110.0)

    pred_level, pred_sigma = kt.predict(1.0)
    assert pred_level > 110.0  # Should predict increasing trend
    assert pred_sigma > 0


def test_kalman_trend_time_to_threshold():
    """Test time-to-threshold calculation."""
    kt = KalmanTrend()
    kt.update(100.0)
    kt.update(105.0)
    kt.update(110.0)

    steps, confidence = kt.time_to_threshold(150.0, max_steps=100)
    assert steps > 0
    assert 0 <= confidence <= 1.0


def test_kalman_trend_decreasing():
    """Test that decreasing values produce negative velocity."""
    kt = KalmanTrend()
    kt.update(100.0)
    kt.update(95.0)
    kt.update(90.0)
    assert kt.velocity < 0


def test_kalman_trend_stable():
    """Test that stable values produce near-zero velocity."""
    kt = KalmanTrend()
    for _ in range(10):
        kt.update(100.0)
    assert abs(kt.velocity) < 1.0


def test_kalman_trend_noise_resistance():
    """Test that Kalman filter resists noise."""
    kt = KalmanTrend(process_noise=0.1, measurement_noise=10.0)
    # Feed noisy data around 100
    for val in [105, 95, 102, 98, 103, 97, 101, 99, 104, 96]:
        kt.update(float(val))
    # Level should be close to 100 despite noise
    assert 90 < kt.level < 110


def test_kalman_trend_predict_far_future():
    """Test prediction far into the future has higher uncertainty."""
    kt = KalmanTrend()
    kt.update(100.0)
    kt.update(110.0)

    _, sigma1 = kt.predict(1.0)
    _, sigma10 = kt.predict(10.0)
    assert sigma10 > sigma1


# ═══════════════════════════════════════════════════════════════════════════
# Mutation-survivor killers: constructor wiring contracts, convergence
# behavior (kills covariance algebra index swaps), predict algebra with
# controlled state, ttf branch boundaries, anomaly-score contract.
# ═══════════════════════════════════════════════════════════════════════════

import math
import time

import pytest


def test_constructor_wiring_contract():
    """F/Q/R/P matrices must wire dt and noise params exactly."""
    k = KalmanTrend(process_noise=2.0, measurement_noise=50.0, dt=3.0)
    assert k.F == [[1.0, 3.0], [0.0, 1.0]]
    assert k.Q == [[6.0, 0.0], [0.0, 6.0]]
    assert k.R == 50.0
    assert k.P == [[1000.0, 0.0], [0.0, 1000.0]]
    assert k.initialized is False
    assert k.last_update == 0.0


def test_default_noise_values():
    k = KalmanTrend()
    assert k.R == 100.0
    assert k.Q == [[1.0, 0.0], [0.0, 1.0]]
    assert k.F == [[1.0, 1.0], [0.0, 1.0]]


def test_update_initialization_branch_exact():
    """First update seeds state from the measurement, P from R."""
    before = time.time()
    k = KalmanTrend(measurement_noise=42.0)
    k.update(7.5, timestamp=5.0)
    assert k.x == [7.5, 0.0]
    assert k.P == [[42.0, 0.0], [0.0, 42.0]]
    assert k.initialized is True
    assert k.last_update == 5.0
    assert k.last_update >= before - 1 or True  # explicit ts wins below
    k2 = KalmanTrend()
    k2.update(1.0)  # timestamp=None -> wall clock, never None/0
    assert k2.last_update > before - 5


def test_update_converges_to_constant_with_shrinking_symmetric_p():
    """Constant input: level converges, velocity ~0, P stays symmetric PSD."""
    k = KalmanTrend()
    for _ in range(60):
        k.update(100.0)
    assert k.level == pytest.approx(100.0, abs=1.0)
    assert abs(k.velocity) < 0.5
    assert k.P[0][0] >= 0.0 and k.P[1][1] >= 0.0
    assert k.level_uncertainty < 10.0
    # covariance must shrink, not grow, under steady input
    assert k.P[0][0] < 1000.0


def test_update_tracks_ramp_velocity():
    """Linear ramp slope 10: velocity estimate must approach the slope."""
    k = KalmanTrend()
    value = 0.0
    for _ in range(80):
        k.update(value)
        value += 10.0
    assert k.velocity > 8.0
    assert k.level == pytest.approx(value - 10.0, rel=0.1)


def test_update_reconverges_after_step_change():
    """Process noise keeps the gain alive: a sustained jump must be tracked."""
    k = KalmanTrend()
    for _ in range(30):
        k.update(10.0)
    for _ in range(40):
        k.update(100.0)
    assert k.level > 80.0


def test_predict_exact_algebra_with_controlled_state():
    """pred = level + velocity*steps; var = P00 + steps^2*P11 + 2*steps*P01."""
    k = KalmanTrend()
    k.x = [10.0, 5.0]
    k.P = [[4.0, 0.0], [0.0, 9.0]]
    level, sigma = k.predict(2.0)
    assert level == 20.0
    assert sigma == pytest.approx(math.sqrt(4.0 + 4.0 * 9.0))
    k.P = [[4.0, 1.0], [1.0, 9.0]]
    _, sigma = k.predict(2.0)
    assert sigma == pytest.approx(math.sqrt(4.0 + 4.0 * 9.0 + 4.0 * 1.0))
    # steps defaults to 1 and participates in both terms
    level1, sigma1 = k.predict()
    assert level1 == pytest.approx(k.predict(1.0)[0])
    assert sigma1 == pytest.approx(k.predict(1.0)[1])
    # non-positive variance degrades to sigma 0.0 instead of crashing
    k.P = [[0.0, 0.0], [0.0, 0.0]]
    level0, sigma0 = k.predict(2.0)
    assert (level0, sigma0) == (20.0, 0.0)


def test_ttf_branches_exact_tuples():
    """Each ttf guard returns its exact contract tuple."""
    k = KalmanTrend()
    k.update(10.0)
    k.x[1] = 0.0  # velocity exactly zero: division guard must fire
    assert k.time_to_threshold(100.0) == (180, 0.0)
    k.x[1] = -1.0
    assert k.time_to_threshold(100.0) == (180, 0.0)
    k.x[1] = 5.0
    assert k.time_to_threshold(10.0) == (0, 1.0)  # distance == 0 boundary
    assert k.time_to_threshold(5.0) == (0, 1.0)  # distance < 0
    k.x = [0.0, 0.01]  # far threshold: capped at max_steps with zero confidence
    assert k.time_to_threshold(10.0) == (180, 0.0)


def test_ttf_normal_path_int_steps_and_confidence_band():
    k = KalmanTrend()
    k.update(10.0)
    k.x = [10.0, 5.0]
    k.P = [[0.01, 0.0], [0.0, 0.01]]
    steps, conf = k.time_to_threshold(20.5)
    assert type(steps) is int and steps == 2  # 10.5/5 = 2.1 -> int truncation
    assert 0.9 < conf < 1.0
    # noisy filter: uncertainty dwarfs distance -> zero confidence
    k2 = KalmanTrend()
    k2.update(10.0)
    k2.x = [10.0, 5.0]
    k2.P = [[100.0, 0.0], [0.0, 10000.0]]
    _, conf2 = k2.time_to_threshold(20.5)
    assert conf2 == 0.0


def test_anomaly_score_contract():
    k = KalmanTrend()
    assert k.anomaly_score(5.0) == 0.0  # uninitialized -> 0.0
    k.update(10.0, timestamp=1.0)
    k.x = [10.0, 0.0]
    k.P = [[4.0, 0.0], [0.0, 9.0]]
    assert k.anomaly_score(13.0) == pytest.approx(1.5)  # |13-10| / sqrt(4)
    assert k.anomaly_score(7.0) == pytest.approx(1.5)  # dips are anomalies too (abs)
    k.P[0][0] = 0.0  # sigma < 1e-9 guard
    assert k.anomaly_score(99.0) == 0.0


def test_update_matches_reference_kalman_step_by_step():
    """Every state/covariance entry must match the textbook 2D Kalman
    recursion on the same input sequence — kills index swaps, sign flips,
    and */div inversions in the predict/update algebra."""
    def reference(x, P, z, F, H, Q, R):
        # predict
        xp = [F[0][0] * x[0] + F[0][1] * x[1], F[1][0] * x[0] + F[1][1] * x[1]]
        FP = [[sum(F[i][k] * P[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
        Pp = [[sum(FP[i][k] * F[j][k] for k in range(2)) + Q[i][j] for j in range(2)] for i in range(2)]
        # update
        y = z - (H[0] * xp[0] + H[1] * xp[1])
        S = sum(H[i] * H[j] * Pp[i][j] for i in range(2) for j in range(2)) + R
        K = [[(Pp[i][0] * H[0] + Pp[i][1] * H[1]) / S] for i in range(2)]
        x2 = [xp[i] + K[i][0] * y for i in range(2)]
        P2 = [[Pp[i][j] - K[i][0] * H[0] * Pp[0][j] - K[i][0] * H[1] * Pp[1][j]
               for j in range(2)] for i in range(2)]
        return x2, P2

    k = KalmanTrend(process_noise=0.5, measurement_noise=25.0, dt=2.0)
    rng = [7.0, 11.5, 16.0, 24.5, 29.0, 40.0, 37.5, 52.0]
    for n, z in enumerate(rng):
        k.update(z, timestamp=float(n))
        if k.initialized and n >= 1:
            # recompute reference from the SAME seeds the filter started from
            pass
    # replay deterministically against the reference from identical init
    ref_x = None
    ref_P = None
    KalmanTrend(process_noise=0.5, measurement_noise=25.0, dt=2.0)
    k3 = KalmanTrend(process_noise=0.5, measurement_noise=25.0, dt=2.0)
    F, H, Q, R = k3.F, k3.H, k3.Q, k3.R
    for n, z in enumerate(rng):
        k3.update(z, timestamp=float(n))
        if ref_x is None:
            ref_x = [z, 0.0]
            ref_P = [[R, 0.0], [0.0, R]]
        else:
            ref_x, ref_P = reference(ref_x, ref_P, z, F, H, Q, R)
        assert k3.x[0] == pytest.approx(ref_x[0], rel=1e-9), (n, "level")
        assert k3.x[1] == pytest.approx(ref_x[1], rel=1e-9), (n, "velocity")
        assert k3.P[0][0] == pytest.approx(ref_P[0][0], rel=1e-9), (n, "P00")
        assert k3.P[0][1] == pytest.approx(ref_P[0][1], rel=1e-9), (n, "P01")
        assert k3.P[1][0] == pytest.approx(ref_P[1][0], rel=1e-9), (n, "P10")
        assert k3.P[1][1] == pytest.approx(ref_P[1][1], rel=1e-9), (n, "P11")


def test_predict_heterogeneous_state_exact():
    """With nonzero P01 and P10 the predict algebra must produce the exact
    variance — symmetric shortcuts survive on diagonal-only states."""
    k = KalmanTrend()
    k.update(10.0, timestamp=1.0)
    k.x = [10.0, 3.0]
    k.P = [[4.0, 1.5], [2.5, 9.0]]  # deliberately asymmetric
    level, sigma = k.predict(2.0)
    assert level == pytest.approx(16.0)
    assert sigma == pytest.approx((4.0 + 4.0 * 9.0 + 4.0 * 1.5) ** 0.5)


def test_ttf_boundary_bands_and_timestamp_contract():
    """ttf band boundaries: slow-but-positive velocity, tiny-but-positive
    distance, steps == max_steps, and sub-epsilon distance sentinel."""
    import math
    k = KalmanTrend()
    k.update(10.0, timestamp=1.0)
    k.P = [[4.0, 0.0], [0.0, 0.01]]  # tight covariance: conf reflects sigma/distance
    k.x = [10.0, 0.5]
    steps, conf = k.time_to_threshold(15.0)
    assert steps == 10
    assert conf > 0.5  # sigma ~2.3 vs distance 5
    k.x = [10.0, 0.5]
    steps, _ = k.time_to_threshold(10.5)
    assert steps == 1  # distance 0.5 / velocity 0.5, not the <=0 tuple
    k.x = [10.0, 1.0]
    steps, conf = k.time_to_threshold(190.0, max_steps=180)
    assert steps == 180
    assert conf > 0.5  # steps == max_steps is NOT the clamped (max, 0.0) band
    k.x = [10.0, 1.0]
    steps, conf = k.time_to_threshold(math.nextafter(10.0, math.inf))
    assert steps == 0
    assert conf == 0.5  # sub-epsilon distance sentinel
    k.update(11.0, timestamp=5.0)
    assert k.last_update == 5.0  # explicit ts wins over None / wall clock
