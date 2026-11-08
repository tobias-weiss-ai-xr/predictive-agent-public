"""Kalman filter for trend estimation (2D: level + velocity)."""

import time


class KalmanTrend:
    """2D Kalman filter for trend estimation.

    State vector: [level, velocity]
    Observation: [level]
    """

    def __init__(self, process_noise=1.0, measurement_noise=100.0, dt=1.0):
        self.x = [0.0, 0.0]  # [level, velocity]
        self.P = [[1000.0, 0.0], [0.0, 1000.0]]  # Covariance
        self.F = [[1.0, dt], [0.0, 1.0]]  # State transition
        self.H = [1.0, 0.0]  # Observation matrix
        self.Q = [[process_noise * dt, 0.0], [0.0, process_noise * dt]]  # Process noise
        self.R = measurement_noise  # Measurement noise
        self.initialized = False
        self.last_update = 0.0

    def update(self, measurement, timestamp=None):
        """Update Kalman filter with new measurement."""
        if not self.initialized:
            self.x = [measurement, 0.0]
            self.P = [[self.R, 0.0], [0.0, self.R]]
            self.initialized = True
            self.last_update = timestamp or time.time()
            return

        # Predict
        x_pred = [
            self.F[0][0] * self.x[0] + self.F[0][1] * self.x[1],
            self.F[1][0] * self.x[0] + self.F[1][1] * self.x[1],
        ]

        # P_pred = F @ P @ F.T + Q
        FP00 = self.F[0][0] * self.P[0][0] + self.F[0][1] * self.P[1][0]
        FP01 = self.F[0][0] * self.P[0][1] + self.F[0][1] * self.P[1][1]
        FP10 = self.F[1][0] * self.P[0][0] + self.F[1][1] * self.P[1][0]
        FP11 = self.F[1][0] * self.P[0][1] + self.F[1][1] * self.P[1][1]

        P00 = FP00 * self.F[0][0] + FP01 * self.F[0][1] + self.Q[0][0]
        P01 = FP00 * self.F[1][0] + FP01 * self.F[1][1] + self.Q[0][1]
        P10 = FP10 * self.F[0][0] + FP11 * self.F[0][1] + self.Q[1][0]
        P11 = FP10 * self.F[1][0] + FP11 * self.F[1][1] + self.Q[1][1]

        # Innovation
        y = measurement - (self.H[0] * x_pred[0] + self.H[1] * x_pred[1])

        # Innovation covariance
        S = (
            P00 * self.H[0] * self.H[0]
            + P01 * self.H[0] * self.H[1]
            + P10 * self.H[1] * self.H[0]
            + P11 * self.H[1] * self.H[1]
            + self.R
        )

        # Kalman gain
        K0 = (P00 * self.H[0] + P01 * self.H[1]) / S
        K1 = (P10 * self.H[0] + P11 * self.H[1]) / S

        # Update state
        self.x = [x_pred[0] + K0 * y, x_pred[1] + K1 * y]

        # Update covariance
        self.P = [
            [
                P00 - K0 * self.H[0] * P00 - K0 * self.H[1] * P10,
                P01 - K0 * self.H[0] * P01 - K0 * self.H[1] * P11,
            ],
            [
                P10 - K1 * self.H[0] * P00 - K1 * self.H[1] * P10,
                P11 - K1 * self.H[0] * P01 - K1 * self.H[1] * P11,
            ],
        ]

        self.last_update = timestamp or time.time()

    @property
    def level(self):
        """Current level estimate."""
        return self.x[0]

    @property
    def velocity(self):
        """Current velocity estimate."""
        return self.x[1]

    @property
    def level_uncertainty(self):
        """Level uncertainty (standard deviation)."""
        return self.P[0][0] ** 0.5

    @property
    def velocity_uncertainty(self):
        """Velocity uncertainty (standard deviation)."""
        return self.P[1][1] ** 0.5

    def predict(self, steps=1.0):
        """Predict state at future time. Returns (predicted_level, sigma)."""
        pred_level = self.x[0] + self.x[1] * steps
        pred_var = self.P[0][0] + steps * steps * self.P[1][1] + 2 * steps * self.P[0][1]
        pred_sigma = pred_var ** 0.5 if pred_var > 0 else 0.0
        return pred_level, pred_sigma

    def time_to_threshold(self, threshold, max_steps=180):
        """Estimate time (steps) to reach threshold. Returns (steps, confidence).

        Confidence reflects how certain we are in the TTF estimate, based on
        prediction uncertainty relative to the distance to threshold.
        High confidence: prediction uncertainty is small compared to distance.
        Low confidence: prediction uncertainty is large compared to distance.
        """
        if self.velocity <= 0:
            return max_steps, 0.0

        distance = threshold - self.x[0]
        if distance <= 0:
            return 0, 1.0  # Already at or above threshold

        steps = distance / self.velocity
        if steps > max_steps:
            return max_steps, 0.0

        _, pred_sigma = self.predict(steps)
        # Confidence: high when prediction uncertainty is small relative to distance.
        # Low when uncertainty is large relative to distance.
        if distance > 1e-9:
            confidence = max(0.0, min(1.0, 1.0 - (pred_sigma / distance)))
        else:
            confidence = 0.5

        return int(steps), confidence

    def anomaly_score(self, measurement):
        """Return anomaly score for a measurement (0 = normal, >3 = anomalous).

        Uses the innovation (measurement - predicted level) normalized by
        the level uncertainty (sigma). A score > 3 indicates a 3-sigma anomaly
        (sudden spike or dip that the filter did not expect).
        """
        if not self.initialized:
            return 0.0
        innovation = abs(measurement - self.level)
        sigma = self.level_uncertainty
        if sigma < 1e-9:
            return 0.0
        return innovation / sigma
