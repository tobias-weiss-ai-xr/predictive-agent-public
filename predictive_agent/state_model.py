"""Pod state tracking with Kalman filters and state classification."""

import os
import threading
import time
from datetime import datetime, timezone

from predictive_agent.kalman import KalmanTrend
from predictive_agent.markov import MarkovChain
from predictive_agent.seasonal import HoltWinters

# Per-metric Kalman parameters (configurable via environment)
# Memory changes gradually — low process noise, moderate measurement noise
_KALMAN_MEM_PROCESS_NOISE = float(os.environ.get("KALMAN_MEMORY_PROCESS_NOISE", "0.5"))
_KALMAN_MEM_MEASUREMENT_NOISE = float(os.environ.get("KALMAN_MEMORY_MEASUREMENT_NOISE", "50.0"))
# CPU is noisier — higher process noise to track spikes, higher measurement noise
_KALMAN_CPU_PROCESS_NOISE = float(os.environ.get("KALMAN_CPU_PROCESS_NOISE", "5.0"))
_KALMAN_CPU_MEASUREMENT_NOISE = float(os.environ.get("KALMAN_CPU_MEASUREMENT_NOISE", "200.0"))

# Seasonal (Holt-Winters) trend alongside the linear Kalman velocity.
# Modes:
#   off    (default) — no seasonal state is kept; memory_trend/cpu_trend are
#          exactly the Kalman velocities (unchanged behaviour).
#   shadow — compute the seasonal signal + model-selection stats but keep
#          returning Kalman trends; /state & /metrics expose the would-be
#          seasonal values so the improvement can be validated before use.
#   on     — memory_trend/cpu_trend return the seasonal expected velocity
#          once the per-metric model is fitted and confident (threshold
#          SEASONAL_CONF_TREND), falling back to Kalman otherwise.
_SEASONAL_TREND_MODE = os.environ.get("SEASONAL_TREND_MODE", "off").lower()
_SEASONAL_PERIOD = int(os.environ.get("SEASONAL_PERIOD_SAMPLES", "1440"))  # 24h @ 60s
# Minimum seasonal confidence before shadow/on mode trusts the seasonal
# expected velocity over the linear Kalman velocity.
_SEASONAL_CONF_TREND = float(os.environ.get("SEASONAL_CONF_TREND", "0.3"))
# Consecutive agreeing classifications required to commit a state change.
# Damps boundary jitter (e.g. a pod hovering at the DEGRADED threshold
# flapping every reconcile cycle). 1 = old instantaneous behavior.
_STATE_HYSTERESIS = int(os.environ.get("STATE_HYSTERESIS", "2"))


def classify_state(memory_pct, cpu_pct, restart_rate, log_errors, node_pressure,
                   markov_state, cpu_limit_known=True):
    """Classify pod state based on current metrics.

    Args:
        memory_pct: Memory usage percentage
        cpu_pct: CPU usage percentage
        restart_rate: Pod restarts per hour
        log_errors: Log error count
        node_pressure: Whether node is under memory/disk pressure
        markov_state: Current Markov chain state
        cpu_limit_known: False when no container declares a CPU limit, in which
            case cpu_pct is measured against a synthetic 1000m scale and must
            not be read as saturation (a burst is legal without a limit).

    Returns:
        str: One of HEALTHY, DEGRADED, STRESSED, CRITICAL
    """
    score = 0

    # Memory scoring
    if memory_pct > 95:
        score += 4
    elif memory_pct > 85:
        score += 3
    elif memory_pct > 70:
        score += 2
    elif memory_pct > 50:
        score += 1

    # CPU scoring. Only when a real CPU limit exists: without one there is no
    # ceiling to saturate (cpu_pct then uses the synthetic 1000m scale), and an
    # unlimited pod on a big node routinely and legally exceeds one core.
    if cpu_limit_known:
        if cpu_pct > 95:
            score += 4
        elif cpu_pct > 80:
            score += 3
        elif cpu_pct > 60:
            score += 2
        elif cpu_pct > 40:
            score += 1

    # Restart scoring. restart_rate = restart_count / pod_age_hours, which
    # decays toward 0 but never reaches it, so the old `> 0` branch parked a
    # permanent +1 on any pod that ever restarted (stale-signal trap, observed
    # 2026-11-24 with csi-nfs 26 restarts/66d). High tiers still penalize
    # genuine crash-loops; only the stale tail (<= 0.1/hr, 1 restart per 10h)
    # is allowed to decay naturally with age.
    if restart_rate > 5:
        score += 4
    elif restart_rate > 3:
        score += 3
    elif restart_rate > 1:
        score += 2
    elif restart_rate > 0.1:
        score += 1

    # Log errors scoring
    if log_errors > 10:
        score += 4
    elif log_errors > 5:
        score += 3
    elif log_errors > 2:
        score += 2
    elif log_errors > 0:
        score += 1

    # Node pressure scoring
    if node_pressure:
        score += 3

    # Markov state influence
    markov_scores = {
        "HEALTHY": 0,
        "DEGRADED": 1,
        "STRESSED": 2,
        "CRITICAL": 3,
    }
    score += markov_scores.get(markov_state, 0)

    # Determine state
    if score >= 10:
        return "CRITICAL"
    elif score >= 6:
        return "STRESSED"
    elif score >= 3:
        return "DEGRADED"
    else:
        return "HEALTHY"


class PodTracker:
    """Track state of a single pod using Kalman filters."""

    def __init__(self, namespace, name, kalman_memory=None, kalman_cpu=None, node_name="", hysteresis=None, seasonal_mode=None, seasonal_period=None):
        self.namespace = namespace
        self.name = name
        self.pod_key = f"{namespace}/{name}"
        self.node_name = node_name
        self.state = "HEALTHY"
        self.prev_state = "HEALTHY"
        self.hysteresis = _STATE_HYSTERESIS if hysteresis is None else max(1, hysteresis)
        self._pending_state = None
        self._pending_count = 0
        self.kalman_memory = kalman_memory or KalmanTrend(
            process_noise=_KALMAN_MEM_PROCESS_NOISE,
            measurement_noise=_KALMAN_MEM_MEASUREMENT_NOISE,
        )
        self.kalman_cpu = kalman_cpu or KalmanTrend(
            process_noise=_KALMAN_CPU_PROCESS_NOISE,
            measurement_noise=_KALMAN_CPU_MEASUREMENT_NOISE,
        )
        # Seasonal (Holt-Winters) companions — only fed when the mode is not
        # "off" (see _SEASONAL_TREND_MODE). Not persisted: they self-warm from
        # observed samples across restart/shadow transitions and the state file
        # stays backward-compatible.
        _sper = _SEASONAL_PERIOD if seasonal_period is None else seasonal_period
        self.seasonal_memory = HoltWinters(season_len=_sper)
        self.seasonal_cpu = HoltWinters(season_len=_sper)
        self.seasonal_mode = _SEASONAL_TREND_MODE if seasonal_mode is None else seasonal_mode
        # Last fitted confidence + seasonal expected velocities, exposed for
        # observability in shadow mode and used for selection in "on" mode.
        self.seasonal_mem_conf = 0.0
        self.seasonal_cpu_conf = 0.0
        self.seasonal_mem_vel = 0.0
        self.seasonal_cpu_vel = 0.0
        self.memory_pct = 0.0
        self.cpu_pct = 0.0
        self.memory_mib = 0
        self.memory_limit_mib = 0
        self.cpu_m = 0
        self.cpu_limit_m = 1000
        # False when no container in the pod declares a CPU limit. cpu_limit_m
        # then keeps its 1000m default purely as a numeric scale; consumers
        # must not treat cpu_pct as limit-relative saturation.
        self.cpu_limit_known = True
        self.restart_count = 0
        # Restarts per hour (restart_count / pod age). The raw counter is
        # cumulative for the pod's whole lifetime and never resets, so any
        # consumer asking "is this pod restarting?"/"no recent restarts" must
        # use this rate instead of restart_count > 0 — otherwise every pod that
        # ever restarted is treated as failing forever.
        self.restart_rate = 0.0
        self.log_errors = 0
        self.node_pressure = False
        self.node_disk_pressure = False
        # Closed loop (canary widening): scale of the workload this tracker's
        # action targets (deployment replicas when known, else 1). Canary-capable
        # actions only run two-leg (slice → verify → widen) when replicas > 1.
        # Fed by deployment discovery when available; single-replica targets
        # skip the canary leg entirely.
        self.replicas = 1
        self.memory_anomaly_score = 0.0
        self.cpu_anomaly_score = 0.0
        self._created_at = None  # set on first update

    def update(self, memory_mib, memory_limit_mib, cpu_m, restart_count, log_errors, node_pressure, node_disk_pressure=False, cpu_limit_m=1000):
        """Update pod metrics and Kalman filters.

        cpu_limit_m: summed container CPU limit in millicores (default 1000 =
            a single core) used to normalize cpu_pct; mirrors how memory_pct
            is normalized against memory_limit_mib.
        """
        if self._created_at is None:
            self._created_at = datetime.now(timezone.utc)
        self.memory_mib = memory_mib
        self.memory_limit_mib = memory_limit_mib
        self.cpu_m = cpu_m
        self.cpu_limit_known = bool(cpu_limit_m and cpu_limit_m > 0)
        self.cpu_limit_m = cpu_limit_m if self.cpu_limit_known else 1000
        self.restart_count = restart_count
        self.log_errors = log_errors
        self.node_pressure = node_pressure
        self.node_disk_pressure = node_disk_pressure

        # Update Kalman filters
        self.kalman_memory.update(memory_mib)
        self.kalman_cpu.update(cpu_m)

        # Seasonal companions follow the same observations (shadow/on modes).
        if self.seasonal_mode != "off":
            ts = time.time()
            self.seasonal_memory.observe(ts, float(memory_mib))
            self.seasonal_cpu.observe(ts, float(cpu_m))
            self.seasonal_mem_conf = self.seasonal_memory.confidence
            self.seasonal_cpu_conf = self.seasonal_cpu.confidence
            self.seasonal_mem_vel = self.seasonal_memory.expected_velocity()
            self.seasonal_cpu_vel = self.seasonal_cpu.expected_velocity()

        # Anomaly detection: 3-sigma innovation check
        self.memory_anomaly_score = self.kalman_memory.anomaly_score(memory_mib)
        self.cpu_anomaly_score = self.kalman_cpu.anomaly_score(cpu_m)

        # Calculate percentages
        if memory_limit_mib > 0:
            self.memory_pct = (memory_mib / memory_limit_mib) * 100
        else:
            self.memory_pct = 0.0

        # Estimate CPU percentage — normalized against the pod's summed CPU
        # limit (default 1000m = one core when unknown) so multi-core pods are
        # scored against their real capacity, not a single-core yardstick.
        lim_m = self.cpu_limit_m if self.cpu_limit_m and self.cpu_limit_m > 0 else 1000
        self.cpu_pct = (cpu_m / lim_m) * 100

        # Classify state
        # Calculate restart rate per hour from pod age
        age_s = (datetime.now(timezone.utc) - self._created_at).total_seconds() if self._created_at else 3600
        restart_rate = restart_count / max(age_s / 3600, 1.0)  # restarts per hour
        self.restart_rate = restart_rate

        self.prev_state = self.state
        # Debounced classification: commit a state change only after
        # `hysteresis` consecutive agreeing raw classifications. A single
        # boundary-crossing cycle (metric jitter) cannot flip the state.
        raw = classify_state(
            memory_pct=self.memory_pct,
            cpu_pct=self.cpu_pct,
            restart_rate=restart_rate,  # per hour, not raw count
            log_errors=log_errors,
            node_pressure=node_pressure,
            markov_state=self.state,
            cpu_limit_known=self.cpu_limit_known,
        )
        if raw == self.state:
            self._pending_state = None
            self._pending_count = 0
        elif raw == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = raw
            self._pending_count = 1
        if self._pending_state is not None and self._pending_count >= self.hysteresis:
            self.state = self._pending_state
            self._pending_state = None
            self._pending_count = 0



    def _seasonal_velocity(self, conf, seasonal_vel, kalman_vel):
        """Return the trend to expose for a metric given the mode.

        - off: always the Kalman velocity (unchanged behaviour).
        - shadow: seasonal computed but Kalman exposed (improvement visible in
          `seasonal_mem_vel`/`seasonal_cpu_vel` for validation).
        - on: seasonal velocity when the model is fitted and confident enough.
        """
        if self.seasonal_mode != "on":
            return kalman_vel
        if conf >= _SEASONAL_CONF_TREND:
            return seasonal_vel
        return kalman_vel

    @property
    def memory_trend(self):
        """Current memory trend in MiB/min."""
        return self._seasonal_velocity(
            self.seasonal_mem_conf, self.seasonal_mem_vel, self.kalman_memory.velocity
        )

    @property
    def cpu_trend(self):
        """Current CPU trend in millicores/min."""
        return self._seasonal_velocity(
            self.seasonal_cpu_conf, self.seasonal_cpu_vel, self.kalman_cpu.velocity
        )

    def time_to_failure(self, threshold_pct=95):
        """Estimate time to reach memory threshold.

        Args:
            threshold_pct: Memory percentage threshold for failure

        Returns:
            float or None: Estimated minutes to failure, or None if not approaching threshold
        """
        if self.memory_limit_mib <= 0:
            return None

        threshold_mib = self.memory_limit_mib * (threshold_pct / 100)
        remaining = threshold_mib - self.memory_mib

        if remaining <= 0:
            return 0.0

        velocity = self.kalman_memory.velocity
        if velocity <= 0:
            return None

        return remaining / velocity


class StateModel:
    """Global state model for all tracked pods."""

    def __init__(self):
        self.pods = {}
        self.markov = MarkovChain()
        self._lock = threading.Lock()

    def track_pod(self, namespace, name):
        """Start tracking a pod."""
        pod_key = f"{namespace}/{name}"
        with self._lock:
            if pod_key not in self.pods:
                self.pods[pod_key] = PodTracker(namespace, name)
            return self.pods[pod_key]

    def get_pods_snapshot(self):
        """Return a thread-safe snapshot copy of all tracked pods."""
        with self._lock:
            return dict(self.pods)

    def update_pod(self, namespace, name, memory_mib, memory_limit_mib, cpu_m,
                   restart_count, log_errors, node_pressure, node_disk_pressure=False,
                   node_name="", created_at=None, cpu_limit_m=1000):
        """Update pod metrics and return the tracker.

        Records Markov chain state transitions when the pod's state changes.
        Args:
            created_at: Optional pod creation timestamp (datetime or ISO string).
                       Used to compute accurate restart rate.
        """
        tracker = self.track_pod(namespace, name)
        tracker.node_name = node_name
        if created_at is not None:
            if isinstance(created_at, str):
                try:
                    tracker._created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except ValueError:
                    pass
            else:
                tracker._created_at = created_at
        prev_state = tracker.state
        tracker.update(
            memory_mib=memory_mib,
            memory_limit_mib=memory_limit_mib,
            cpu_m=cpu_m,
            restart_count=restart_count,
            log_errors=log_errors,
            node_pressure=node_pressure,
            node_disk_pressure=node_disk_pressure,
            cpu_limit_m=cpu_limit_m,
        )
        # Record Markov transition when state changes
        if tracker.state != prev_state:
            self.markov.record_transition(prev_state, tracker.state)
        return tracker

    def to_dict(self):
        """Serialize state model to dict."""
        return {
            "pods": {key: {
                "namespace": pod.namespace,
                "name": pod.name,
                "state": pod.state,
                "kalman_memory": pod.kalman_memory.__dict__,
                "kalman_cpu": pod.kalman_cpu.__dict__,
                "memory_pct": pod.memory_pct,
                "cpu_pct": pod.cpu_pct,
                "memory_mib": pod.memory_mib,
                "memory_limit_mib": pod.memory_limit_mib,
                "cpu_m": pod.cpu_m,
                "cpu_limit_m": pod.cpu_limit_m,
                "cpu_limit_known": pod.cpu_limit_known,
                "restart_count": pod.restart_count,
                "log_errors": pod.log_errors,
                "node_pressure": pod.node_pressure,
                "node_disk_pressure": pod.node_disk_pressure,
                "memory_anomaly_score": pod.memory_anomaly_score,
                "cpu_anomaly_score": pod.cpu_anomaly_score,
                "node_name": pod.node_name,
                "seasonal": {
                    "mode": pod.seasonal_mode,
                    "mem_conf": pod.seasonal_mem_conf,
                    "cpu_conf": pod.seasonal_cpu_conf,
                    "mem_vel": pod.seasonal_mem_vel,
                    "cpu_vel": pod.seasonal_cpu_vel,
                } if pod.seasonal_mode != "off" else None,
                "created_at": pod._created_at.isoformat() if pod._created_at else None,
            } for key, pod in self.pods.items()},
            "markov": self.markov.to_dict() if self.markov else None,
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize state model from dict."""
        model = cls()
        if "pods" in data:
            for key, pod_data in data["pods"].items():
                tracker = PodTracker(pod_data["namespace"], pod_data["name"])
                tracker.state = pod_data["state"]
                tracker.memory_pct = pod_data["memory_pct"]
                tracker.cpu_pct = pod_data["cpu_pct"]
                tracker.memory_mib = pod_data["memory_mib"]
                tracker.memory_limit_mib = pod_data["memory_limit_mib"]
                tracker.cpu_m = pod_data["cpu_m"]
                tracker.cpu_limit_m = pod_data.get("cpu_limit_m", 1000)
                # Default True: state files written before the flag existed
                # only ever contained real (or defaulted) limits.
                tracker.cpu_limit_known = pod_data.get("cpu_limit_known", True)
                tracker.restart_count = pod_data["restart_count"]
                tracker.log_errors = pod_data["log_errors"]
                tracker.node_pressure = pod_data["node_pressure"]
                tracker.node_disk_pressure = pod_data.get("node_disk_pressure", False)
                tracker.memory_anomaly_score = pod_data.get("memory_anomaly_score", 0.0)
                tracker.cpu_anomaly_score = pod_data.get("cpu_anomaly_score", 0.0)
                tracker.node_name = pod_data.get("node_name", "")
                created_at = pod_data.get("created_at")
                if created_at:
                    try:
                        tracker._created_at = datetime.fromisoformat(created_at)
                    except ValueError:
                        # Unparseable timestamp → leave None (age defaults to 3600s)
                        tracker._created_at = None

                # Restore Kalman filters
                tracker.kalman_memory.__dict__ = pod_data["kalman_memory"]
                tracker.kalman_cpu.__dict__ = pod_data["kalman_cpu"]

                model.pods[key] = tracker

        if "markov" in data and data["markov"]:
            from predictive_agent.markov import MarkovChain
            model.markov = MarkovChain.from_dict(data["markov"])

        return model
