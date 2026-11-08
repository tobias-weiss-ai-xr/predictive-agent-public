"""Bayesian risk scoring combining multiple signals."""

from predictive_agent.config import PREDICTION_BASE_RISK


def calculate_risk(pod_metrics, markov_state, markov_p_critical, markov_p_failed):
    """Calculate Bayesian risk score (0.0 to 1.0).

    Combines:
    - Memory percentage
    - Memory trend (via time-to-failure)
    - CPU percentage
    - CPU trend (millicores/min)
    - Restart rate
    - Log error rate
    - Node pressure (memory/disk)
    - Markov state + transition probabilities
    - Anomaly scores (memory and CPU, from Kalman innovation)
    - Blast radius (from knowledge graph, 0=not connected)
    - Pod phase (Pending, Failed, Unknown)
    - Container status (not ready, CrashLoopBackOff, CreateContainerConfigError, etc.)
    - Container termination (OOMKilled, Error)

    Args:
        pod_metrics: dict with keys like memory_pct, memory_trend_mib_per_min,
                     cpu_pct, cpu_trend_m_per_min, restart_rate_per_hr,
                     log_error_rate_per_min, node_memory_pressure,
                     node_disk_pressure, memory_limit_mib, memory_mib,
                     memory_anomaly_score, cpu_anomaly_score, blast_radius,
                     pod_phase, container_ready, wait_state, terminated,
                     terminated_reason, pod_scheduled
        markov_state: current Markov state string
        markov_p_critical: probability of transitioning to CRITICAL
        markov_p_failed: probability of transitioning to FAILED

    Returns:
        float: risk score 0.0 to 0.99
    """
    prior = PREDICTION_BASE_RISK  # base rate (~15-20% of pods experience issues per cycle based on backtest calibration)
    lr = 1.0  # likelihood ratio multiplier

    mem_pct = pod_metrics.get("memory_pct", 0)
    mem_trend = pod_metrics.get("memory_trend_mib_per_min", 0)
    cpu_pct = pod_metrics.get("cpu_pct", 0)
    cpu_trend = pod_metrics.get("cpu_trend_m_per_min", 0)
    restart_rate = pod_metrics.get("restart_rate_per_hr", 0)
    log_error_rate = pod_metrics.get("log_error_rate_per_min", 0)
    node_mem_pressure = pod_metrics.get("node_memory_pressure", False)
    node_disk_pressure = pod_metrics.get("node_disk_pressure", False)
    mem_anomaly = pod_metrics.get("memory_anomaly_score", 0)
    cpu_anomaly = pod_metrics.get("cpu_anomaly_score", 0)
    blast_radius = pod_metrics.get("blast_radius", 0)

    # Memory percentage
    if mem_pct > 95:
        lr *= 12.0
    elif mem_pct > 85:
        lr *= 6.0
    elif mem_pct > 70:
        lr *= 2.5

    # Memory trend + time to OOM
    # Evaluate TTF regardless of current memory % — rapid growth toward
    # limit is dangerous even when current usage is low (e.g. 3% with
    # 5 MiB/min trend → OOM in 20 minutes).
    if mem_trend > 0 and pod_metrics.get("memory_limit_mib", 0) > 0:
        remaining = pod_metrics["memory_limit_mib"] - pod_metrics.get("memory_mib", 0)
        if remaining > 0 and mem_trend > 0:
            ttf = remaining / mem_trend  # minutes
            if ttf < 5:
                lr *= 8.0
            elif ttf < 10:
                lr *= 4.0
            elif ttf < 30:
                lr *= 2.0
            elif ttf < 60 and mem_pct > 30:
                # Only escalate for moderate TTF if memory is already notable
                lr *= 1.5

    # CPU percentage
    if cpu_pct > 95:
        lr *= 3.0
    elif cpu_pct > 80:
        lr *= 1.5

    # CPU trend (rapid CPU growth is a warning sign)
    if cpu_trend > 50:  # >50 millicores/min sustained growth
        lr *= 2.0
    elif cpu_trend > 20:
        lr *= 1.3

    # Restart rate (any restart is a strong signal — backtester shows
    # restart_count > 0 correlates with actual failures)
    if restart_rate > 5:
        lr *= 15.0
    elif restart_rate > 3:
        lr *= 10.0
    elif restart_rate > 1:
        lr *= 6.0
    elif restart_rate > 0:
        lr *= 3.0

    # Log errors
    if log_error_rate > 10:
        lr *= 4.0
    elif log_error_rate > 5:
        lr *= 2.5
    elif log_error_rate > 0:
        lr *= 1.5

    # Node pressure
    if node_mem_pressure:
        lr *= 4.0
    if node_disk_pressure:
        lr *= 6.0

    # Markov state
    if markov_state == "CRITICAL":
        lr *= 20.0
    elif markov_state == "STRESSED":
        lr *= 3.0
    elif markov_state == "DEGRADED":
        lr *= 1.5

    # Markov prediction
    lr *= 1 + markov_p_critical * 5 + markov_p_failed * 10

    # Anomaly detection (3-sigma spikes from Kalman filter)
    if mem_anomaly > 3:
        lr *= 3.0
    elif mem_anomaly > 2:
        lr *= 1.5
    if cpu_anomaly > 3:
        lr *= 2.0
    elif cpu_anomaly > 2:
        lr *= 1.3

    # Blast radius (knowledge graph impact)
    # High blast radius means many pods depend on this one — escalate risk
    if blast_radius > 20:
        lr *= 3.0
    elif blast_radius > 10:
        lr *= 2.0
    elif blast_radius > 5:
        lr *= 1.5

    # ─── Pod phase / container status signals ─────────────────────────
    # These capture failures that CPU/memory/restart signals miss:
    # CrashLoopBackOff, CreateContainerConfigError, Pending, etc.
    pod_phase = pod_metrics.get("pod_phase", "Running")
    container_ready = pod_metrics.get("container_ready", True)
    wait_state = pod_metrics.get("wait_state", None)
    terminated = pod_metrics.get("terminated", False)
    terminated_reason = pod_metrics.get("terminated_reason", None)
    pod_scheduled = pod_metrics.get("pod_scheduled", True)

    # Pod phase: non-Running phases are strong risk signals
    if pod_phase == "Failed":
        lr *= 20.0
    elif pod_phase == "Unknown":
        lr *= 10.0
    elif pod_phase == "Pending":
        lr *= 3.0  # Can't schedule — likely resource/dependency issue
    elif pod_phase == "Succeeded":
        lr *= 0.1  # Completed successfully — very low risk

    # Pod not scheduled — stuck in scheduling
    if not pod_scheduled:
        lr *= 5.0

    # Container not ready — strong signal something is wrong
    if not container_ready:
        lr *= 5.0

    # Container waiting states (CrashLoopBackOff, CreateContainerConfigError, etc.)
    if wait_state:
        if wait_state == "CrashLoopBackOff":
            lr *= 15.0
        elif wait_state in ("CreateContainerConfigError", "CreateContainerError"):
            lr *= 10.0
        elif wait_state in ("ImagePullBackOff", "ErrImagePull") or wait_state == "InvalidImageName":
            lr *= 8.0
        elif wait_state == "RunContainerError":
            lr *= 10.0
        elif wait_state in ("ContainerCreating", "PodInitializing"):
            lr *= 2.0  # Normal during startup, but if persistent, risky
        else:
            lr *= 3.0  # Unknown waiting state

    # Container terminated — check termination reason
    if terminated:
        if terminated_reason == "OOMKilled":
            lr *= 12.0
        elif terminated_reason == "Error":
            lr *= 5.0
        elif terminated_reason == "Completed":
            lr *= 0.1  # Normal completion
        elif terminated_reason:
            lr *= 3.0  # Other termination reason

    # Bayesian update
    posterior = (prior * lr) / (prior * lr + (1 - prior))
    return min(posterior, 0.99)
