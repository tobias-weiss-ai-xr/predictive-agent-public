"""Metrics collection from kubectl (top pods, get pods, logs, node conditions)."""

import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

# ─── Collector State Management (P4: Degraded-Mode Protocol) ──────────────
# Collector state: HEALTHY, DEGRADED, or DOWN
_COLLECTOR_STATE = "HEALTHY"
_HAS_PRIOR_STATE = False
_LAST_COLLECT_TIME: Optional[float] = None

COLLECTOR_STATE_HEALTHY = "HEALTHY"
COLLECTOR_STATE_DEGRADED = "DEGRADED"
COLLECTOR_STATE_DOWN = "DOWN"

COLLECTOR_STATE_NUMERIC = {
    COLLECTOR_STATE_DOWN: 0,
    COLLECTOR_STATE_DEGRADED: 1,
    COLLECTOR_STATE_HEALTHY: 2,
}


def get_collector_state() -> str:
    """Get current collector state."""
    return _COLLECTOR_STATE


def get_collector_state_numeric() -> int:
    """Get numeric representation for Prometheus gauge."""
    return COLLECTOR_STATE_NUMERIC.get(_COLLECTOR_STATE, 0)


def has_prior_state() -> bool:
    """Check if we have ever successfully collected data."""
    return _HAS_PRIOR_STATE


def record_successful_collection() -> None:
    """Record successful collection - transition to HEALTHY."""
    global _COLLECTOR_STATE, _HAS_PRIOR_STATE, _LAST_COLLECT_TIME
    _COLLECTOR_STATE = COLLECTOR_STATE_HEALTHY
    _HAS_PRIOR_STATE = True
    _LAST_COLLECT_TIME = time.time()


def record_collection_failure() -> None:
    """Record collection failure - transition to DEGRADED or DOWN."""
    global _COLLECTOR_STATE
    if _HAS_PRIOR_STATE:
        _COLLECTOR_STATE = COLLECTOR_STATE_DEGRADED
    else:
        _COLLECTOR_STATE = COLLECTOR_STATE_DOWN


def set_collector_state(state: str, has_prior: bool = False, last_collect: Optional[float] = None) -> None:
    """Set the collector state directly.
    
    Args:
        state: One of COLLECTOR_STATE_HEALTHY, COLLECTOR_STATE_DEGRADED, COLLECTOR_STATE_DOWN
        has_prior: Whether we have prior state
        last_collect: Timestamp of last successful collection
    """
    global _COLLECTOR_STATE, _HAS_PRIOR_STATE, _LAST_COLLECT_TIME
    if state in COLLECTOR_STATE_NUMERIC:
        _COLLECTOR_STATE = state
    else:
        _COLLECTOR_STATE = COLLECTOR_STATE_HEALTHY
    _HAS_PRIOR_STATE = has_prior
    if last_collect is not None:
        _LAST_COLLECT_TIME = last_collect


# Pre-compiled regex for error detection (much faster than 10 separate re.search calls)
_ERROR_RE = re.compile(
    r"\b(?:ERROR|Error|FATAL|PANIC|OOM|CrashLoopBackOff|Exception|Traceback)\b"
    r"|\bpanic:|\bfatal:"
)

# Cache for kubectl command results (to avoid repeated calls)
_kubectl_cache: Dict[str, Tuple[Any, float]] = {}
_kubectl_cache_lock = None  # Will be initialized when cache is used

# Statistics for cache performance
_kubectl_cache_hits = 0
_kubectl_cache_misses = 0


def _get_cache_lock():
    """Lazy initialization of cache lock."""
    global _kubectl_cache_lock
    if _kubectl_cache_lock is None:
        import threading
        _kubectl_cache_lock = threading.Lock()
    return _kubectl_cache_lock


def _get_kubectl_cache(key: str, ttl: float = 30.0) -> Tuple[bool, Any]:
    """Get a value from the kubectl cache if it exists and is not expired."""
    global _kubectl_cache_hits, _kubectl_cache_misses

    lock = _get_cache_lock()
    with lock:
        if key in _kubectl_cache:
            value, timestamp = _kubectl_cache[key]
            if time.time() - timestamp < ttl:
                _kubectl_cache_hits += 1
                return True, value
            else:
                # Expired
                del _kubectl_cache[key]
        _kubectl_cache_misses += 1
        return False, None


def _set_kubectl_cache(key: str, value: Any) -> None:
    """Set a value in the kubectl cache."""
    lock = _get_cache_lock()
    with lock:
        _kubectl_cache[key] = (value, time.time())


def get_kubectl_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    total = _kubectl_cache_hits + _kubectl_cache_misses
    hit_rate = _kubectl_cache_hits / total if total > 0 else 0.0
    return {
        'hits': _kubectl_cache_hits,
        'misses': _kubectl_cache_misses,
        'hit_rate': f"{hit_rate:.2%}",
        'size': len(_kubectl_cache),
    }


def clear_kubectl_cache() -> None:
    """Clear the kubectl cache."""
    global _kubectl_cache_hits, _kubectl_cache_misses, _kubectl_cache
    lock = _get_cache_lock()
    with lock:
        _kubectl_cache.clear()
        _kubectl_cache_hits = 0
        _kubectl_cache_misses = 0


def _normalize_created_at(value):
    """Normalize a docker/k8s creation timestamp to an ISO 8601 UTC string.

    Accepts python isoformat ('2026-08-21T23:03:39+00:00'), RFC3339 ending in
    'Z' ('2026-08-21T23:03:39Z'), and the Go time.String() format emitted by
    'docker ps' CreatedAt ('2026-08-21 23:03:39 +0200 CEST'). Returns '' if the
    value cannot be parsed so callers never crash on a malformed timestamp.
    """
    if not value:
        return ""
    v = str(value).strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(v).astimezone(timezone.utc).isoformat()
    except ValueError:
        pass
    # Go time.String(): "2026-08-21 23:03:39 +0200 CEST"
    try:
        parts = v.split()
        if len(parts) >= 3 and parts[0].count("-") == 2 and parts[2][0] in "+-0123456789":
            dt = datetime.strptime(
                f"{parts[0]} {parts[1]} {parts[2]}", "%Y-%m-%d %H:%M:%S %z"
            )
            return dt.astimezone(timezone.utc).isoformat()
    except (ValueError, IndexError):
        pass
    return ""


def run_cmd(cmd, timeout=30, use_cache: bool = True, cache_ttl: float = 30.0) -> Tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr).
    
    Args:
        cmd: Command to run (list or string)
        timeout: Command timeout in seconds
        use_cache: Whether to use caching for this command
        cache_ttl: Cache TTL in seconds for this command
        
    Returns:
        Tuple of (returncode, stdout, stderr)
    """
    # Convert list to string for hashing
    if isinstance(cmd, list):
        cmd_str = ' '.join(cmd)
        cmd_for_hash = cmd_str
    else:
        cmd_str = cmd
        cmd_for_hash = cmd

    # Try to get from cache
    if use_cache:
        cache_key = hashlib.sha256(cmd_for_hash.encode()).hexdigest()
        hit, cached_result = _get_kubectl_cache(cache_key, cache_ttl)
        if hit:
            # Return cached result
            return cached_result

    # Execute command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        rc = result.returncode
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        # Cache the result
        if use_cache:
            cache_value = (rc, stdout, stderr)
            _set_kubectl_cache(cache_key, cache_value)

        return rc, stdout, stderr
    except Exception as e:
        return 1, "", str(e)


def parse_cpu(value):
    """Parse CPU string to millicores. '100m' → 100, '1.5' → 1500."""
    value = value.strip()
    if value.endswith("m"):
        return int(value[:-1])
    try:
        return int(float(value) * 1000)
    except ValueError:
        return 0


def parse_memory(value):
    """Parse memory string to MiB. '128Mi' → 128, '2Gi' → 2048."""
    value = value.strip()
    if value.endswith("Mi"):
        return int(value[:-2])
    elif value.endswith("Gi"):
        return int(value[:-2]) * 1024
    elif value.endswith("Ki"):
        return int(value[:-2]) // 1024
    elif value.endswith("Ti"):
        return int(value[:-2]) * 1024 * 1024
    try:
        return int(value) // (1024 * 1024)  # bytes to MiB
    except ValueError:
        return 0


def collect_top_metrics(output):
    """Parse `kubectl top pods -A` output. Returns {ns/name: {cpu_m, memory_mib}}."""
    metrics = {}
    lines = output.strip().split("\n")
    for line in lines[1:]:  # skip header
        parts = line.split()
        if len(parts) < 4:
            continue
        ns = parts[0]
        name = parts[1]
        cpu = parse_cpu(parts[2])
        mem = parse_memory(parts[3])
        metrics[f"{ns}/{name}"] = {"cpu_m": cpu, "memory_mib": mem}
    return metrics


def collect_top_nodes(output):
    """Parse `kubectl top nodes` output. Returns {node_name: {cpu_m, memory_mib, cpu_pct, memory_pct}}."""
    metrics = {}
    lines = output.strip().split("\n")
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        name = parts[0]
        cpu = parse_cpu(parts[1])
        mem = parse_memory(parts[3])
        cpu_pct = float(parts[2].rstrip("%")) if parts[2].endswith("%") else 0.0
        mem_pct = float(parts[4].rstrip("%")) if parts[4].endswith("%") else 0.0
        metrics[name] = {
            "cpu_m": cpu,
            "memory_mib": mem,
            "cpu_pct": cpu_pct,
            "memory_pct": mem_pct,
        }
    return metrics


def get_pod_resources(pod_json):
    """Extract resource limits from pod JSON. Returns {container_name: {cpu_m, memory_mib}}."""
    resources = {}
    spec = pod_json.get("spec", {})
    for container in spec.get("containers", []):
        name = container.get("name", "")
        limits = container.get("resources", {}).get("limits", {})
        cpu_limit = 0
        mem_limit = 0
        if "cpu" in limits:
            cpu_limit = parse_cpu(limits["cpu"])
        if "memory" in limits:
            mem_limit = parse_memory(limits["memory"])
        resources[name] = {"cpu_m": cpu_limit, "memory_mib": mem_limit}
    return resources


def get_node_conditions(node_json):
    """Extract node conditions from `kubectl get nodes -o json`."""
    conditions: dict = {}
    for item in node_json.get("items", []):
        name = item.get("metadata", {}).get("name", "")
        for cond in item.get("status", {}).get("conditions", []):
            cond_type = cond.get("type", "")
            status = cond.get("status", "Unknown")
            conditions[name] = conditions.get(name, {})
            conditions[name][cond_type] = status
    return conditions


def count_log_errors(log_text):
    """Count error-level log lines."""
    count = 0
    for line in log_text.split("\n"):
        if _ERROR_RE.search(line):
            count += 1
    return count


# Container statuses that indicate serious problems
_CRASH_STATES = {
    "CrashLoopBackOff", "CreateContainerConfigError", "CreateContainerError",
    "ImagePullBackOff", "ErrImagePull", "InvalidImageName",
    "RunContainerError", "ContainerStatusUnknown",
}


# ─── Docker Runtime Detection and Selector Parsing ──────────────────────────

import os as _os

_RUNTIME_CACHE = None


def detect_runtime():
    """Auto-detect the container runtime (Docker or Kubernetes).
    
    Checks in order:
    1. OPERATOR_RUNTIME env var (explicit override)
    2. DOCKER_SOCKET env var pointing to a valid socket
    3. Default Docker socket at /var/run/docker.sock
    4. Default to 'kubernetes'
    
    Returns:
        'docker' or 'kubernetes'
    """
    global _RUNTIME_CACHE

    # Return cached value if available
    if _RUNTIME_CACHE is not None:
        return _RUNTIME_CACHE

    # Check explicit runtime setting
    runtime = _os.environ.get("OPERATOR_RUNTIME", "").lower()
    if runtime in ("docker", "kubernetes", "k8s"):
        if runtime == "k8s":
            _RUNTIME_CACHE = "kubernetes"
        else:
            _RUNTIME_CACHE = runtime
        return _RUNTIME_CACHE

    # Check for Docker socket
    docker_socket = _os.environ.get("DOCKER_SOCKET", "/var/run/docker.sock")
    if _os.path.exists(docker_socket):
        _RUNTIME_CACHE = "docker"
        return _RUNTIME_CACHE

    # Default to Kubernetes
    _RUNTIME_CACHE = "kubernetes"
    return _RUNTIME_CACHE


def parse_watch_labels(labels_str):
    """Parse OPERATOR_WATCH_LABELS environment variable.
    
    Format: comma-separated list of key=value pairs
    Example: "app=web,env=prod,version=1.0"
    
    Args:
        labels_str: String of comma-separated labels, or None
        
    Returns:
        dict: {key: value} or empty dict if None/empty
    """
    if not labels_str or not labels_str.strip():
        return {}

    labels = {}
    for pair in labels_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        # Split on first '=' to handle values that might contain '='
        if "=" in pair:
            key, value = pair.split("=", 1)
            labels[key.strip()] = value.strip()
    return labels


def parse_watch_compose_projects(projects_str):
    """Parse OPERATOR_WATCH_COMPOSE_PROJECTS environment variable.
    
    Format: comma-separated list of compose project names
    Example: "myapp,webapp,api"
    
    Args:
        projects_str: String of comma-separated project names, or None
        
    Returns:
        list: list of project names or empty list if None/empty
    """
    if not projects_str or not projects_str.strip():
        return []

    projects = []
    for project in projects_str.split(","):
        project = project.strip()
        if project:
            projects.append(project)
    return projects


def parse_watch_names(names_str):
    """Parse OPERATOR_WATCH_NAMES environment variable.
    
    Format: comma-separated list of container name patterns (supports wildcards)
    Example: "web-*,api-*,db"
    
    Args:
        names_str: String of comma-separated name patterns, or None
        
    Returns:
        list: list of name patterns or empty list if None/empty
    """
    if not names_str or not names_str.strip():
        return []

    names = []
    for name in names_str.split(","):
        name = name.strip()
        if name:
            names.append(name)
    return names


def get_watch_selectors():
    """Get all watch selectors based on the detected runtime.
    
    Returns:
        dict: Dictionary containing:
            - runtime: 'docker' or 'kubernetes'
            - labels: dict of label selectors (Docker mode only)
            - compose_projects: list of compose project names (Docker mode only)
            - names: list of name patterns (Docker mode only)
            - namespaces: list of K8s namespaces (Kubernetes mode only, from config)
    """
    runtime = detect_runtime()

    if runtime == "docker":
        return {
            "runtime": "docker",
            "labels": parse_watch_labels(_os.environ.get("OPERATOR_WATCH_LABELS")),
            "compose_projects": parse_watch_compose_projects(
                _os.environ.get("OPERATOR_WATCH_COMPOSE_PROJECTS")
            ),
            "names": parse_watch_names(_os.environ.get("OPERATOR_WATCH_NAMES")),
        }
    else:
        # Kubernetes mode - read from environment directly to avoid caching
        namespaces_str = _os.environ.get("OPERATOR_WATCH_NAMESPACES", "")
        namespaces = namespaces_str.split(",") if namespaces_str else []
        return {
            "runtime": "kubernetes",
            "namespaces": namespaces,
        }


def get_pod_status_signals(pod_json):
    """Extract pod phase and container status signals from pod JSON.

    Returns a dict with:
        pod_phase: str (Pending, Running, Succeeded, Failed, Unknown)
        container_ready: bool (True if main container is ready)
        wait_state: str or None (reason if container is waiting, e.g. CrashLoopBackOff)
        terminated: bool (True if container was terminated)
        terminated_reason: str or None (e.g. OOMKilled, Error)
        restart_count: int (total restarts across all containers)
        pod_initialized: bool
        pod_scheduled: bool
    """
    status = pod_json.get("status", {})
    _spec = pod_json.get("spec", {})

    pod_phase = status.get("phase", "Unknown")

    # Pod conditions
    conditions = {c.get("type"): c.get("status") for c in status.get("conditions", [])}
    pod_initialized = conditions.get("Initialized", "False") == "True"
    pod_scheduled = conditions.get("PodScheduled", "False") == "True"

    # Container statuses
    container_ready = True
    wait_state = None
    terminated = False
    terminated_reason = None
    restart_count = 0

    for cs in status.get("containerStatuses", []):
        restart_count += cs.get("restartCount", 0)
        if not cs.get("ready", False):
            container_ready = False
        # Check waiting state
        waiting = cs.get("state", {}).get("waiting", {})
        if waiting:
            reason = waiting.get("reason", "")
            if reason:
                wait_state = reason
        # Check terminated state
        term = cs.get("state", {}).get("terminated", {})
        if term:
            terminated = True
            terminated_reason = term.get("reason", "")
        # Check last state for terminated info
        last_term = cs.get("lastState", {}).get("terminated", {})
        if last_term and not terminated:
            terminated_reason = last_term.get("reason", "")

    return {
        "pod_phase": pod_phase,
        "container_ready": container_ready,
        "wait_state": wait_state,
        "terminated": terminated,
        "terminated_reason": terminated_reason,
        "restart_count": restart_count,
        "pod_initialized": pod_initialized,
        "pod_scheduled": pod_scheduled,
    }


def get_container_status_signals(container_json):
    """Extract container status signals from Docker inspect JSON.

    Takes the output of `docker inspect <container>` (a single container dict,
    not the list) and extracts status signals matching the fields from
    get_pod_status_signals() for consistency.

    Returns a dict with:
        wait_state: str or None (reason if container is waiting/restarting, e.g. Restarting, Dead)
        terminated: bool (True if container is not running - exited, dead, etc.)
        restart_count: int (number of restarts from container state)
        ready: bool (True if container is running, not restarting, and healthy)
    """
    state = container_json.get("State", {})

    # RestartCount lives at the TOP LEVEL of `docker inspect` JSON in
    # modern Docker (State.RestartCount is null); fall back to State.
    restart_count = container_json.get("RestartCount", state.get("RestartCount", 0)) or 0

    # Determine state flags
    running = state.get("Running", False)
    paused = state.get("Paused", False)
    restarting = state.get("Restarting", False)
    dead = state.get("Dead", False)
    oom_killed = state.get("OOMKilled", False)
    exit_code = state.get("ExitCode")

    # Determine wait_state
    wait_state = None
    if restarting:
        wait_state = "Restarting"
    elif dead:
        wait_state = "Dead"
    elif oom_killed:
        wait_state = "OOMKilled"
    elif exit_code is not None and exit_code != 0:
        wait_state = f"Exited ({exit_code})"
    elif exit_code == 0:
        wait_state = "Exited"
    elif not running and not paused and not restarting and not dead and not oom_killed:
        # Container is not running but we don't have specific info - treat as none
        # This handles the case of empty state dict
        wait_state = None

    # terminated is True if container is not in a running state
    # (docker keeps Running: true during restart backoff, so a restarting
    # container counts as down for risk purposes)
    terminated = not running or restarting or dead or oom_killed or exit_code is not None

    # Determine ready state
    # Container is ready if it's running, not restarting, and healthy
    health = state.get("Health", {})
    health_status = health.get("Status", "") if health else ""
    is_healthy = True
    if health:
        # If health check exists, container is only ready if explicitly "healthy"
        is_healthy = health_status == "healthy"

    ready = running and not restarting and not paused and not dead and is_healthy

    return {
        "wait_state": wait_state,
        "terminated": terminated,
        "restart_count": restart_count,
        "ready": ready,
    }


def docker_status_signals(docker_info):
    """Convert discover_docker_containers() output into pod-style status signals.

    ``discover_docker_containers`` returns the rich container_info shape
    (status/healthy/exit_code/restart_count), NOT raw ``docker inspect`` JSON.
    ``get_container_status_signals`` expects the raw JSON with a ``State`` dict,
    so feeding it container_info made every docker pod look Running/ready.
    This mirrors ``get_pod_status_signals`` for the Kubernetes path.

    Mapping:
      running   -> pod_phase Running, ready driven by the health check
      restarting-> wait_state Restarting, not ready, terminated (crash loop)
      exited    -> pod_phase Succeeded (mirrors k8s completed pods: skipped)
      dead      -> pod_phase Failed, wait_state Dead
    """
    status = docker_info.get("status", "unknown")
    healthy = docker_info.get("healthy")

    if status == "running":
        pod_phase = "Running"
    elif status == "exited":
        pod_phase = "Succeeded"
    elif status == "dead":
        pod_phase = "Failed"
    elif status in ("paused", "restarting"):
        pod_phase = "Running"
    else:
        pod_phase = "Unknown"

    # Ready: running and (no healthcheck -> assumed healthy, or explicitly healthy)
    container_ready = status == "running" and (healthy or healthy is None)

    wait_state = None
    if status == "restarting":
        wait_state = "Restarting"
    elif status == "dead":
        wait_state = "Dead"

    terminated = status in ("restarting", "exited", "dead")
    terminated_reason = None
    if status == "exited" and docker_info.get("exit_code") is not None:
        terminated_reason = f"Exited ({docker_info['exit_code']})"
    elif status == "dead":
        terminated_reason = "Error"

    return {
        "pod_phase": pod_phase,
        "container_ready": container_ready,
        "wait_state": wait_state,
        "terminated": terminated,
        "terminated_reason": terminated_reason,
        "restart_count": docker_info.get("restart_count", 0),
    }


# ─── Docker Container Discovery ──────────────────────────────────────────

import fnmatch


def _parse_docker_ps_line(line):
    """Parse a single line of docker ps --format json output.
    
    docker ps --format '{{json .}}' outputs one JSON object per line.
    Returns a dict with container info, or None if parsing fails.
    """
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None


def _get_container_inspect(container_id):
    """Get docker inspect output for a container.
    
    Returns the first inspect result (docker inspect returns a list) or None on error.
    """
    cmd = ["docker", "inspect", container_id]
    returncode, stdout, stderr = run_cmd(cmd, timeout=10)
    if returncode != 0 or not stdout:
        return None
    try:
        results = json.loads(stdout)
        return results[0] if results else None
    except (json.JSONDecodeError, ValueError, IndexError, KeyError, TypeError):
        # Malformed / unexpected inspect shapes are treated like a failed
        # inspect: discover falls back to the docker ps line instead.
        return None


def _matches_selectors(container_info, selectors):
    """Check if a container matches the configured selectors.
    
    Args:
        container_info: dict with container info (from docker ps or inspect)
        selectors: dict from get_watch_selectors() with labels, compose_projects, names
        
    Returns:
        bool: True if container matches all configured selectors
    """
    # No selectors means match all
    if not selectors.get("labels") and not selectors.get("compose_projects") and not selectors.get("names"):
        return True

    # Check label selectors
    if selectors.get("labels"):
        container_labels = container_info.get("Labels", {})
        # Parse container labels from "key=value,key2=value2" format
        if isinstance(container_labels, str):
            container_labels = parse_watch_labels(container_labels)
        # Check all selector labels are present and match
        for key, value in selectors["labels"].items():
            if container_labels.get(key) != value:
                return False

    # Check compose project selectors
    if selectors.get("compose_projects"):
        # Compose project is typically in the label com.docker.compose.project
        container_labels = container_info.get("Labels", {})
        if isinstance(container_labels, str):
            container_labels = parse_watch_labels(container_labels)
        project = container_labels.get("com.docker.compose.project", "")
        if project not in selectors["compose_projects"]:
            return False

    # Check name patterns
    if selectors.get("names"):
        container_name = container_info.get("Names", "")
        # Remove leading '/' from container name if present
        if container_name.startswith("/"):
            container_name = container_name[1:]
        # Check if name matches any of the patterns (supports wildcards)
        matched = False
        for pattern in selectors["names"]:
            if fnmatch.fnmatch(container_name, pattern):
                matched = True
                break
        if not matched:
            return False

    return True


def discover_docker_containers():
    """Discover Docker containers using docker ps and docker inspect.
    
    Uses docker ps --format '{{json .}}' to list containers, then docker inspect
    to get detailed information. Filters containers based on configured selectors
    from get_watch_selectors().
    
    Returns:
        dict: Dictionary keyed by container ID (short form, e.g. 'abc123') containing:
            - id: str, full container ID
            - name: str, container name (without leading '/')
            - image: str, container image
            - status: str, container status (e.g. 'running', 'exited')
            - state: dict with detailed state info (Running, Paused, Restarting, OOMKilled, etc.)
            - labels: dict, container labels
            - compose_project: str or None, compose project name from labels
            - ports: dict, exposed ports
            - networks: list, network names
            - created_at: str, creation timestamp
            - healthy: bool, health status if health check is configured
            - exit_code: int or None, exit code if container exited
            - restart_count: int, number of restarts
            
        Returns empty dict {} if docker is not available or no containers found.
    """
    selectors = get_watch_selectors()

    # If not in Docker mode, return empty
    if selectors.get("runtime") != "docker":
        return {}

    # Run docker ps to list all containers
    cmd = ["docker", "ps", "-a", "--format", "{{json .}}"]
    returncode, stdout, stderr = run_cmd(cmd, timeout=10)

    if returncode != 0:
        # docker command failed, return empty
        return {}

    containers = {}
    for line in stdout.split("\n"):
        container_ps = _parse_docker_ps_line(line)
        if not container_ps:
            continue

        container_id = container_ps.get("ID", "")
        if not container_id:
            continue

        # Get detailed info from docker inspect
        inspect_info = _get_container_inspect(container_id)
        if not inspect_info:
            # If inspect fails, use ps info only
            inspect_info = {}

        # Build container info dict
        container_name = container_ps.get("Names", "") or inspect_info.get("Name", "")
        # Remove leading '/' from name
        if container_name.startswith("/"):
            container_name = container_name[1:]

        # Parse labels from ps output (comma-separated key=value)
        labels_str = container_ps.get("Labels", "")
        labels = parse_watch_labels(labels_str) if labels_str else {}
        # Merge with inspect labels if available
        inspect_labels = inspect_info.get("Config", {}).get("Labels", {})
        if inspect_labels:
            labels.update(inspect_labels)

        # Get compose project from labels
        compose_project = labels.get("com.docker.compose.project")

        # Get state from inspect
        state = inspect_info.get("State", {})

        # Determine status
        status = "unknown"
        if state.get("Running"):
            status = "running"
        elif state.get("Paused"):
            status = "paused"
        elif state.get("Restarting"):
            status = "restarting"
        elif state.get("Dead"):
            status = "dead"
        else:
            # Check ExitCode
            if state.get("ExitCode") is not None:
                status = "exited"

        # Get ports from inspect
        ports = {}
        network_settings = inspect_info.get("NetworkSettings", {})
        port_bindings = network_settings.get("Ports", {})
        if port_bindings:
            for container_port, bindings in port_bindings.items():
                if bindings:
                    for binding in bindings:
                        host_ip = binding.get("HostIp", "0.0.0.0")
                        host_port = binding.get("HostPort", "")
                        ports[f"{host_ip}:{host_port}"] = container_port

        # Get networks
        networks = list(network_settings.get("Networks", {}).keys())

        # Get health status
        health = inspect_info.get("State", {}).get("Health", {})
        healthy = None
        if health:
            health_status = health.get("Status", "")
            healthy = health_status == "healthy"

        # Build container info
        container_info = {
            "id": container_id,
            "name": container_name,
            "image": container_ps.get("Image", "") or inspect_info.get("Config", {}).get("Image", ""),
            "status": status,
            "state": state,
            "labels": labels,
            "compose_project": compose_project,
            "ports": ports,
            "networks": networks,
            "created_at": _normalize_created_at(
                container_ps.get("CreatedAt", "") or state.get("StartedAt", "")
            ),
            "healthy": healthy,
            "exit_code": inspect_info.get("ExitCode", state.get("ExitCode")),
            # RestartCount lives at the top level of `docker inspect` JSON in
            # modern Docker (State.RestartCount is null); fall back to State.
            "restart_count": inspect_info.get("RestartCount", state.get("RestartCount", 0)) or 0,
        }

        # Check if container matches selectors
        if not _matches_selectors(container_ps, selectors):
            continue

        # Use short ID as key for stability
        containers[container_id[:12]] = container_info

    return containers
