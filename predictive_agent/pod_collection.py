"""Runtime metric collection helpers for the reconcile pipeline.

Pure runtime-data functions extracted from main.py (behavior preserved):
kubectl pod discovery, pod field extraction, Docker container discovery
selectors and stats collection. The main module re-binds these names so
existing ``patch("predictive_agent.main._get_pods_json")`` targets keep
working; the reconcile pipeline resolves them through main's namespace.
"""

import fnmatch
import json
import logging
from datetime import datetime, timezone
from typing import Tuple

from predictive_agent import config
from predictive_agent.collector import parse_memory

logger = logging.getLogger("predictive-agent")


def now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def kubectl_available() -> bool:
    """Check if kubectl binary is available and can connect to a cluster."""
    from predictive_agent import main  # deferred: late-bound patch target
    rc, _, _ = main.run_cmd(["kubectl", "version", "--client"], timeout=5)
    return rc == 0


def kubectl_binary_missing() -> bool:
    """True when the kubectl executable is not on PATH.

    In kubernetes mode collection (pods, metrics, logs) shells out to
    ``kubectl``; without the binary nothing can be collected. Used both to
    raise a loud ``KubectlUnavailableError`` per cycle and by the startup
    fail-fast gate in ``main._check_runtime_prerequisites``.
    """
    import shutil
    return shutil.which("kubectl") is None


_SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"


def ensure_incluster_kubeconfig() -> bool:
    """Write an in-cluster kubeconfig from the mounted ServiceAccount token.

    kubectl does *not* auto-discover ``KUBERNETES_SERVICE_HOST`` or the
    service-account token; with no kubeconfig every invocation fails with
    "The connection to the server localhost:8080 was refused". That is the
    second half of the silent-blindness bug: even an image carrying kubectl
    could never list pods. The entrypoint creates ``~/.kube`` but leaves it
    empty — this call materialises a working config when running in-cluster
    (mount at /var/run/secrets/kubernetes.io/serviceaccount).

    Returns True when a usable kubeconfig exists (pre-existing or freshly
    written), False when not running in-cluster (rely on ambient config).
    """
    import os

    sa_dir = _SA_DIR
    host = os.environ.get("KUBERNETES_SERVICE_HOST")
    port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
    if not host or not os.path.isdir(sa_dir):
        return False  # not in-cluster
    token_path = os.path.join(sa_dir, "token")
    ca_path = os.path.join(sa_dir, "ca.crt")
    if not (os.path.isfile(token_path) and os.path.isfile(ca_path)):
        return False
    home = os.environ.get("HOME", "/root")
    kube_dir = os.path.join(home, ".kube")
    config_path = os.environ.get("KUBECONFIG") or os.path.join(kube_dir, "config")
    if os.path.exists(config_path):
        return True  # already configured (mounted or externally provisioned)
    with open(token_path, encoding="utf-8") as fh:
        token = fh.read().strip()
    cfg = (
        "apiVersion: v1\n"
        "kind: Config\n"
        "clusters:\n"
        "- name: in-cluster\n"
        "  cluster:\n"
        f"    certificate-authority: {ca_path}\n"
        f"    server: https://{host}:{port}\n"
        "contexts:\n"
        "- name: in-cluster\n"
        "  context:\n"
        "    cluster: in-cluster\n"
        "    user: in-cluster\n"
        "current-context: in-cluster\n"
        "users:\n"
        "- name: in-cluster\n"
        "  user:\n"
        f"    token: {token}\n"
    )
    os.makedirs(kube_dir, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as fh:
        fh.write(cfg)
    os.chmod(config_path, 0o600)
    return True


class KubectlUnavailableError(RuntimeError):
    """Collection could not run because kubectl is missing or failed.

    Previously a missing/broken kubectl silently degraded to ``{"items": []}``,
    which made the reconcile loop treat ``0 pods tracked`` as a *valid* empty
    result: no warnings, HEALTHY collector state, /ready 200. That failure mode
    blinded a live deployment for weeks (image shipped without kubectl).
    Raising instead routes into the existing collection-failure machinery:
    collector_state -> DEGRADED, per-cycle warning, /ready -> 503, and the
    keep-last-known degradation protocol stays effective.
    """


def get_pods_json() -> dict:
    """Fetch all pods across watched namespaces as JSON.

    Raises :class:`KubectlUnavailableError` on any collection failure (binary
    missing, non-zero exit, empty output, or unparsable JSON) so the caller can
    mark the cycle as failed instead of silently tracking zero pods.
    """
    from predictive_agent import main  # deferred: late-bound patch target

    if kubectl_binary_missing():
        raise KubectlUnavailableError(
            "kubectl binary not found on PATH — kubernetes-mode collection "
            "cannot run (operator would silently track 0 pods). Install kubectl "
            "into the image or set COLLECTOR_MODE=docker."
        )
    rc, stdout, _ = main.run_cmd(
        ["kubectl", "get", "pods", "-A", "-o", "json"], timeout=10
    )
    if rc != 0:
        raise KubectlUnavailableError(f"kubectl get pods failed (rc={rc})")
    if not stdout:
        raise KubectlUnavailableError("kubectl get pods returned empty stdout")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise KubectlUnavailableError(
            f"kubectl get pods returned unparsable JSON: {exc}"
        ) from exc


def get_pod_restart_count(pod: dict) -> int:
    """Extract total restart count from a pod JSON object."""
    statuses = pod.get("status", {}).get("containerStatuses", [])
    return sum(cs.get("restartCount", 0) for cs in statuses)


def get_pod_namespace(pod: dict) -> str:
    return pod.get("metadata", {}).get("namespace", "")


def get_pod_name(pod: dict) -> str:
    return pod.get("metadata", {}).get("name", "")


def get_pod_memory_limit(pod: dict, container_name: str = "") -> int:
    """Extract memory limit in MiB from pod spec.

    Without ``container_name`` the sum over all containers is returned, so the
    limit is comparable to ``kubectl top pods`` which reports whole-pod usage.
    Returning only the first container's limit here inflated memory_pct for
    multi-container pods (e.g. csi-nfs-node: ~95Mi used across 3 containers vs
    the 100Mi limit of liveness-probe -> a wrong 95%).
    """
    spec = pod.get("spec", {})
    total = 0
    found = False
    for container in spec.get("containers", []):
        if container_name and container.get("name", "") != container_name:
            continue
        limits = container.get("resources", {}).get("limits", {})
        if "memory" in limits:
            from predictive_agent.collector import parse_memory
            total += parse_memory(limits["memory"])
            found = True
            if container_name:
                # Named lookup: return that container's limit directly.
                return parse_memory(limits["memory"])
    return total if (found and not container_name) else 0


def should_skip_namespace(ns: str) -> bool:
    """Check if a namespace should be skipped."""
    if ns in config.SKIP_NAMESPACES:
        return True
    if config.WATCH_NAMESPACES and ns not in config.WATCH_NAMESPACES:
        return True
    return False


def is_pod_excluded(pod: dict) -> bool:
    """True when a pod must be ignored by monitoring/remediation.

    Checks ``config.EXCLUDED_POD_LABELS`` / ``EXCLUDED_POD_ANNOTATIONS``
    (key=value or bare-key=any) against the pod's metadata. The canonical use
    is excluding KubeVirt ``virt-launcher-*`` pods (label
    ``kubevirt.io=virt-launcher``): their memory/CPU profile mirrors a whole
    guest OS (a Windows guest deliberately uses ~all allocated RAM), so the
    risk model labels them CRITICAL forever and restarting the launcher pod is
    a disruptive VM reboot that never durably reduces P(FAILED).
    """
    meta = pod.get("metadata", {}) or {}
    labels = meta.get("labels", {}) or {}
    annotations = meta.get("annotations", {}) or {}
    for key, value in config.EXCLUDED_POD_LABELS.items():
        if key in labels and (value is None or labels[key] == value):
            return True
    for key, value in config.EXCLUDED_POD_ANNOTATIONS.items():
        if key in annotations and (value is None or annotations[key] == value):
            return True
    return False


def get_node_pressure(node_conditions: dict, node_name: str) -> Tuple[bool, bool]:
    """Check if a node has memory or disk pressure."""
    conds = node_conditions.get(node_name, {})
    mem_pressure = conds.get("MemoryPressure", "False") == "True"
    disk_pressure = conds.get("DiskPressure", "False") == "True"
    return mem_pressure, disk_pressure


def matches_docker_selectors(container_info: dict, selectors: dict) -> bool:
    """Check if a Docker container matches configured selectors.

    Args:
        container_info: dict with container info from discover_docker_containers
        selectors: dict from get_watch_selectors() with labels, compose_projects, names

    Returns:
        bool: True if container matches all configured selectors
    """
    # No selectors means match all
    if not selectors.get("labels") and not selectors.get("compose_projects") and not selectors.get("names"):
        return True

    # Check label selectors
    if selectors.get("labels"):
        container_labels = container_info.get("labels", {})
        for key, value in selectors["labels"].items():
            if container_labels.get(key) != value:
                return False

    # Check compose project selectors
    if selectors.get("compose_projects"):
        project = container_info.get("compose_project", "")
        if project not in selectors["compose_projects"]:
            return False

    # Check name patterns
    if selectors.get("names"):
        container_name = container_info.get("name", "")
        matched = False
        for pattern in selectors["names"]:
            if fnmatch.fnmatch(container_name, pattern):
                matched = True
                break
        if not matched:
            return False

    return True


def collect_docker_container_metrics() -> dict:
    """Collect Docker container metrics using docker stats.

    Returns:
        dict: Container ID -> {cpu_m, memory_mib} mapping
    """
    metrics = {}
    try:
        from predictive_agent import main  # deferred: late-bound patch target
        # Use docker stats --no-stream --format to get current metrics
        rc, stdout, _ = main.run_cmd(
            ["docker", "stats", "--no-stream", "--format", "{{.Container}},{{.CPUPerc}},{{.MemUsage}}"],
            timeout=10
        )
        if rc == 0 and stdout:
            for line in stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split(",")
                if len(parts) >= 3:
                    container_id = parts[0].strip()
                    cpu_pct_str = parts[1].strip().rstrip("%")
                    mem_usage_str = parts[2].strip()

                    try:
                        cpu_pct = float(cpu_pct_str)
                        cpu_m = int(cpu_pct * 10)  # Convert % to millicores (assuming 100% = 1000m)

                        # Parse memory usage like "50MiB / 1GiB"
                        mem_usage = mem_usage_str.split("/")[0].strip()
                        memory_mib = parse_memory(mem_usage)

                        # Use short container ID as key
                        short_id = container_id[:12] if len(container_id) > 12 else container_id
                        metrics[short_id] = {
                            "cpu_m": cpu_m,
                            "memory_mib": memory_mib,
                        }
                    except (ValueError, IndexError):
                        continue
    except Exception as e:  # noqa: BLE001 - metric collection must never crash the loop
        logger.debug("Failed to collect Docker metrics: %s", e)

    return metrics


def get_container_memory_limit(inspect_info: dict) -> int:
    """Extract memory limit from Docker inspect info in MiB."""
    try:
        # Check HostConfig.Memory (in bytes) first
        memory_bytes = inspect_info.get("HostConfig", {}).get("Memory", 0)
        if memory_bytes > 0:
            return memory_bytes // (1024 * 1024)  # Convert to MiB

        # Fallback: check Config.Memory if it exists
        memory_bytes = inspect_info.get("Config", {}).get("Memory", 0)
        if memory_bytes > 0:
            return memory_bytes // (1024 * 1024)

        return 0
    except (TypeError, ValueError):
        return 0
