"""Direct unit tests for predictive_agent.pod_collection (PA-S3).

Exercises the moved collection helpers in isolation: pod JSON parsing,
memory-limit extraction (with the collector.parse_memory dependency),
Docker metric parsing (CPU% → millicores, mem "50MiB / 1GiB" → MiB),
namespace-skipping, node-pressure flags, and Docker-selector matching
including fnmatch name patterns. Patches ``predictive_agent.main.run_cmd``
(the late-bound resolution target) where CLI calls are needed.
"""

from unittest.mock import patch

import pytest

import predictive_agent.config as config
import predictive_agent.main as main_mod
from predictive_agent import pod_collection

# ── _now_iso / _kubectl_available ───────────────────────────────────────────


def test_now_iso_format():
    ts = pod_collection.now_iso()
    assert ts.endswith("Z")
    assert len(ts) == 20  # YYYY-MM-DDTHH:MM:SSZ


def test_kubectl_available_false_when_run_cmd_nonzero():
    with patch.object(main_mod, "run_cmd", return_value=(1, "", "")):
        assert pod_collection.kubectl_available() is False


# ── get_pods_json ──────────────────────────────────────────────────────────


def test_get_pods_json_parses_items():
    with patch.object(main_mod, "run_cmd",
                      return_value=(0, '{"items": [{"metadata": {"name": "a"}}]}', "")), \
         patch.object(pod_collection, "kubectl_binary_missing", return_value=False):
        out = pod_collection.get_pods_json()
    assert out["items"][0]["metadata"]["name"] == "a"


def test_get_pods_json_empty_on_nonzero_rc():
    """A failed kubectl call is a *collection failure*, not a valid empty list."""
    with patch.object(main_mod, "run_cmd", return_value=(1, "", "")):
        with pytest.raises(pod_collection.KubectlUnavailableError):
            pod_collection.get_pods_json()


# ── ensure_incluster_kubeconfig ────────────────────────────────────────────


def _write_sa_dir(path, token="t0k3n", ca="fake-ca"):
    sa = path / "serviceaccount"
    sa.mkdir()
    (sa / "token").write_text(token)
    (sa / "ca.crt").write_text(ca)
    return sa


def test_kubeconfig_written_from_sa_token(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.9")
    monkeypatch.setenv("KUBERNETES_SERVICE_PORT", "443")
    _write_sa_dir(tmp_path, token="abc123")
    monkeypatch.setattr(
        pod_collection, "_SA_DIR", str(tmp_path / "serviceaccount"))
    assert pod_collection.ensure_incluster_kubeconfig() is True
    cfg = (home / ".kube" / "config").read_text()
    assert "server: https://10.0.0.9:443" in cfg
    assert "token: abc123" in cfg
    assert (home / ".kube" / "config").stat().st_mode & 0o777 == 0o600


def test_kubeconfig_returns_true_when_already_present(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".kube").mkdir(parents=True)
    (home / ".kube" / "config").write_text("pre-existing")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.9")
    _write_sa_dir(tmp_path)
    monkeypatch.setattr(pod_collection, "_SA_DIR", str(tmp_path / "serviceaccount"))
    assert pod_collection.ensure_incluster_kubeconfig() is True
    # untouched (a mounted real kubeconfig must win)
    assert (home / ".kube" / "config").read_text() == "pre-existing"


def test_kubeconfig_noop_outside_cluster(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    assert pod_collection.ensure_incluster_kubeconfig() is False
    assert not (home / ".kube" / "config").exists()


def test_get_pods_json_raises_on_missing_binary():
    """Without a kubectl binary, collection must fail loudly (not return [])."""
    with patch.object(pod_collection, "kubectl_binary_missing", return_value=True):
        with pytest.raises(pod_collection.KubectlUnavailableError):
            pod_collection.get_pods_json()


def test_get_pods_json_empty_on_bad_json():
    """Unparsable kubectl output is a collection failure, not an empty list."""
    with patch.object(main_mod, "run_cmd", return_value=(0, "not json", "")):
        with pytest.raises(pod_collection.KubectlUnavailableError):
            pod_collection.get_pods_json()


# ── field extractors ───────────────────────────────────────────────────────


def test_get_pod_restart_count_sums_statuses():
    pod = {"status": {"containerStatuses": [
        {"restartCount": 1}, {"restartCount": 3}]}}
    assert pod_collection.get_pod_restart_count(pod) == 4


def test_get_pod_restart_count_missing_returns_zero():
    assert pod_collection.get_pod_restart_count({}) == 0


def test_get_pod_namespace_and_name():
    pod = {"metadata": {"namespace": "kube-system", "name": "web-0"}}
    assert pod_collection.get_pod_namespace(pod) == "kube-system"
    assert pod_collection.get_pod_name(pod) == "web-0"


def test_get_pod_memory_limit_from_pod_spec():
    pod = {"spec": {"containers": [
        {"name": "c1", "resources": {"limits": {"memory": "512Mi"}}}]}}
    assert pod_collection.get_pod_memory_limit(pod) > 0


def test_get_pod_memory_limit_zero_when_absent():
    assert pod_collection.get_pod_memory_limit({"spec": {"containers": []}}) == 0
    assert pod_collection.get_pod_memory_limit({}) == 0


def test_get_pod_memory_limit_matches_named_container():
    pod = {"spec": {"containers": [
        {"name": "c1", "resources": {"limits": {"memory": "256Mi"}}},
        {"name": "c2", "resources": {"limits": {"memory": "768Mi"}}}]}}
    assert pod_collection.get_pod_memory_limit(pod, "c2") > 0


def test_get_pod_memory_limit_sums_all_containers():
    """Whole-pod limit must be the sum over containers (kubectl top reports
    whole-pod usage). Returning only the first limit inflated memory_pct for
    multi-container pods (e.g. csi-nfs-node)."""
    pod = {"spec": {"containers": [
        {"name": "c1", "resources": {"limits": {"memory": "100Mi"}}},
        {"name": "c2", "resources": {"limits": {"memory": "100Mi"}}},
        {"name": "c3", "resources": {"limits": {"memory": "300Mi"}}}]}}
    assert pod_collection.get_pod_memory_limit(pod) == 500


def test_get_pod_memory_limit_named_returns_only_that_container():
    pod = {"spec": {"containers": [
        {"name": "c1", "resources": {"limits": {"memory": "100Mi"}}},
        {"name": "c2", "resources": {"limits": {"memory": "300Mi"}}}]}}
    assert pod_collection.get_pod_memory_limit(pod, "c1") == 100
    assert pod_collection.get_pod_memory_limit(pod, "c2") == 300


# ── should_skip_namespace ──────────────────────────────────────────────────


def test_skip_namespace_in_skip_list():
    old = config.SKIP_NAMESPACES
    config.SKIP_NAMESPACES = {"kube-system"}
    old_watch = config.WATCH_NAMESPACES
    config.WATCH_NAMESPACES = None
    try:
        assert pod_collection.should_skip_namespace("kube-system") is True
    finally:
        config.SKIP_NAMESPACES = old
        config.WATCH_NAMESPACES = old_watch


def test_skip_namespace_when_watch_list_excludes():
    old = config.WATCH_NAMESPACES
    config.WATCH_NAMESPACES = ["default"]
    try:
        assert pod_collection.should_skip_namespace("other") is True
        assert pod_collection.should_skip_namespace("default") is False
    finally:
        config.WATCH_NAMESPACES = old


# ── get_node_pressure ──────────────────────────────────────────────────────


def test_get_node_pressure_flags():
    conds = {"n1": {"MemoryPressure": "True", "DiskPressure": "False"}}
    mem, disk = pod_collection.get_node_pressure(conds, "n1")
    assert mem is True
    assert disk is False


def test_get_node_pressure_missing_node():
    mem, disk = pod_collection.get_node_pressure({}, "nope")
    assert (mem, disk) == (False, False)


# ── matches_docker_selectors ───────────────────────────────────────────────


def _info(name="web", project="shop", labels=None):
    return {"name": name, "compose_project": project,
            "labels": labels or {"app": "web"}}


def test_matches_no_selectors_matches_all():
    assert pod_collection.matches_docker_selectors(_info(), {}) is True


def test_matches_label_selector():
    sel = {"labels": {"app": "web"}}
    assert pod_collection.matches_docker_selectors(_info(), sel) is True
    assert pod_collection.matches_docker_selectors(_info(labels={"app": "db"}), sel) is False


def test_matches_compose_project_selector():
    sel = {"compose_projects": {"shop"}}
    assert pod_collection.matches_docker_selectors(_info(), sel) is True
    assert pod_collection.matches_docker_selectors(_info(project="other"), sel) is False


def test_matches_name_pattern_fnmatch():
    sel = {"names": ["web*"]}
    assert pod_collection.matches_docker_selectors(_info(name="web-1"), sel) is True
    assert pod_collection.matches_docker_selectors(_info(name="db-1"), sel) is False


# ── collect_docker_container_metrics ───────────────────────────────────────


def _lines_to_stdout(lines):
    return "\n".join(lines) + ("\n" if lines else "")


def test_collect_docker_metrics_parses_cpu_and_mem():
    lines = [
        "e1f2a3b4c5d6e7f8,45.0%,50MiB / 1GiB",
        "container2,12.5%,200MiB / 512MiB",
    ]
    with patch.object(main_mod, "run_cmd",
                      return_value=(0, _lines_to_stdout(lines), "")):
        m = pod_collection.collect_docker_container_metrics()
    assert "e1f2a3b4c5d6" in m
    assert m["e1f2a3b4c5d6"]["cpu_m"] == 450  # 45.0% → 450m
    assert "container2" in m


def test_collect_docker_metrics_skips_malformed_lines():
    lines = ["e1f2a3b4c5d6,45.0%,50MiB / 1GiB", "garbage", ""]
    with patch.object(main_mod, "run_cmd",
                      return_value=(0, _lines_to_stdout(lines), "")):
        m = pod_collection.collect_docker_container_metrics()
    # only the valid line parses
    assert "e1f2a3b4c5d6" in m
    assert len(m) == 1


def test_collect_docker_metrics_empty_on_no_containers():
    with patch.object(main_mod, "run_cmd", return_value=(0, "", "")):
        assert pod_collection.collect_docker_container_metrics() == {}


def test_collect_docker_metrics_handles_parse_errors():
    lines = ["badcpu,abc%,50MiB / 1GiB"]
    with patch.object(main_mod, "run_cmd",
                      return_value=(0, _lines_to_stdout(lines), "")):
        m = pod_collection.collect_docker_container_metrics()
    assert m == {}  # ValueError on float() → line skipped


# ── get_container_memory_limit ─────────────────────────────────────────────


def test_get_container_memory_limit_from_hostconfig():
    info = {"HostConfig": {"Memory": 100 * 1024 * 1024}}  # 100 MiB in bytes
    assert pod_collection.get_container_memory_limit(info) == 100


def test_get_container_memory_limit_fallback_to_config():
    info = {"Config": {"Memory": 512 * 1024 * 1024}}  # 512 MiB
    assert pod_collection.get_container_memory_limit(info) == 512


def test_get_container_memory_limit_zero_when_absent():
    assert pod_collection.get_container_memory_limit({}) == 0


def test_get_container_memory_limit_handles_invalid_type():
    info = {"HostConfig": {"Memory": "not-a-number"}}
    # "not-a-number" has no .get attribute → TypeError → caught → 0
    assert pod_collection.get_container_memory_limit(info) == 0


# ── is_pod_excluded ─────────────────────────────────────────────────────────


def test_is_pod_excluded_by_label_key_value(monkeypatch):
    monkeypatch.setattr(config, "EXCLUDED_POD_LABELS", {"kubevirt.io": "virt-launcher"})
    monkeypatch.setattr(config, "EXCLUDED_POD_ANNOTATIONS", {})
    launcher = {"metadata": {"name": "virt-launcher-win2025-eval-x",
                             "labels": {"kubevirt.io": "virt-launcher",
                                        "kubevirt.io/domain": "win2025-eval"}}}
    assert pod_collection.is_pod_excluded(launcher) is True


def test_is_pod_excluded_label_wrong_value(monkeypatch):
    monkeypatch.setattr(config, "EXCLUDED_POD_LABELS", {"kubevirt.io": "virt-launcher"})
    monkeypatch.setattr(config, "EXCLUDED_POD_ANNOTATIONS", {})
    normal = {"metadata": {"name": "web-0", "labels": {"kubevirt.io": "virt-handler"}}}
    assert pod_collection.is_pod_excluded(normal) is False


def test_is_pod_excluded_by_bare_label_any_value(monkeypatch):
    monkeypatch.setattr(config, "EXCLUDED_POD_LABELS", {"kubevirt.io": None})
    monkeypatch.setattr(config, "EXCLUDED_POD_ANNOTATIONS", {})
    launcher = {"metadata": {"name": "virt-launcher-x", "labels": {"kubevirt.io": "anything"}}}
    assert pod_collection.is_pod_excluded(launcher) is True


def test_is_pod_excluded_by_annotation(monkeypatch):
    monkeypatch.setattr(config, "EXCLUDED_POD_LABELS", {})
    monkeypatch.setattr(config, "EXCLUDED_POD_ANNOTATIONS",
                        {"predictive-agent.openedu.svc/exclude": "true"})
    opted_out = {"metadata": {"name": "legacy-app", "annotations": {
        "predictive-agent.openedu.svc/exclude": "true"}}}
    assert pod_collection.is_pod_excluded(opted_out) is True


def test_is_pod_excluded_empty_config(monkeypatch):
    monkeypatch.setattr(config, "EXCLUDED_POD_LABELS", {})
    monkeypatch.setattr(config, "EXCLUDED_POD_ANNOTATIONS", {})
    pod = {"metadata": {"name": "app-0", "labels": {"kubevirt.io": "virt-launcher"}}}
    # Empty exclusion config → nothing excluded, regardless of pod labels.
    assert pod_collection.is_pod_excluded(pod) is False


def test_is_pod_excluded_missing_metadata(monkeypatch):
    monkeypatch.setattr(config, "EXCLUDED_POD_LABELS", {"kubevirt.io": "virt-launcher"})
    monkeypatch.setattr(config, "EXCLUDED_POD_ANNOTATIONS", {})
    assert pod_collection.is_pod_excluded({}) is False
