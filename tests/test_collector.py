"""Test metrics collection from kubectl and Docker."""
import os
from unittest.mock import patch

from predictive_agent.collector import (
    _normalize_created_at,
    collect_top_metrics,
    collect_top_nodes,
    count_log_errors,
    detect_runtime,
    get_watch_selectors,
    parse_cpu,
    parse_memory,
    parse_watch_compose_projects,
    parse_watch_labels,
    parse_watch_names,
)


def test_normalize_created_at():
    """Created-at timestamps from docker/k8s must normalize to ISO 8601 UTC."""
    # Go time.String() format emitted by 'docker ps' CreatedAt
    assert _normalize_created_at("2026-08-21 23:03:39 +0200 CEST") == \
        "2026-08-21T21:03:39+00:00"
    # RFC3339 with Z
    assert _normalize_created_at("2026-08-21T21:03:39Z") == \
        "2026-08-21T21:03:39+00:00"
    # Already ISO 8601 UTC
    assert _normalize_created_at("2026-08-21T21:03:39+00:00") == \
        "2026-08-21T21:03:39+00:00"
    # Empty / garbage must not crash
    assert _normalize_created_at("") == ""
    assert _normalize_created_at(None) == ""
    assert _normalize_created_at("not-a-date") == ""


def test_parse_cpu():
    """Test CPU parsing."""
    assert parse_cpu("100m") == 100
    assert parse_cpu("1.5") == 1500
    assert parse_cpu("0m") == 0
    assert parse_cpu("250m") == 250


def test_parse_cpu_invalid():
    """Test CPU parsing with invalid input."""
    assert parse_cpu("invalid") == 0
    assert parse_cpu("") == 0


def test_parse_memory():
    """Test memory parsing."""
    assert parse_memory("128Mi") == 128
    assert parse_memory("2Gi") == 2048
    assert parse_memory("1024Ki") == 1
    assert parse_memory("1Ti") == 1048576


def test_parse_memory_invalid():
    """Test memory parsing with invalid input."""
    assert parse_memory("invalid") == 0
    assert parse_memory("") == 0


def test_collect_top_metrics():
    """Test collecting metrics from kubectl top pods output."""
    mock_output = """NAMESPACE            NAME                                                  CPU(cores)   MEMORY(bytes)    
argocd               argocd-application-controller-0                       14m          169Mi            
opendesk             openldap-0                                           17m          124Mi            
"""
    metrics = collect_top_metrics(mock_output)
    assert "argocd/argocd-application-controller-0" in metrics
    assert metrics["argocd/argocd-application-controller-0"]["cpu_m"] == 14
    assert metrics["argocd/argocd-application-controller-0"]["memory_mib"] == 169
    assert "opendesk/openldap-0" in metrics


def test_collect_top_metrics_empty():
    """Test collecting metrics from empty output."""
    metrics = collect_top_metrics("")
    assert metrics == {}


def test_collect_top_nodes():
    """Test collecting node metrics."""
    mock_output = """NAME           CPU(cores)   CPU%   MEMORY(bytes)   MEMORY%   
clrz14-06      1000m        10%    8000Mi          20%       
clrz14-07      2000m        20%    16000Mi         40%       
"""
    metrics = collect_top_nodes(mock_output)
    assert "clrz14-06" in metrics
    assert metrics["clrz14-06"]["cpu_m"] == 1000
    assert metrics["clrz14-06"]["memory_mib"] == 8000
    assert metrics["clrz14-06"]["cpu_pct"] == 10.0
    assert metrics["clrz14-07"]["memory_pct"] == 40.0


def test_count_log_errors():
    """Test counting error lines in logs."""
    logs = """[INFO] Starting server
[ERROR] Connection failed
[WARN] Slow query
[FATAL] Database error
[DEBUG] Query executed
Traceback (most recent call last):
panic: runtime error
"""
    count = count_log_errors(logs)
    assert count == 4  # ERROR, FATAL, Traceback, panic


def test_count_log_errors_empty():
    """Test counting errors in empty logs."""
    assert count_log_errors("") == 0
    assert count_log_errors("Just normal logs\nNo errors here") == 0


# === New Docker selector parsing tests ===


def test_parse_watch_labels_empty():
    """Test parsing empty label selector."""
    assert parse_watch_labels("") == {}
    assert parse_watch_labels(None) == {}


def test_parse_watch_labels_single():
    """Test parsing a single label selector."""
    labels = parse_watch_labels("app=web")
    assert labels == {"app": "web"}


def test_parse_watch_labels_multiple():
    """Test parsing multiple label selectors."""
    labels = parse_watch_labels("app=web,env=prod,version=1.0")
    assert labels == {"app": "web", "env": "prod", "version": "1.0"}


def test_parse_watch_labels_with_spaces():
    """Test parsing label selectors with spaces."""
    labels = parse_watch_labels("app=web, env=prod, version=1.0")
    assert labels == {"app": "web", "env": "prod", "version": "1.0"}


def test_parse_watch_labels_whitespace():
    """Test parsing label selectors with extra whitespace."""
    labels = parse_watch_labels("  app=web  ,  env=prod  ")
    assert labels == {"app": "web", "env": "prod"}


def test_parse_watch_labels_equals_in_value():
    """Test parsing label selectors where value contains equals sign."""
    labels = parse_watch_labels("app=web=1")
    # Should split on first '=' only
    assert labels == {"app": "web=1"}


def test_parse_watch_compose_projects_empty():
    """Test parsing empty compose projects."""
    assert parse_watch_compose_projects("") == []
    assert parse_watch_compose_projects(None) == []


def test_parse_watch_compose_projects_single():
    """Test parsing a single compose project."""
    projects = parse_watch_compose_projects("myapp")
    assert projects == ["myapp"]


def test_parse_watch_compose_projects_multiple():
    """Test parsing multiple compose projects."""
    projects = parse_watch_compose_projects("myapp,webapp,api")
    assert projects == ["myapp", "webapp", "api"]


def test_parse_watch_compose_projects_with_spaces():
    """Test parsing compose projects with spaces."""
    projects = parse_watch_compose_projects("myapp, webapp, api")
    assert projects == ["myapp", "webapp", "api"]


def test_parse_watch_compose_projects_whitespace():
    """Test parsing compose projects with extra whitespace."""
    projects = parse_watch_compose_projects("  myapp  ,  webapp  ")
    assert projects == ["myapp", "webapp"]


def test_parse_watch_names_empty():
    """Test parsing empty name patterns."""
    assert parse_watch_names("") == []
    assert parse_watch_names(None) == []


def test_parse_watch_names_single():
    """Test parsing a single name pattern."""
    names = parse_watch_names("web-*")
    assert names == ["web-*"]


def test_parse_watch_names_multiple():
    """Test parsing multiple name patterns."""
    names = parse_watch_names("web-*,api-*,db")
    assert names == ["web-*", "api-*", "db"]


def test_parse_watch_names_with_spaces():
    """Test parsing name patterns with spaces."""
    names = parse_watch_names("web-*, api-*, db")
    assert names == ["web-*", "api-*", "db"]


def test_parse_watch_names_whitespace():
    """Test parsing name patterns with extra whitespace."""
    names = parse_watch_names("  web-*  ,  api-*  ")
    assert names == ["web-*", "api-*"]


# === Runtime detection tests ===


def test_detect_runtime_docker():
    """Test runtime detection for Docker (explicit)."""
    with patch.dict(os.environ, {"OPERATOR_RUNTIME": "docker"}):
        # Reset cache
        import predictive_agent.collector
        predictive_agent.collector._RUNTIME_CACHE = None
        assert detect_runtime() == "docker"


def test_detect_runtime_kubernetes():
    """Test runtime detection for Kubernetes (explicit)."""
    with patch.dict(os.environ, {"OPERATOR_RUNTIME": "kubernetes"}):
        # Reset cache
        import predictive_agent.collector
        predictive_agent.collector._RUNTIME_CACHE = None
        assert detect_runtime() == "kubernetes"


def test_detect_runtime_k8s_alias():
    """Test runtime detection for Kubernetes (k8s alias)."""
    with patch.dict(os.environ, {"OPERATOR_RUNTIME": "k8s"}):
        # Reset cache
        import predictive_agent.collector
        predictive_agent.collector._RUNTIME_CACHE = None
        assert detect_runtime() == "kubernetes"


def test_detect_runtime_auto_docker():
    """Test auto-detection of Docker runtime (no OPERATOR_RUNTIME set, Docker socket exists)."""
    env = os.environ.copy()
    env.pop("OPERATOR_RUNTIME", None)
    env["DOCKER_SOCKET"] = "/var/run/docker.sock"

    with patch.dict(os.environ, env, clear=True):
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True
            # Reset cache
            import predictive_agent.collector
            predictive_agent.collector._RUNTIME_CACHE = None
            assert detect_runtime() == "docker"


def test_detect_runtime_auto_kubernetes():
    """Test auto-detection of Kubernetes runtime (no OPERATOR_RUNTIME set, no Docker socket)."""
    env = os.environ.copy()
    env.pop("OPERATOR_RUNTIME", None)
    env.pop("DOCKER_SOCKET", None)

    with patch.dict(os.environ, env, clear=True):
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = False
            # Reset cache
            import predictive_agent.collector
            predictive_agent.collector._RUNTIME_CACHE = None
            assert detect_runtime() == "kubernetes"


def test_detect_runtime_default():
    """Test default runtime is kubernetes."""
    env = os.environ.copy()
    env.pop("OPERATOR_RUNTIME", None)
    env.pop("DOCKER_SOCKET", None)

    with patch.dict(os.environ, env, clear=True):
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = False
            # Reset cache
            import predictive_agent.collector
            predictive_agent.collector._RUNTIME_CACHE = None
            assert detect_runtime() == "kubernetes"


# === Get watch selectors tests ===


def test_get_watch_selectors_docker_mode():
    """Test getting watch selectors in Docker mode."""
    env = os.environ.copy()
    env["OPERATOR_RUNTIME"] = "docker"
    env["OPERATOR_WATCH_LABELS"] = "app=web,env=prod"
    env["OPERATOR_WATCH_COMPOSE_PROJECTS"] = "myapp,webapp"
    env["OPERATOR_WATCH_NAMES"] = "web-*,api-*"

    with patch.dict(os.environ, env, clear=True):
        # Reset cache
        import predictive_agent.collector
        predictive_agent.collector._RUNTIME_CACHE = None
        selectors = get_watch_selectors()
        assert selectors["runtime"] == "docker"
        assert selectors["labels"] == {"app": "web", "env": "prod"}
        assert selectors["compose_projects"] == ["myapp", "webapp"]
        assert selectors["names"] == ["web-*", "api-*"]


def test_get_watch_selectors_kubernetes_mode():
    """Test getting watch selectors in Kubernetes mode."""
    env = os.environ.copy()
    env["OPERATOR_RUNTIME"] = "kubernetes"
    env["OPERATOR_WATCH_NAMESPACES"] = "default,kube-system"

    with patch.dict(os.environ, env, clear=True):
        # Reset cache
        import predictive_agent.collector
        predictive_agent.collector._RUNTIME_CACHE = None
        selectors = get_watch_selectors()
        assert selectors["runtime"] == "kubernetes"
        assert selectors["namespaces"] == ["default", "kube-system"]


def test_get_watch_selectors_no_selectors_docker():
    """Test that no selectors in Docker mode means all containers."""
    env = os.environ.copy()
    env["OPERATOR_RUNTIME"] = "docker"
    env.pop("OPERATOR_WATCH_LABELS", None)
    env.pop("OPERATOR_WATCH_COMPOSE_PROJECTS", None)
    env.pop("OPERATOR_WATCH_NAMES", None)

    with patch.dict(os.environ, env, clear=True):
        # Reset cache
        import predictive_agent.collector
        predictive_agent.collector._RUNTIME_CACHE = None
        selectors = get_watch_selectors()
        assert selectors["runtime"] == "docker"
        assert selectors["labels"] == {}
        assert selectors["compose_projects"] == []
        assert selectors["names"] == []


# === Docker container discovery tests ===

from predictive_agent.collector import (
    _matches_selectors,
    _parse_docker_ps_line,
    discover_docker_containers,
)


class MockCompletedProcess:
    """Mock for subprocess.CompletedProcess."""
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_discover_docker_containers_empty():
    """Test discover_docker_containers with no containers."""
    env = os.environ.copy()
    env["OPERATOR_RUNTIME"] = "docker"

    with patch.dict(os.environ, env, clear=True):
        with patch("predictive_agent.collector.run_cmd") as mock_run:
            # Mock docker ps to return no containers
            mock_run.return_value = (0, "", "")
            containers = discover_docker_containers()
            assert containers == {}


def test_discover_docker_containers_empty_result():
    """Test discover_docker_containers with empty docker ps output."""
    env = os.environ.copy()
    env["OPERATOR_RUNTIME"] = "docker"

    with patch.dict(os.environ, env, clear=True):
        with patch("predictive_agent.collector.run_cmd") as mock_run:
            # Mock docker ps to return empty lines
            mock_run.return_value = (0, "\n\n", "")
            containers = discover_docker_containers()
            assert containers == {}


def test_discover_docker_containers_docker_command_error():
    """Test discover_docker_containers when docker command fails."""
    env = os.environ.copy()
    env["OPERATOR_RUNTIME"] = "docker"

    with patch.dict(os.environ, env, clear=True):
        with patch("predictive_agent.collector.run_cmd") as mock_run:
            # Mock docker ps to fail
            mock_run.return_value = (1, "", "docker: command not found")
            containers = discover_docker_containers()
            assert containers == {}


def test_discover_docker_containers_docker_error():
    """Test discover_docker_containers when docker inspect fails."""
    env = os.environ.copy()
    env["OPERATOR_RUNTIME"] = "docker"

    with patch.dict(os.environ, env, clear=True):
        with patch("predictive_agent.collector.run_cmd") as mock_run:
            # Mock docker ps to return a container that can't be inspected
            def mock_run_cmd(cmd, timeout=10):
                if "docker ps" in ' '.join(cmd):
                    return (0, '{"ID": "abc123def456", "Image": "nginx", "Names": "web-1", "Status": "Up 5 minutes", "Labels": "app=web"}', "")
                elif "docker inspect" in ' '.join(cmd):
                    # Simulate inspect failure - return error and no stdout
                    return (1, "", "inspect error")
                return (0, "", "")

            mock_run.side_effect = mock_run_cmd
            containers = discover_docker_containers()
            # Should still return the container from ps data
            assert len(containers) == 1
            assert "abc123def456" in containers


def test_discover_docker_containers_from_env():
    """Test discover_docker_containers respects OPERATOR_RUNTIME env var."""
    env = os.environ.copy()
    env["OPERATOR_RUNTIME"] = "kubernetes"

    with patch.dict(os.environ, env, clear=True):
        # Reset cache
        import predictive_agent.collector
        predictive_agent.collector._RUNTIME_CACHE = None
        with patch("predictive_agent.collector.run_cmd") as mock_run:
            mock_run.return_value = (0, "", "")
            containers = discover_docker_containers()
            # Should return empty because not in Docker mode
            assert containers == {}
            # docker ps should not be called
            mock_run.assert_not_called()


def test_discover_docker_containers_multiple_containers():
    """Test discover_docker_containers with multiple containers."""
    env = os.environ.copy()
    env["OPERATOR_RUNTIME"] = "docker"

    with patch.dict(os.environ, env, clear=True):
        # Reset cache
        import predictive_agent.collector
        predictive_agent.collector._RUNTIME_CACHE = None
        with patch("predictive_agent.collector.run_cmd") as mock_run:
            # Mock docker ps to return multiple containers
            ps_output = '\n'.join([
                '{"ID": "abc123def456", "Image": "nginx", "Names": "web-1", "Status": "Up 5 minutes", "Labels": "app=web,env=prod"}',
                '{"ID": "def456abc789", "Image": "redis", "Names": "cache-1", "Status": "Up 10 minutes", "Labels": "app=cache,env=prod"}',
            ])
            inspect_output_1 = '[{"Id": "abc123def456", "Name": "/web-1", "State": {"Running": true, "Paused": false, "Restarting": false}, "Config": {"Labels": {"app": "web", "env": "prod"}}, "NetworkSettings": {"Ports": {}, "Networks": {}}}]'
            inspect_output_2 = '[{"Id": "def456abc789", "Name": "/cache-1", "State": {"Running": true, "Paused": false, "Restarting": false}, "Config": {"Labels": {"app": "cache", "env": "prod"}}, "NetworkSettings": {"Ports": {}, "Networks": {}}}]'

            def mock_run_cmd(cmd, timeout=10):
                if "docker ps" in ' '.join(cmd):
                    return (0, ps_output, "")
                elif "docker inspect" in ' '.join(cmd):
                    if "abc123def456" in ' '.join(cmd):
                        return (0, inspect_output_1, "")
                    elif "def456abc789" in ' '.join(cmd):
                        return (0, inspect_output_2, "")
                return (1, "", "")

            mock_run.side_effect = mock_run_cmd
            containers = discover_docker_containers()

            assert len(containers) == 2
            assert "abc123def456" in containers
            assert "def456abc789" in containers
            assert containers["abc123def456"]["name"] == "web-1"
            assert containers["def456abc789"]["name"] == "cache-1"


def test_discover_docker_containers_by_label():
    """Test discover_docker_containers filtered by label."""
    env = os.environ.copy()
    env["OPERATOR_RUNTIME"] = "docker"
    env["OPERATOR_WATCH_LABELS"] = "app=web"

    with patch.dict(os.environ, env, clear=True):
        # Reset cache
        import predictive_agent.collector
        predictive_agent.collector._RUNTIME_CACHE = None
        with patch("predictive_agent.collector.run_cmd") as mock_run:
            # Mock docker ps to return multiple containers with different labels
            ps_output = '\n'.join([
                '{"ID": "abc123def456", "Image": "nginx", "Names": "web-1", "Status": "Up 5 minutes", "Labels": "app=web,env=prod"}',
                '{"ID": "def456abc789", "Image": "redis", "Names": "cache-1", "Status": "Up 10 minutes", "Labels": "app=cache,env=prod"}',
            ])
            inspect_output_1 = '[{"Id": "abc123def456", "Name": "/web-1", "State": {"Running": true}, "Config": {"Labels": {"app": "web", "env": "prod"}}, "NetworkSettings": {"Ports": {}, "Networks": {}}}]'
            inspect_output_2 = '[{"Id": "def456abc789", "Name": "/cache-1", "State": {"Running": true}, "Config": {"Labels": {"app": "cache", "env": "prod"}}, "NetworkSettings": {"Ports": {}, "Networks": {}}}]'

            def mock_run_cmd(cmd, timeout=10):
                if "docker ps" in ' '.join(cmd):
                    return (0, ps_output, "")
                elif "docker inspect" in ' '.join(cmd):
                    if "abc123def456" in ' '.join(cmd):
                        return (0, inspect_output_1, "")
                    elif "def456abc789" in ' '.join(cmd):
                        return (0, inspect_output_2, "")
                return (1, "", "")

            mock_run.side_effect = mock_run_cmd
            containers = discover_docker_containers()

            # Only web-1 should match the label selector app=web
            assert len(containers) == 1
            assert "abc123def456" in containers
            assert containers["abc123def456"]["name"] == "web-1"


def test_discover_docker_containers_by_compose_project():
    """Test discover_docker_containers filtered by compose project."""
    env = os.environ.copy()
    env["OPERATOR_RUNTIME"] = "docker"
    env["OPERATOR_WATCH_COMPOSE_PROJECTS"] = "myapp"

    with patch.dict(os.environ, env, clear=True):
        # Reset cache
        import predictive_agent.collector
        predictive_agent.collector._RUNTIME_CACHE = None
        with patch("predictive_agent.collector.run_cmd") as mock_run:
            # Mock docker ps to return containers with different compose projects
            ps_output = '\n'.join([
                '{"ID": "abc123def456", "Image": "nginx", "Names": "myapp-web-1", "Status": "Up 5 minutes", "Labels": "com.docker.compose.project=myapp,app=web"}',
                '{"ID": "def456abc789", "Image": "redis", "Names": "otherapp-cache-1", "Status": "Up 10 minutes", "Labels": "com.docker.compose.project=otherapp,app=cache"}',
            ])
            inspect_output_1 = '[{"Id": "abc123def456", "Name": "/myapp-web-1", "State": {"Running": true}, "Config": {"Labels": {"com.docker.compose.project": "myapp", "app": "web"}}, "NetworkSettings": {"Ports": {}, "Networks": {}}}]'
            inspect_output_2 = '[{"Id": "def456abc789", "Name": "/otherapp-cache-1", "State": {"Running": true}, "Config": {"Labels": {"com.docker.compose.project": "otherapp", "app": "cache"}}, "NetworkSettings": {"Ports": {}, "Networks": {}}}]'

            def mock_run_cmd(cmd, timeout=10):
                if "docker ps" in ' '.join(cmd):
                    return (0, ps_output, "")
                elif "docker inspect" in ' '.join(cmd):
                    if "abc123def456" in ' '.join(cmd):
                        return (0, inspect_output_1, "")
                    elif "def456abc789" in ' '.join(cmd):
                        return (0, inspect_output_2, "")
                return (1, "", "")

            mock_run.side_effect = mock_run_cmd
            containers = discover_docker_containers()

            # Only myapp container should match
            assert len(containers) == 1
            assert "abc123def456" in containers
            assert containers["abc123def456"]["compose_project"] == "myapp"


def test_discover_docker_containers_by_name():
    """Test discover_docker_containers filtered by name pattern."""
    env = os.environ.copy()
    env["OPERATOR_RUNTIME"] = "docker"
    env["OPERATOR_WATCH_NAMES"] = "web-*"

    with patch.dict(os.environ, env, clear=True):
        # Reset cache
        import predictive_agent.collector
        predictive_agent.collector._RUNTIME_CACHE = None
        with patch("predictive_agent.collector.run_cmd") as mock_run:
            # Mock docker ps to return containers with different names
            ps_output = '\n'.join([
                '{"ID": "abc123def456", "Image": "nginx", "Names": "web-1", "Status": "Up 5 minutes", "Labels": "app=web"}',
                '{"ID": "def456abc789", "Image": "redis", "Names": "cache-1", "Status": "Up 10 minutes", "Labels": "app=cache"}',
            ])
            inspect_output_1 = '[{"Id": "abc123def456", "Name": "/web-1", "State": {"Running": true}, "Config": {"Labels": {}}, "NetworkSettings": {"Ports": {}, "Networks": {}}}]'
            inspect_output_2 = '[{"Id": "def456abc789", "Name": "/cache-1", "State": {"Running": true}, "Config": {"Labels": {}}, "NetworkSettings": {"Ports": {}, "Networks": {}}}]'

            def mock_run_cmd(cmd, timeout=10):
                if "docker ps" in ' '.join(cmd):
                    return (0, ps_output, "")
                elif "docker inspect" in ' '.join(cmd):
                    if "abc123def456" in ' '.join(cmd):
                        return (0, inspect_output_1, "")
                    elif "def456abc789" in ' '.join(cmd):
                        return (0, inspect_output_2, "")
                return (1, "", "")

            mock_run.side_effect = mock_run_cmd
            containers = discover_docker_containers()

            # Only web-1 should match the name pattern web-*
            assert len(containers) == 1
            assert "abc123def456" in containers
            assert containers["abc123def456"]["name"] == "web-1"


def test_discover_docker_containers_no_selectors():
    """Test discover_docker_containers with no selectors returns all containers."""
    env = os.environ.copy()
    env["OPERATOR_RUNTIME"] = "docker"
    env.pop("OPERATOR_WATCH_LABELS", None)
    env.pop("OPERATOR_WATCH_COMPOSE_PROJECTS", None)
    env.pop("OPERATOR_WATCH_NAMES", None)

    with patch.dict(os.environ, env, clear=True):
        # Reset cache
        import predictive_agent.collector
        predictive_agent.collector._RUNTIME_CACHE = None
        with patch("predictive_agent.collector.run_cmd") as mock_run:
            # Mock docker ps to return multiple containers
            ps_output = '\n'.join([
                '{"ID": "abc123def456", "Image": "nginx", "Names": "web-1", "Status": "Up 5 minutes"}',
                '{"ID": "def456abc789", "Image": "redis", "Names": "cache-1", "Status": "Up 10 minutes"}',
            ])
            inspect_output_1 = '[{"Id": "abc123def456", "Name": "/web-1", "State": {"Running": true}, "Config": {"Labels": {}}, "NetworkSettings": {"Ports": {}, "Networks": {}}}]'
            inspect_output_2 = '[{"Id": "def456abc789", "Name": "/cache-1", "State": {"Running": true}, "Config": {"Labels": {}}, "NetworkSettings": {"Ports": {}, "Networks": {}}}]'

            def mock_run_cmd(cmd, timeout=10):
                if "docker ps" in ' '.join(cmd):
                    return (0, ps_output, "")
                elif "docker inspect" in ' '.join(cmd):
                    if "abc123def456" in ' '.join(cmd):
                        return (0, inspect_output_1, "")
                    elif "def456abc789" in ' '.join(cmd):
                        return (0, inspect_output_2, "")
                return (1, "", "")

            mock_run.side_effect = mock_run_cmd
            containers = discover_docker_containers()

            # All containers should be returned
            assert len(containers) == 2
            assert "abc123def456" in containers
            assert "def456abc789" in containers


def test_discover_docker_containers_ports_and_networks():
    """Test discover_docker_containers captures ports and networks."""
    env = os.environ.copy()
    env["OPERATOR_RUNTIME"] = "docker"

    with patch.dict(os.environ, env, clear=True):
        # Reset cache
        import predictive_agent.collector
        predictive_agent.collector._RUNTIME_CACHE = None
        with patch("predictive_agent.collector.run_cmd") as mock_run:
            ps_output = '{"ID": "abc123def456", "Image": "nginx", "Names": "web-1", "Status": "Up 5 minutes", "Labels": "app=web", "Ports": "0.0.0.0:80->80/tcp"}'
            inspect_output = '[{"Id": "abc123def456", "Name": "/web-1", "State": {"Running": true}, "Config": {"Labels": {}}, "NetworkSettings": {"Ports": {"80/tcp": [{"HostIp": "0.0.0.0", "HostPort": "80"}]}, "Networks": {"bridge": {"IPAddress": "172.17.0.2"}}}}]'

            def mock_run_cmd(cmd, timeout=10):
                if "docker ps" in ' '.join(cmd):
                    return (0, ps_output, "")
                elif "docker inspect" in ' '.join(cmd):
                    return (0, inspect_output, "")
                return (1, "", "")

            mock_run.side_effect = mock_run_cmd
            containers = discover_docker_containers()

            assert len(containers) == 1
            assert "abc123def456" in containers
            assert containers["abc123def456"]["ports"] == {"0.0.0.0:80": "80/tcp"}
            assert "bridge" in containers["abc123def456"]["networks"]


def test_discover_docker_containers_health_status():
    """Test discover_docker_containers captures health status."""
    env = os.environ.copy()
    env["OPERATOR_RUNTIME"] = "docker"

    with patch.dict(os.environ, env, clear=True):
        # Reset cache
        import predictive_agent.collector
        predictive_agent.collector._RUNTIME_CACHE = None
        with patch("predictive_agent.collector.run_cmd") as mock_run:
            ps_output = '{"ID": "abc123def456", "Image": "nginx", "Names": "web-1", "Status": "Up 5 minutes (healthy)", "Labels": "app=web"}'
            inspect_output = '[{"Id": "abc123def456", "Name": "/web-1", "State": {"Running": true, "Health": {"Status": "healthy", "FailingStreak": 0}}, "Config": {"Labels": {}}, "NetworkSettings": {"Ports": {}, "Networks": {}}}]'

            def mock_run_cmd(cmd, timeout=10):
                if "docker ps" in ' '.join(cmd):
                    return (0, ps_output, "")
                elif "docker inspect" in ' '.join(cmd):
                    return (0, inspect_output, "")
                return (1, "", "")

            mock_run.side_effect = mock_run_cmd
            containers = discover_docker_containers()

            assert len(containers) == 1
            assert containers["abc123def456"]["healthy"] is True


def test_discover_docker_containers_exited_state():
    """Test discover_docker_containers handles exited containers."""
    env = os.environ.copy()
    env["OPERATOR_RUNTIME"] = "docker"

    with patch.dict(os.environ, env, clear=True):
        # Reset cache
        import predictive_agent.collector
        predictive_agent.collector._RUNTIME_CACHE = None
        with patch("predictive_agent.collector.run_cmd") as mock_run:
            ps_output = '{"ID": "abc123def456", "Image": "nginx", "Names": "web-1", "Status": "Exited (0) 5 minutes ago", "Labels": "app=web"}'
            inspect_output = '[{"Id": "abc123def456", "Name": "/web-1", "State": {"Running": false, "Paused": false, "Restarting": false, "Dead": false, "ExitCode": 0}, "Config": {"Labels": {}}, "NetworkSettings": {"Ports": {}, "Networks": {}}}]'

            def mock_run_cmd(cmd, timeout=10):
                if "docker ps" in ' '.join(cmd):
                    return (0, ps_output, "")
                elif "docker inspect" in ' '.join(cmd):
                    return (0, inspect_output, "")
                return (1, "", "")

            mock_run.side_effect = mock_run_cmd
            containers = discover_docker_containers()

            assert len(containers) == 1
            assert containers["abc123def456"]["status"] == "exited"
            assert containers["abc123def456"]["exit_code"] == 0


def test_discover_docker_containers_inspect_error():
    """Test discover_docker_containers handles inspect errors gracefully."""
    env = os.environ.copy()
    env["OPERATOR_RUNTIME"] = "docker"

    with patch.dict(os.environ, env, clear=True):
        # Reset cache
        import predictive_agent.collector
        predictive_agent.collector._RUNTIME_CACHE = None
        with patch("predictive_agent.collector.run_cmd") as mock_run:
            ps_output = '{"ID": "abc123def456", "Image": "nginx", "Names": "web-1", "Status": "Up 5 minutes", "Labels": "app=web"}'

            def mock_run_cmd(cmd, timeout=10):
                if "docker ps" in ' '.join(cmd):
                    return (0, ps_output, "")
                elif "docker inspect" in ' '.join(cmd):
                    # Simulate inspect failure
                    return (1, "", "inspect error")
                return (1, "", "")

            mock_run.side_effect = mock_run_cmd
            containers = discover_docker_containers()

            # Should still return the container with ps data
            assert len(containers) == 1
            assert "abc123def456" in containers
            assert containers["abc123def456"]["name"] == "web-1"


def test_discover_docker_containers_lifecycle_removal():
    """Test discover_docker_containers handles container removal."""
    env = os.environ.copy()
    env["OPERATOR_RUNTIME"] = "docker"

    with patch.dict(os.environ, env, clear=True):
        # Reset cache
        import predictive_agent.collector
        predictive_agent.collector._RUNTIME_CACHE = None
        with patch("predictive_agent.collector.run_cmd") as mock_run:
            # First call: container exists
            ps_output_1 = '{"ID": "abc123def456", "Image": "nginx", "Names": "web-1", "Status": "Up 5 minutes", "Labels": "app=web"}'
            inspect_output = '[{"Id": "abc123def456", "Name": "/web-1", "State": {"Running": true}, "Config": {"Labels": {}}, "NetworkSettings": {"Ports": {}, "Networks": {}}}]'

            # Second call: container removed
            ps_output_2 = ""

            call_count = [0]
            def mock_run_cmd(cmd, timeout=10):
                if "docker ps" in ' '.join(cmd):
                    call_count[0] += 1
                    if call_count[0] == 1:
                        return (0, ps_output_1, "")
                    else:
                        return (0, ps_output_2, "")
                elif "docker inspect" in ' '.join(cmd):
                    return (0, inspect_output, "")
                return (1, "", "")

            mock_run.side_effect = mock_run_cmd

            # First discovery should find the container
            containers = discover_docker_containers()
            assert len(containers) == 1
            assert "abc123def456" in containers

            # Second discovery should not find the container
            containers = discover_docker_containers()
            assert len(containers) == 0


def test_discover_docker_containers_no_labels():
    """Test discover_docker_containers handles containers without labels."""
    env = os.environ.copy()
    env["OPERATOR_RUNTIME"] = "docker"
    env["OPERATOR_WATCH_LABELS"] = "app=web"

    with patch.dict(os.environ, env, clear=True):
        # Reset cache
        import predictive_agent.collector
        predictive_agent.collector._RUNTIME_CACHE = None
        with patch("predictive_agent.collector.run_cmd") as mock_run:
            # Container without labels
            ps_output = '{"ID": "abc123def456", "Image": "nginx", "Names": "web-1", "Status": "Up 5 minutes"}'
            inspect_output = '[{"Id": "abc123def456", "Name": "/web-1", "State": {"Running": true}, "Config": {"Labels": {}}, "NetworkSettings": {"Ports": {}, "Networks": {}}}]'

            def mock_run_cmd(cmd, timeout=10):
                if "docker ps" in ' '.join(cmd):
                    return (0, ps_output, "")
                elif "docker inspect" in ' '.join(cmd):
                    return (0, inspect_output, "")
                return (1, "", "")

            mock_run.side_effect = mock_run_cmd
            containers = discover_docker_containers()

            # Container without app=web label should not match
            assert len(containers) == 0


# === Helper function tests ===


def test_parse_docker_ps_line():
    """Test _parse_docker_ps_line with valid JSON."""
    line = '{"ID": "abc123", "Image": "nginx", "Names": "web-1"}'
    result = _parse_docker_ps_line(line)
    assert result == {"ID": "abc123", "Image": "nginx", "Names": "web-1"}


def test_parse_docker_ps_line_empty():
    """Test _parse_docker_ps_line with empty line."""
    result = _parse_docker_ps_line("")
    assert result is None


def test_parse_docker_ps_line_invalid():
    """Test _parse_docker_ps_line with invalid JSON."""
    result = _parse_docker_ps_line("not json")
    assert result is None


def test_matches_selectors_no_selectors():
    """Test _matches_selectors with no selectors returns True."""
    container = {"ID": "abc123", "Names": "web-1"}
    selectors = {"labels": {}, "compose_projects": [], "names": []}
    assert _matches_selectors(container, selectors) is True


def test_matches_selectors_by_label():
    """Test _matches_selectors filters by label."""
    container = {"ID": "abc123", "Names": "web-1", "Labels": "app=web,env=prod"}
    selectors = {"labels": {"app": "web"}, "compose_projects": [], "names": []}
    assert _matches_selectors(container, selectors) is True

    # Non-matching label
    selectors = {"labels": {"app": "api"}, "compose_projects": [], "names": []}
    assert _matches_selectors(container, selectors) is False


def test_matches_selectors_by_compose_project():
    """Test _matches_selectors filters by compose project."""
    container = {"ID": "abc123", "Names": "web-1", "Labels": "com.docker.compose.project=myapp"}
    selectors = {"labels": {}, "compose_projects": ["myapp"], "names": []}
    assert _matches_selectors(container, selectors) is True

    # Non-matching compose project
    selectors = {"labels": {}, "compose_projects": ["otherapp"], "names": []}
    assert _matches_selectors(container, selectors) is False


def test_matches_selectors_by_name():
    """Test _matches_selectors filters by name pattern."""
    container = {"ID": "abc123", "Names": "web-1"}
    selectors = {"labels": {}, "compose_projects": [], "names": ["web-*"]}
    assert _matches_selectors(container, selectors) is True

    # Non-matching name
    selectors = {"labels": {}, "compose_projects": [], "names": ["api-*"]}
    assert _matches_selectors(container, selectors) is False


def test_matches_selectors_name_with_slash():
    """Test _matches_selectors handles container names with leading slash."""
    container = {"ID": "abc123", "Names": "/web-1"}
    selectors = {"labels": {}, "compose_projects": [], "names": ["web-*"]}
    assert _matches_selectors(container, selectors) is True
