"""Docker container restart action: restart unhealthy Docker containers.

Restores unhealthy Docker containers that have excessive restart counts or are
not running. Two execution paths (union of AIO-2 + crash-loop workstreams):
  * injected DockerSocketClient -> Docker Engine API over unix socket (Linux)
  * no client -> docker CLI subprocess (`docker restart <name>`), Windows-safe
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
from typing import Optional

from predictive_agent import config
from predictive_agent.remediator import ActionContext, RemediationAction, RemediationResult

logger = logging.getLogger(__name__)

# Thresholds (mirror pod_restart.py but with Docker-style checks)
MIN_RESTART_COUNT = 5
RISK_THRESHOLD = float(
    os.environ.get("REMEDIATION_RISK_THRESHOLD", "0.7")
)

# Protected namespaces from config
REMEDIATION_PROTECTED_NS = {
    ns.strip()
    for ns in os.environ.get(
        "REMEDIATION_PROTECTED_NS", "kube-system,opendesk-predictive-agent"
    ).split(",")
    if ns.strip()
}


class DockerSocketClient:
    """HTTP client for Docker Engine API over unix socket.
    
    Uses stdlib http.client over AF_UNIX socket to communicate with Docker daemon.
    Mirrors the urllib style used in llm.py.
    """

    def __init__(self, socket_path: Optional[str] = None):
        """Initialize Docker socket client.
        
        Args:
            socket_path: Path to Docker socket (default from DOCKER_SOCKET env)
        """
        self.socket_path = socket_path or config.DOCKER_SOCKET

    def _get_connection(self) -> tuple[socket.socket, str]:
        """Get a connected socket and HTTP host header value.
        
        Returns:
            Tuple of (connected socket, host header string)
        Raises:
            FileNotFoundError: If socket path doesn't exist
            PermissionError: If socket is not accessible
            OSError: For other socket errors
        """
        if not os.path.exists(self.socket_path):
            raise FileNotFoundError(
                f"Docker socket not found: {self.socket_path}"
            )

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(self.socket_path)
        except PermissionError:
            raise PermissionError(
                f"Permission denied accessing Docker socket: {self.socket_path}"
            )
        except OSError as e:
            raise OSError(f"Failed to connect to Docker socket: {e}")

        # Docker API expects a simple Host header over the unix socket
        # (the literal "unix" works; a "unix://<path>" value is rejected
        # with HTTP 400 by the real daemon).
        host = "unix"
        return sock, host

    def request(
        self,
        method: str,
        path: str,
        data: Optional[bytes] = None,
        headers: Optional[dict] = None,
        timeout: int = 30,
    ) -> tuple[int, dict, str]:
        """Make an HTTP request to Docker API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., "/containers/json")
            data: Optional request body as bytes
            headers: Optional dict of HTTP headers
            timeout: Request timeout in seconds
        
        Returns:
            Tuple of (status_code, response_headers, response_body)
        
        Raises:
            ConnectionError: If cannot connect to Docker socket
            Exception: For other errors
        """
        sock = None
        try:
            sock, host = self._get_connection()

            default_headers = {
                "Host": host,
                "Accept": "application/json",
            }
            if headers:
                default_headers.update(headers)

            # For POST with data, add Content-Type and Content-Length
            if data:
                default_headers["Content-Type"] = "application/json"
                default_headers["Content-Length"] = str(len(data))

            sock.settimeout(timeout)

            # Use http.client over the connected socket
            # We need to manually send the HTTP request
            request_line = f"{method} {path} HTTP/1.1\r\n"
            header_lines = "".join(
                f"{k}: {v}\r\n" for k, v in default_headers.items()
            )
            request = request_line + header_lines + "\r\n"

            if data:
                request += data.decode("utf-8", errors="replace")

            sock.sendall(request.encode("utf-8", errors="replace"))

            # Read response status line
            status_line = ""
            while True:
                chunk = sock.recv(1).decode("utf-8", errors="replace")
                if chunk == "\n" or not chunk:
                    break
                status_line += chunk

            # Parse status line: "HTTP/1.1 200 OK"
            status_parts = status_line.strip().split(" ", 2)
            if len(status_parts) < 2:
                raise ValueError(f"Invalid HTTP response: {status_line}")

            status_code = int(status_parts[1])

            # Read headers
            response_headers = {}
            while True:
                line = ""
                while True:
                    chunk = sock.recv(1).decode("utf-8", errors="replace")
                    if chunk == "\n":
                        break
                    line += chunk
                line = line.strip()
                if not line:
                    break
                if ": " in line:
                    key, value = line.split(": ", 1)
                    response_headers[key] = value

            # Read body. Docker returns list/stream endpoints with
            # Transfer-Encoding: chunked (no Content-Length), so handle both.
            body = ""
            te = response_headers.get("Transfer-Encoding", "")
            if isinstance(te, str) and "chunked" in te.lower():
                body_bytes = b""
                while True:
                    size_line = ""
                    while True:
                        c = sock.recv(1)
                        if not c or c == b"\n":
                            break
                        size_line += c.decode("utf-8", errors="replace")
                    size_line = size_line.strip()
                    if not size_line:
                        continue
                    try:
                        chunk_size = int(size_line.split(";")[0], 16)
                    except ValueError:
                        break
                    if chunk_size == 0:
                        sock.recv(2)  # trailing CRLF
                        break
                    chunk_data = b""
                    while len(chunk_data) < chunk_size:
                        got = sock.recv(chunk_size - len(chunk_data))
                        if not got:
                            break
                        chunk_data += got
                    body_bytes += chunk_data
                    sock.recv(2)  # CRLF after chunk
                body = body_bytes.decode("utf-8", errors="replace")
            else:
                content_length = int(response_headers.get("Content-Length", 0))
                if content_length > 0:
                    body_bytes = b""
                    remaining = content_length
                    while remaining > 0:
                        piece = sock.recv(min(remaining, 8192))
                        if not piece:
                            break
                        body_bytes += piece
                        remaining -= len(piece)
                    body = body_bytes.decode("utf-8", errors="replace")

            return status_code, response_headers, body

        except TimeoutError:
            raise TimeoutError(f"Docker API request timed out after {timeout}s")
        except Exception as e:
            raise ConnectionError(f"Docker API request failed: {e}")
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass


class ContainerRestartAction(RemediationAction):
    """Restart unhealthy Docker containers to recover from failures.

    Uses the docker CLI (`docker restart <name>`) to restart containers that meet
    the remediation criteria: high risk score, excessive restarts, or not running state.
    
    This action is only registered when detect_runtime() returns "docker".
    """

    name = "container_restart"
    # Restarting a running container is safe to re-run (P3 idempotent remediation).
    idempotent: bool = True
    # P6 bounded autonomy: risk tier of this action
    risk_tier: str = "safe"

    def __init__(self, docker_client: Optional[DockerSocketClient] = None,
                 socket_path: Optional[str] = None):
        """Initialize action.

        Args:
            docker_client: Optional DockerSocketClient instance. When provided,
                execute() talks to the Docker Engine API over the unix socket
                (Linux daemons). When omitted, execute() shells out to the
                docker CLI, which also works on Windows hosts without AF_UNIX.
            socket_path: Optional explicit socket path (creates a client).
        """
        self.docker_client = docker_client
        if self.docker_client is None and socket_path:
            self.docker_client = DockerSocketClient(socket_path)

    def should_execute(self, pod_state, prediction, risk_score: float) -> bool:
        """Check if container should be restarted.

        Returns True when:
        - risk_score > REMEDIATION_RISK_THRESHOLD
        - restart_count >= 5 OR container is not running
        - namespace NOT in REMEDIATION_PROTECTED_NS
        """
        if risk_score <= RISK_THRESHOLD:
            return False

        namespace = getattr(pod_state, "namespace", "") or ""
        if namespace in REMEDIATION_PROTECTED_NS:
            return False

        restart_count = getattr(pod_state, "restart_count", 0) or 0

        # Check if container is not running
        # In the docker collector, running state would be tracked in state attribute
        # Use explicit checks to handle both real objects and test mocks
        is_running = True

        running_attr = getattr(pod_state, "running", None)
        if running_attr is not None and running_attr is not True:
            is_running = False

        # Also check state for "running" vs other states
        state = getattr(pod_state, "state", None)
        if state is not None:
            try:
                if state.lower() != "running":
                    is_running = False
            except (AttributeError, TypeError):
                # state is not a string (e.g., MagicMock in tests)
                pass

        # For Docker, also check status field
        status = getattr(pod_state, "status", None)
        if status is not None:
            try:
                if status.lower() not in ("running", "healthy"):
                    is_running = False
            except (AttributeError, TypeError):
                # status is not a string (e.g., MagicMock in tests)
                pass

        if restart_count >= MIN_RESTART_COUNT:
            return True
        if not is_running:
            return True

        return False

    def _get_container_id(self, target: str, namespace: str) -> Optional[str]:
        """Get Docker container ID by name.
        
        Args:
            target: Container name
            namespace: Compose project name (used as label filter)
        
        Returns:
            Container ID if found, None otherwise
        """
        client = self.docker_client
        try:
            if client is None:
                raise RuntimeError("Docker socket client not configured")
            status_code, headers, body = client.request(
                "GET",
                "/containers/json?all=1"
            )

            if status_code != 200:
                # Try without label filter
                status_code, headers, body = client.request(
                    "GET", "/containers/json?all=1"
                )

            if status_code == 200 and body:
                containers = json.loads(body)
                for container in containers:
                    names = container.get("Names", [])
                    # Container names have leading /, so we need to match
                    for name in names:
                        # name is like "/project_service_1"
                        if name.startswith("/"):
                            name = name[1:]
                        # Compose-style names are "{project}_{service}_{index}",
                        # e.g. "monitoring_predictive-agent_1". Match if the
                        # container name equals the target or references it as a
                        # compose service token. The reconcile loop passes the
                        # pod_key as "ns/name", so also try the bare name.
                        bare_target = target.split("/")[-1]
                        if (
                            name in (target, bare_target)
                            or name.endswith("/" + target)
                            or name.endswith("/" + bare_target)
                        ):
                            return container.get("Id", "")
                        if target and (
                            name.startswith(f"{namespace}_{target}_")
                            or name.startswith(f"{namespace}-{target}-")
                            or f"_{target}_" in name
                            or f"-{target}-" in name
                        ):
                            return container.get("Id", "")
        except Exception as e:
            logger.error("Failed to get container ID for %s: %s", target, e)

        return None

    def _execute_via_socket(self, target: str, context: ActionContext) -> RemediationResult:
        """Restart the Docker container via the Docker Engine API (unix socket).

        POSTs to /containers/{id}/restart via the Docker socket.
        In dry_run mode, logs the action but doesn't execute.
        Returns a human-readable command string like 'docker restart <name>'.
        """
        namespace = context.namespace or getattr(context, "namespace", "")
        cmd_str = f"docker restart {target}"

        if context.dry_run:
            return RemediationResult(
                action=self.name,
                target=target,
                success=True,
                dry_run=True,
                message=f"Would restart container {target} (dry run)",
                command=cmd_str,
            )

        try:
            # Get container ID
            container_id = self._get_container_id(target, namespace)
            if not container_id:
                return RemediationResult(
                    action=self.name,
                    target=target,
                    success=False,
                    dry_run=False,
                    message=f"Container {target} not found",
                    command=cmd_str,
                )

            # Restart container via Docker API
            # POST /containers/{id}/restart?t=1 (t=1 means wait 1 second before killing)
            restart_client = self.docker_client
            if restart_client is None:
                raise RuntimeError("Docker socket client not configured")
            status_code, headers, body = restart_client.request(
                "POST",
                f"/containers/{container_id}/restart?t=1",
                timeout=30,
            )

            if status_code in (200, 204):
                return RemediationResult(
                    action=self.name,
                    target=target,
                    success=True,
                    dry_run=False,
                    message=f"Container {target} restarted successfully",
                    command=cmd_str,
                )
            else:
                return RemediationResult(
                    action=self.name,
                    target=target,
                    success=False,
                    dry_run=False,
                    message=f"Failed to restart container: HTTP {status_code}",
                    command=cmd_str,
                )

        except TimeoutError as e:
            return RemediationResult(
                action=self.name,
                target=target,
                success=False,
                dry_run=False,
                message=f"Container restart timed out: {e}",
                command=cmd_str,
            )
        except FileNotFoundError as e:
            return RemediationResult(
                action=self.name,
                target=target,
                success=False,
                dry_run=False,
                message=f"Docker socket not available: {e}",
                command=cmd_str,
            )
        except PermissionError as e:
            return RemediationResult(
                action=self.name,
                target=target,
                success=False,
                dry_run=False,
                message=f"Permission denied for Docker socket: {e}",
                command=cmd_str,
            )
        except Exception as e:
            return RemediationResult(
                action=self.name,
                target=target,
                success=False,
                dry_run=False,
                message=f"Error restarting container: {e}",
                command=cmd_str,
            )

    def execute(self, target: str, context: ActionContext) -> RemediationResult:
        """Restart the container: via Docker socket when a client is injected,
        otherwise via the docker CLI subprocess."""
        if self.docker_client is not None:
            return self._execute_via_socket(target, context)
        return self._execute_via_cli(target, context)

    def _execute_via_cli(self, target: str, context: ActionContext) -> RemediationResult:
        """Restart the Docker container via CLI.

        Executes `docker restart <target>` using subprocess.
        In dry_run mode, logs the action but doesn't execute.
        Returns a RemediationResult with the docker command string.
        """
        cmd = ["docker", "restart", target]
        cmd_str = " ".join(cmd)

        if context.dry_run:
            return RemediationResult(
                action=self.name,
                target=target,
                success=True,
                dry_run=True,
                message=f"Would restart container {target} (dry run)",
                command=cmd_str,
            )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            success = result.returncode == 0
            message = result.stdout.strip() if success else result.stderr.strip()
            if not message:
                message = f"container {target} restarted" if success else f"docker restart failed for {target}"

            return RemediationResult(
                action=self.name,
                target=target,
                success=success,
                dry_run=context.dry_run,
                message=message,
                command=cmd_str,
            )
        except subprocess.TimeoutExpired:
            return RemediationResult(
                action=self.name,
                target=target,
                success=False,
                dry_run=context.dry_run,
                message=f"docker restart {target} timed out after 30s",
                command=cmd_str,
            )
        except FileNotFoundError as e:
            return RemediationResult(
                action=self.name,
                target=target,
                success=False,
                dry_run=context.dry_run,
                message=f"docker CLI not found: {e}",
                command=cmd_str,
            )
        except PermissionError as e:
            return RemediationResult(
                action=self.name,
                target=target,
                success=False,
                dry_run=context.dry_run,
                message=f"Permission denied executing docker CLI: {e}",
                command=cmd_str,
            )
        except Exception as e:
            return RemediationResult(
                action=self.name,
                target=target,
                success=False,
                dry_run=context.dry_run,
                message=f"Error restarting container: {e}",
                command=cmd_str,
            )
