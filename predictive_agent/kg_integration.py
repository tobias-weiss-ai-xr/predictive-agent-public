"""Knowledge graph integration for impact-aware risk scoring.

Queries the Graphiti MCP knowledge graph to determine the blast radius of a pod
— how many other pods, services, and namespaces depend on it. This feeds into
the risk scorer to escalate risk for pods with high dependency impact.

Uses the Graphiti MCP REST API (search_nodes endpoint) with Mcp-Session-Id header.
If Graphiti MCP is unavailable, all queries return 0 (graceful degradation).
"""

import base64
import json
import logging
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

GRAPHITI_MCP_URL = os.environ.get("GRAPHITI_MCP_URL", "http://127.0.0.1:8000")
# Backward compatibility alias
DGRAPH_URL = GRAPHITI_MCP_URL
BLAST_RADIUS_CACHE_SECONDS = int(os.environ.get("BLAST_RADIUS_CACHE_SECONDS", "300"))
# HTTP Basic credential "user:pass" — sent as an Authorization: Basic header on
# every request when non-empty. The production Graphiti MCP endpoint requires
# Basic Auth (since 2026-08-16).
GRAPHITI_AUTH = os.environ.get("GRAPHITI_AUTH", "")
GRAPHITI_TIMEOUT = int(os.environ.get("GRAPHITI_TIMEOUT", "5"))


class GraphitiMCPClient:
    """Thin Graphiti MCP HTTP client for blast-radius queries.

    Uses the Graphiti MCP REST API to search for nodes and their dependencies.
    Falls back gracefully when Graphiti MCP is not running.
    """

    def __init__(self, url: str = GRAPHITI_MCP_URL, timeout: int = GRAPHITI_TIMEOUT):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._session_id = str(uuid.uuid4())  # placeholder until server issues one
        self._initialized = False
        self._cache: dict[str, tuple[float, int]] = {}  # pod_key -> (timestamp, blast_radius)

    def _auth_headers(self) -> dict:
        """Authorization headers derived from GRAPHITI_AUTH (empty -> {})."""
        if not GRAPHITI_AUTH:
            return {}
        token = base64.b64encode(GRAPHITI_AUTH.encode()).decode()
        return {"Authorization": f"Basic {token}"}

    def _ensure_initialized(self) -> bool:
        """Run the MCP initialize handshake once (Streamable HTTP).

        Returns False when the endpoint is unreachable — callers degrade to 0.
        """
        if self._initialized:
            return True
        init = self._post(
            self.url,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "predictive-agent", "version": "4.0.0"},
                },
            },
            capture_session=True,
        )
        if init is None:
            return False
        # Complete the handshake; response body is ignored per spec.
        self._post(self.url, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        self._initialized = True
        return True

    def _parse_body(self, raw: bytes, content_type) -> Optional[dict]:
        """Parse a plain-JSON or SSE (data: lines) response body."""
        text = raw.decode()
        if isinstance(content_type, str) and "text/event-stream" in content_type:
            for line in text.splitlines():
                if line.startswith("data: "):
                    text = line[len("data: "):]
                    break
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return None
        # Unwrap a JSON-RPC tools/call envelope: {"result": {"content": [{"text": "..."}]}}
        result = parsed.get("result")
        if isinstance(result, dict):
            if "error" in parsed:
                logger.debug("Graphiti MCP JSON-RPC error: %s", parsed["error"])
                return None
            content = result.get("content")
            if isinstance(content, list) and content and isinstance(content[0], dict):
                text = content[0].get("text", "")
                try:
                    inner = json.loads(text)
                    if isinstance(inner, dict):
                        return inner
                except (json.JSONDecodeError, ValueError):
                    pass
            return result
        if "error" in parsed:
            logger.debug("Graphiti MCP JSON-RPC error: %s", parsed["error"])
            return None
        return parsed

    def _post(self, url: str, data: Optional[dict] = None,
              capture_session: bool = False) -> Optional[dict]:
        """POST a JSON payload to the Graphiti MCP endpoint.

        Returns None on any error (unreachable, auth failure, bad payload).
        When capture_session is True, adopts the server-issued Mcp-Session-Id.
        """
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            }
            headers.update(self._auth_headers())
            if not capture_session:
                headers["Mcp-Session-Id"] = self._session_id
            body = json.dumps(data).encode() if data is not None else None
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if capture_session:
                    session = resp.headers.get("Mcp-Session-Id")
                    if isinstance(session, str) and session:
                        self._session_id = session
                return self._parse_body(resp.read(), resp.headers.get("Content-Type"))
        except (urllib.error.URLError, json.JSONDecodeError, OSError, ValueError) as e:
            logger.debug("Graphiti MCP request failed: %s", e)
            return None

    def search_nodes(self, query: str, limit: int = 100) -> Optional[dict]:
        """Search for nodes in the knowledge graph using the Graphiti MCP API.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            API response as dict, or None on error
        """
        if not self._ensure_initialized():
            return None
        return self._post(
            self.url,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "search_nodes", "arguments": {"query": query, "limit": limit}},
            },
        )

    def get_blast_radius(self, pod_key: str) -> int:
        """Get the blast radius (dependent count) for a pod.

        This queries the knowledge graph for all nodes that depend on the
        given pod, either directly or transitively. Returns 0 if Graphiti MCP
        is unavailable or the pod is not in the graph.

        Args:
            pod_key: Pod identifier (namespace/name)

        Returns:
            Number of dependent nodes (pods, services, namespaces).
        """
        # Check cache
        cached = self._cache.get(pod_key)
        if cached is not None:
            ts, radius = cached
            if time.time() - ts < BLAST_RADIUS_CACHE_SECONDS:
                return radius

        # Query Graphiti MCP for dependent nodes
        # We search for the pod by name and then traverse dependencies
        namespace, _, name = pod_key.partition("/")

        # Search for the specific pod node
        search_result = self.search_nodes(f"{name}")
        if search_result is None:
            self._cache[pod_key] = (time.time(), 0)
            return 0

        # Extract nodes from the response
        # The Graphiti MCP API returns nodes in the response
        nodes = []
        if isinstance(search_result, dict):
            nodes = search_result.get("nodes", [])
            if not nodes:
                # Try other response formats
                nodes = search_result.get("data", [])
                if isinstance(nodes, dict):
                    nodes = nodes.get("nodes", [])

        if not nodes:
            self._cache[pod_key] = (time.time(), 0)
            return 0

        # Find our specific pod (match by name and namespace if available)
        target_node = None
        for node in nodes:
            if isinstance(node, dict):
                node_name = node.get("name", node.get("k8s_pod_name", ""))
                node_ns = node.get("namespace", node.get("k8s_pod_namespace", ""))
                if node_name == name and (not namespace or node_ns == namespace):
                    target_node = node
                    break

        if target_node is None:
            self._cache[pod_key] = (time.time(), 0)
            return 0

        # Count dependent nodes from the target node
        dependent_uids = set()

        # Look for dependency relationships in the response
        if isinstance(target_node, dict):
            # Check for direct dependencies
            deps = target_node.get("dependencies", [])
            for dep in deps:
                if isinstance(dep, dict):
                    uid = dep.get("uid", dep.get("id", ""))
                    if uid:
                        dependent_uids.add(uid)

            # Check for reverse dependencies (~depends_on, ~pod_node)
            rev_deps = target_node.get("~depends_on", [])
            for dep in rev_deps:
                if isinstance(dep, dict):
                    uid = dep.get("uid", dep.get("id", ""))
                    if uid:
                        dependent_uids.add(uid)

            rev_node_deps = target_node.get("~pod_node", [])
            for dep in rev_node_deps:
                if isinstance(dep, dict):
                    uid = dep.get("uid", dep.get("id", ""))
                    if uid:
                        dependent_uids.add(uid)

        radius = len(dependent_uids)
        self._cache[pod_key] = (time.time(), radius)
        return radius

    def get_node_blast_radius(self, node_name: str) -> int:
        """Get blast radius for a node (all pods on that node)."""
        search_result = self.search_nodes(node_name)
        if search_result is None:
            return 0

        nodes = []
        if isinstance(search_result, dict):
            nodes = search_result.get("nodes", [])
            if not nodes:
                nodes = search_result.get("data", [])
                if isinstance(nodes, dict):
                    nodes = nodes.get("nodes", [])

        # Count pods on this node
        pod_count = 0
        for node in nodes:
            if isinstance(node, dict):
                node_type = node.get("type", node.get("dgraph.type", ""))
                node_val = node.get("k8s_pod_node", "")
                if node_type == "K8sPod" and node_val == node_name:
                    pod_count += 1
                elif node_type == "Node" and node.get("name", "") == node_name:
                    # If this is the node itself, it might have pod count
                    pod_count = int(node.get("pod_count", node.get("count", 0)) or 0)
                    break

        return pod_count

    def health(self) -> bool:
        """Check if Graphiti MCP is reachable."""
        try:
            req = urllib.request.Request(
                f"{self.url}/health",
                headers={"Mcp-Session-Id": self._session_id},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def clear_cache(self) -> None:
        """Clear the blast radius cache."""
        self._cache.clear()


# Backward compatibility alias
KnowledgeGraphClient = GraphitiMCPClient


# Singleton instance (lazy-initialized)
_kg_client: Optional[GraphitiMCPClient] = None


def get_kg_client() -> Optional[GraphitiMCPClient]:
    """Get the singleton GraphitiMCPClient instance.

    Returns None if GRAPHITI_MCP_URL is not set or Graphiti MCP is unreachable on init.
    The client still works — it returns 0 for all queries when Graphiti MCP is down.
    """
    global _kg_client
    if _kg_client is None:
        _kg_client = GraphitiMCPClient()
    return _kg_client
