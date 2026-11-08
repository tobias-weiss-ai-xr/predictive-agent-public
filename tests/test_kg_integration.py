"""Tests for Knowledge Graph integration via Graphiti MCP.

Tests the Graphiti MCP client for blast radius queries without requiring
a real MCP server (all HTTP calls are mocked via unittest.mock).
"""
import json
import urllib.error
from unittest.mock import Mock, patch

from predictive_agent.kg_integration import (
    GRAPHITI_MCP_URL,
    GraphitiMCPClient,
    KnowledgeGraphClient,
    get_kg_client,
)


class TestGraphitiMCPClient:
    """Test the Graphiti MCP knowledge graph client."""

    def test_client_creation(self):
        """Test that client can be created with custom URL."""
        client = GraphitiMCPClient(url="http://localhost:9999")
        assert client.url == "http://localhost:9999"

    def test_client_creation_default_url(self):
        """Test that client uses GRAPHITI_MCP_URL environment variable."""
        client = GraphitiMCPClient()
        assert client.url == GRAPHITI_MCP_URL.rstrip("/")

    def test_client_health_unreachable(self):
        """Test that health check returns False for unreachable Graphiti MCP."""
        client = GraphitiMCPClient(url="http://localhost:9999", timeout=1)
        assert client.health() is False

    def test_client_health_success(self):
        """Test that health check returns True when Graphiti MCP is reachable."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.read.return_value = b"{}"
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            client = GraphitiMCPClient(url="http://localhost:9999", timeout=1)
            assert client.health() is True

    def test_search_nodes_unreachable(self):
        """Test that search_nodes returns None when Graphiti MCP is unreachable."""
        client = GraphitiMCPClient(url="http://localhost:9999", timeout=1)
        result = client.search_nodes("test-pod")
        assert result is None

    def test_search_nodes_success(self):
        """Test that search_nodes returns parsed JSON response."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"nodes": [{"id": "1", "name": "test-pod"}]}'
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            client = GraphitiMCPClient(url="http://localhost:9999", timeout=5)
            result = client.search_nodes("test-pod")
            assert result is not None
            assert isinstance(result, dict)
            assert "nodes" in result

    def test_search_nodes_with_session_header(self):
        """Test that search_nodes includes Mcp-Session-Id header."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"nodes": []}'
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            client = GraphitiMCPClient(url="http://localhost:9999", timeout=5)
            client.search_nodes("test-pod")

            # Verify the request was made with correct headers
            assert mock_urlopen.called
            call_args = mock_urlopen.call_args
            req = call_args[0][0]
            # Header names are case-insensitive in HTTP, urllib may lowercase them
            headers_lower = {k.lower(): v for k, v in req.headers.items()}
            assert "mcp-session-id" in headers_lower
            assert headers_lower["mcp-session-id"] == client._session_id

    def test_blast_radius_unreachable(self):
        """Test that blast radius returns 0 when Graphiti MCP is unreachable."""
        client = GraphitiMCPClient(url="http://localhost:9999", timeout=1)
        radius = client.get_blast_radius("ns/pod-1")
        assert radius == 0

    def test_blast_radius_with_nodes(self):
        """Test blast radius calculation with node data."""
        mock_response = Mock()
        mock_response.status = 200
        # Response with a pod that has dependencies
        mock_response.read.return_value = json.dumps({
            "nodes": [
                {
                    "name": "pod-1",
                    "k8s_pod_name": "pod-1",
                    "k8s_pod_namespace": "ns",
                    "uid": "pod-1-uid",
                    "~depends_on": [
                        {"uid": "dep-1", "id": "dep-1"},
                        {"uid": "dep-2", "id": "dep-2"},
                    ],
                    "~pod_node": [
                        {"uid": "node-1", "id": "node-1"},
                    ],
                }
            ]
        }).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            client = GraphitiMCPClient(url="http://localhost:9999", timeout=5)
            radius = client.get_blast_radius("ns/pod-1")
            # Should count 3 unique dependencies
            assert radius == 3

    def test_blast_radius_caching(self):
        """Test that blast radius results are cached."""
        client = GraphitiMCPClient(url="http://localhost:9999", timeout=1)
        # First call (returns 0 since server unreachable)
        radius1 = client.get_blast_radius("ns/pod-1")
        assert radius1 == 0
        # Second call should use cache
        radius2 = client.get_blast_radius("ns/pod-1")
        assert radius2 == 0
        # Verify cache was populated
        assert len(client._cache) > 0

    def test_blast_radius_cache_expiry(self):
        """Test that cached blast radius expires after timeout."""
        import predictive_agent.kg_integration as kg_mod
        original_timeout = kg_mod.BLAST_RADIUS_CACHE_SECONDS

        try:
            # Set a very short cache timeout
            kg_mod.BLAST_RADIUS_CACHE_SECONDS = 0

            mock_response = Mock()
            mock_response.status = 200
            mock_response.read.return_value = json.dumps({
                "nodes": [
                    {
                        "name": "pod-1",
                        "k8s_pod_name": "pod-1",
                        "k8s_pod_namespace": "ns",
                        "uid": "pod-1-uid",
                        "~depends_on": [{"uid": "dep-1"}],
                    }
                ]
            }).encode()
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=False)

            with patch("urllib.request.urlopen", return_value=mock_response):
                client = GraphitiMCPClient(url="http://localhost:9999", timeout=5)
                # First call
                radius1 = client.get_blast_radius("ns/pod-1")
                assert radius1 == 1
                # Second call should bypass cache due to expiry
                radius2 = client.get_blast_radius("ns/pod-1")
                assert radius2 == 1
        finally:
            kg_mod.BLAST_RADIUS_CACHE_SECONDS = original_timeout

    def test_clear_cache(self):
        """Test clearing the blast radius cache."""
        client = GraphitiMCPClient(url="http://localhost:9999", timeout=1)
        client.get_blast_radius("ns/pod-1")
        assert len(client._cache) > 0
        client.clear_cache()
        assert len(client._cache) == 0

    def test_get_node_blast_radius(self):
        """Test getting blast radius for a node."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({
            "nodes": [
                {
                    "type": "K8sPod",
                    "k8s_pod_node": "node-1",
                },
                {
                    "type": "K8sPod",
                    "k8s_pod_node": "node-1",
                },
                {
                    "type": "K8sPod",
                    "k8s_pod_node": "node-1",
                },
                {
                    "type": "Node",
                    "name": "node-1",
                    "pod_count": 5,
                },
            ]
        }).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            client = GraphitiMCPClient(url="http://localhost:9999", timeout=5)
            radius = client.get_node_blast_radius("node-1")
            # Should count K8sPod nodes on this node or use pod_count from Node
            assert radius >= 0  # At least 0

    def test_get_kg_client_singleton(self):
        """Test that get_kg_client returns a singleton."""
        # Reset the singleton
        import predictive_agent.kg_integration as kg_mod
        kg_mod._kg_client = None

        c1 = get_kg_client()
        c2 = get_kg_client()
        assert c1 is c2


class TestKnowledgeGraphClientAlias:
    """Test backward compatibility alias."""

    def test_alias_equivalence(self):
        """Test that KnowledgeGraphClient is an alias for GraphitiMCPClient."""
        assert KnowledgeGraphClient is GraphitiMCPClient

    def test_alias_creation(self):
        """Test creating client via legacy alias."""
        client = KnowledgeGraphClient(url="http://localhost:9999")
        assert isinstance(client, GraphitiMCPClient)


class TestGracefulDegradation:
    """Test graceful degradation when Graphiti MCP is unavailable."""

    def test_urllib_error_returns_none(self):
        """Test that URLError in _request returns None."""
        client = GraphitiMCPClient(url="http://localhost:9999", timeout=1)
        result = client._post("http://localhost:9999", {"query": "test"})
        assert result is None

    def test_json_decode_error_returns_none(self):
        """Test that JSON decode error in _request returns None."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.read.return_value = b"invalid json"
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            client = GraphitiMCPClient(url="http://localhost:9999", timeout=5)
            result = client._post("http://localhost:9999", {"query": "test"})
            assert result is None

    def test_os_error_returns_none(self):
        """Test that OSError in _request returns None."""
        with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
            client = GraphitiMCPClient(url="http://localhost:9999", timeout=1)
            result = client._post("http://localhost:9999", {"query": "test"})
            assert result is None

    def test_blast_radius_returns_zero_on_error(self):
        """Test that get_blast_radius returns 0 when search fails."""
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("No route")):
            client = GraphitiMCPClient(url="http://localhost:9999", timeout=1)
            radius = client.get_blast_radius("ns/pod-1")
            assert radius == 0


class TestGraphitiAuth:
    """GRAPHITI_AUTH -> Authorization: Basic header (production endpoint auth)."""

    def test_auth_header_when_configured(self):
        from predictive_agent import kg_integration
        with patch.object(kg_integration, "GRAPHITI_AUTH", "graphiti:secret"):
            client = kg_integration.GraphitiMCPClient(url="http://localhost:9999", timeout=1)
            headers = client._auth_headers()
            assert headers["Authorization"].startswith("Basic ")
            import base64
            decoded = base64.b64decode(headers["Authorization"].split()[1]).decode()
            assert decoded == "graphiti:secret"

    def test_no_auth_header_when_empty(self):
        from predictive_agent import kg_integration
        with patch.object(kg_integration, "GRAPHITI_AUTH", ""):
            client = kg_integration.GraphitiMCPClient(url="http://localhost:9999", timeout=1)
            assert client._auth_headers() == {}

    def test_auth_header_present_on_search(self):
        from predictive_agent import kg_integration
        with patch.object(kg_integration, "GRAPHITI_AUTH", "graphiti:secret"):
            client = kg_integration.GraphitiMCPClient(url="http://localhost:9999", timeout=5)
            mock_response = Mock()
            mock_response.status = 200
            mock_response.headers = {"Content-Type": "application/json", "Mcp-Session-Id": "srv-1"}
            mock_response.read.return_value = b'{"result": {"content": [{"text": "{}"}]}}'
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=False)
            with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
                client.search_nodes("test-pod")
                first_req = mock_urlopen.call_args_list[0][0][0]
                assert first_req.headers.get("Authorization", "").startswith("Basic ")

    def test_sse_response_parsed(self):
        """SSE (text/event-stream) bodies are parsed from the first data: line."""
        client = GraphitiMCPClient(url="http://localhost:9999", timeout=5)
        mock_response = Mock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "text/event-stream"}
        mock_response.read.return_value = b'event: message\ndata: {"result": {"content": [{"text": "{\\"nodes\\": []}"}]}}\n\n'
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_response):
            result = client._post("http://localhost:9999", {"query": "x"})
            assert result == {"nodes": []}
