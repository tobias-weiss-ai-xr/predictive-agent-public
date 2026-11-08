"""Edge-case tests for predictive_agent.kg_integration.py.

Exercises the branch behaviour of ``GraphitiMCPClient`` that the mainline suite
does not reach — the MCP JSON-RPC response-parsing corner cases, the
query/lookup error paths (unreachable server, HTTP errors, timeouts) and the
module-level config/singleton edges. Only the real HTTP boundary
(``urllib.request.urlopen``) is stubbed; no real network calls, no sleeps.
"""

import json
import urllib.error
from unittest.mock import Mock, patch

from predictive_agent import kg_integration
from predictive_agent.kg_integration import (
    DGRAPH_URL,
    GRAPHITI_MCP_URL,
    GraphitiMCPClient,
    get_kg_client,
)


def _response(body, content_type="application/json"):
    """Build a urllib.urlopen-style mock response around a JSON body string."""
    mock = Mock()
    mock.status = 200
    mock.headers = {"Content-Type": content_type}
    mock.read.return_value = body.encode("utf-8")
    mock.__enter__ = Mock(return_value=mock)
    mock.__exit__ = Mock(return_value=False)
    return mock


def _ready_client(url="http://mcp.test:8000", timeout=5):
    """Client with the initialize handshake already done (one POST per query)."""
    client = GraphitiMCPClient(url=url, timeout=timeout)
    client._initialized = True
    return client


# ────────────────────────────────────────────────────────────────────────────
# _parse_body: non-dict JSON payloads
# ────────────────────────────────────────────────────────────────────────────
class TestParseBodyNonDict:
    """A JSON body that is not a dict must parse to None (no crash)."""

    def test_list_body_returns_none(self):
        client = _ready_client()
        assert client._parse_body(b"[1, 2, 3]", "application/json") is None

    def test_string_body_returns_none(self):
        client = _ready_client()
        assert client._parse_body(b'"just a string"', "application/json") is None

    def test_number_body_returns_none(self):
        client = _ready_client()
        assert client._parse_body(b"42", "application/json") is None


# ────────────────────────────────────────────────────────────────────────────
# _parse_body: JSON-RPC error keys
# ────────────────────────────────────────────────────────────────────────────
class TestParseBodyErrorKeys:
    """An \"error\" member — with or without a result — degrades to None."""

    def test_result_present_with_error_returns_none(self):
        client = _ready_client()
        raw = b'{"result": {"content": [{"text": "{\\"nodes\\": []}"}]}, "error": {"code": -32000, "message": "boom"}}'
        assert client._parse_body(raw, "application/json") is None

    def test_top_level_error_returns_none(self):
        client = _ready_client()
        raw = b'{"error": {"code": -32601, "message": "Method not found"}}'
        assert client._parse_body(raw, "application/json") is None


# ────────────────────────────────────────────────────────────────────────────
# _parse_body: inner-JSON extraction fallback
# ────────────────────────────────────────────────────────────────────────────
class TestParseBodyInnerJsonFallthrough:
    """content[0].text that is not a decodable dict falls back to raw result."""

    def test_inner_json_decode_error_returns_raw_result(self):
        client = _ready_client()
        raw = b'{"result": {"content": [{"text": "{not-json"}]}}'
        # JSONDecodeError is absorbed; the raw result dict is returned, not None.
        assert client._parse_body(raw, "application/json") == {"content": [{"text": "{not-json"}]}

    def test_inner_json_non_dict_returns_raw_result(self):
        client = _ready_client()
        raw = b'{"result": {"content": [{"text": "[1, 2, 3]"}]}}'
        # Valid JSON but not a dict -> falls through to the raw result.
        assert client._parse_body(raw, "application/json") == {"content": [{"text": "[1, 2, 3]"}]}

    def test_inner_json_empty_text_returns_raw_result(self):
        client = _ready_client()
        raw = b'{"result": {"content": [{"text": ""}]}}'
        assert client._parse_body(raw, "application/json") == {"content": [{"text": ""}]}

    def test_plain_json_without_result_is_returned(self):
        client = _ready_client()
        assert client._parse_body(b'{"foo": 1}', "application/json") == {"foo": 1}


# ────────────────────────────────────────────────────────────────────────────
# get_blast_radius: alternate response shapes ("data" fallback)
# ────────────────────────────────────────────────────────────────────────────
class TestBlastRadiusDataFallback:
    """Responses without a \"nodes\" key fall back to \"data\"."""

    def test_data_list_fallback(self):
        body = json.dumps(
            {
                "data": [
                    {
                        "name": "pod-1",
                        "namespace": "ns",
                        "dependencies": [{"uid": "d1"}],
                    }
                ]
            }
        )
        with patch("urllib.request.urlopen", return_value=_response(body)):
            client = _ready_client()
            assert client.get_blast_radius("ns/pod-1") == 1

    def test_data_dict_wrap_fallback(self):
        body = json.dumps(
            {
                "data": {
                    "nodes": [
                        {
                            "name": "pod-1",
                            "namespace": "ns",
                            "dependencies": [{"uid": "d1"}, {"uid": "d2"}],
                        }
                    ]
                }
            }
        )
        with patch("urllib.request.urlopen", return_value=_response(body)):
            client = _ready_client()
            assert client.get_blast_radius("ns/pod-1") == 2


# ────────────────────────────────────────────────────────────────────────────
# get_blast_radius: empty node lists
# ────────────────────────────────────────────────────────────────────────────
class TestBlastRadiusEmptyNodes:
    """No nodes in the response -> 0, and the result is cached."""

    def test_empty_nodes_returns_zero_and_caches(self):
        with patch("urllib.request.urlopen", return_value=_response('{"nodes": []}')):
            client = _ready_client()
            assert client.get_blast_radius("ns/pod-1") == 0
            assert client._cache["ns/pod-1"][1] == 0

    def test_empty_data_list_returns_zero(self):
        with patch("urllib.request.urlopen", return_value=_response('{"data": []}')):
            client = _ready_client()
            assert client.get_blast_radius("ns/pod-1") == 0
            assert client._cache["ns/pod-1"][1] == 0


# ────────────────────────────────────────────────────────────────────────────
# get_blast_radius: target pod not found
# ────────────────────────────────────────────────────────────────────────────
class TestBlastRadiusTargetNotFound:
    """The queried pod is absent / wrong namespace -> 0, cached."""

    def test_target_not_in_response_returns_zero(self):
        body = json.dumps({"nodes": [{"name": "other-pod", "namespace": "ns"}]})
        with patch("urllib.request.urlopen", return_value=_response(body)):
            client = _ready_client()
            assert client.get_blast_radius("ns/pod-1") == 0
            assert client._cache["ns/pod-1"][1] == 0

    def test_namespace_mismatch_returns_zero(self):
        body = json.dumps({"nodes": [{"name": "pod-1", "namespace": "other-ns"}]})
        with patch("urllib.request.urlopen", return_value=_response(body)):
            client = _ready_client()
            assert client.get_blast_radius("ns/pod-1") == 0


# ────────────────────────────────────────────────────────────────────────────
# get_blast_radius: dependency counting
# ────────────────────────────────────────────────────────────────────────────
class TestBlastRadiusDependencies:
    """The \"dependencies\" list is counted with uid dedup and type guard."""

    def test_counts_dict_dependencies_and_skips_invalid(self):
        body = json.dumps(
            {
                "nodes": [
                    {
                        "name": "pod-1",
                        "namespace": "ns",
                        "dependencies": [
                            {"uid": "d1"},
                            {"uid": "d2"},
                            {"uid": "d3"},
                            "not-a-dict",  # skipped (not a dict)
                            {"id": "no-uid"},  # uid absent -> id fallback, counted
                            {"uid": "d2"},  # duplicate uid, deduped
                        ],
                    }
                ]
            }
        )
        with patch("urllib.request.urlopen", return_value=_response(body)):
            client = _ready_client()
            assert client.get_blast_radius("ns/pod-1") == 4

    def test_only_non_dict_dependencies_count_zero(self):
        body = json.dumps(
            {
                "nodes": [
                    {
                        "name": "pod-1",
                        "namespace": "ns",
                        "dependencies": ["str-dep", 42, None],
                    }
                ]
            }
        )
        with patch("urllib.request.urlopen", return_value=_response(body)):
            client = _ready_client()
            assert client.get_blast_radius("ns/pod-1") == 0


# ────────────────────────────────────────────────────────────────────────────
# get_node_blast_radius: error paths and alternate shapes
# ────────────────────────────────────────────────────────────────────────────
class TestNodeBlastRadiusEdges:
    """Node blast radius: unreachable server / timeout / data fallback."""

    def test_unreachable_server_returns_zero(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no route")):
            client = _ready_client()
            assert client.get_node_blast_radius("node-1") == 0

    def test_timeout_returns_zero(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            client = _ready_client()
            assert client.get_node_blast_radius("node-1") == 0

    def test_data_list_fallback_counts_pods(self):
        body = json.dumps({"data": [{"type": "K8sPod", "k8s_pod_node": "node-1"}]})
        with patch("urllib.request.urlopen", return_value=_response(body)):
            client = _ready_client()
            assert client.get_node_blast_radius("node-1") == 1

    def test_data_dict_wrap_fallback_counts_pods(self):
        body = json.dumps(
            {
                "data": {
                    "nodes": [
                        {"type": "K8sPod", "k8s_pod_node": "node-1"},
                        {"type": "K8sPod", "k8s_pod_node": "node-1"},
                        {"type": "K8sPod", "k8s_pod_node": "node-2"},  # other node
                    ]
                }
            }
        )
        with patch("urllib.request.urlopen", return_value=_response(body)):
            client = _ready_client()
            assert client.get_node_blast_radius("node-1") == 2


# ────────────────────────────────────────────────────────────────────────────
# _post network error absorption
# ────────────────────────────────────────────────────────────────────────────
class TestPostNetworkErrors:
    """HTTP errors and timeouts at the transport boundary degrade to None."""

    def test_http_error_returns_none(self):
        server_error = urllib.error.HTTPError("http://mcp.test:8000", 500, "Internal Server Error", {}, None)
        with patch("urllib.request.urlopen", side_effect=server_error):
            client = _ready_client()
            assert client._post("http://mcp.test:8000", {"a": 1}) is None

    def test_timeout_returns_none(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("read timed out")):
            client = _ready_client()
            assert client._post("http://mcp.test:8000", {"a": 1}) is None


# ────────────────────────────────────────────────────────────────────────────
# Module-level config + singleton
# ────────────────────────────────────────────────────────────────────────────
class TestConfigAndSingleton:
    """DGRAPH_URL backward-compat alias and the lazy singleton."""

    def test_dgraph_url_alias_matches_graphiti_url(self):
        assert DGRAPH_URL == GRAPHITI_MCP_URL

    def test_get_kg_client_returns_shared_singleton(self):
        kg_integration._kg_client = None
        try:
            first = get_kg_client()
            second = get_kg_client()
            assert first is second
            assert isinstance(first, GraphitiMCPClient)
        finally:
            kg_integration._kg_client = None
