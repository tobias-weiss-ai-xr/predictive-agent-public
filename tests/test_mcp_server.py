"""MCP (Model Context Protocol) server tests.

Covers the dependency-free JSON-RPC core (``mcp_server.handle_request``) and
the streamable-HTTP route wired into ``server.py`` (``POST /mcp``). The MCP
surface is intentionally read-only: every tool is a snapshot of agent state so
an AI assistant querying the operator never mutates the cluster.
"""
import json
import urllib.request

import pytest

from predictive_agent.mcp_server import (
    PROTOCOL_VERSION,
    Tool,
    handle_request,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _tools():
    return {
        "get_status": Tool(
            name="get_status",
            description="Operator runtime status and counters.",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=lambda args: {"status": "ok", "reconciles": 42},
        ),
        "get_predictions": Tool(
            name="get_predictions",
            description="Current per-pod risk predictions.",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=lambda args: {"predictions": [], "total": 0},
        ),
    }


def _call(payload, tools=None):
    return handle_request(payload, tools or _tools())


# ── Protocol core ────────────────────────────────────────────────────────────


def test_initialize_returns_protocol_and_capabilities():
    resp = _call({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"},
        },
    })
    assert resp["id"] == 1
    result = resp["result"]
    assert result["protocolVersion"] == PROTOCOL_VERSION
    assert "tools" in result["capabilities"]
    assert result["serverInfo"]["name"] == "predictive-agent"


def test_initialized_notification_gets_no_reply():
    assert _call({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_ping_returns_empty_result():
    resp = _call({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    assert resp["result"] == {}


def test_tools_list_returns_registry():
    resp = _call({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
    tools = resp["result"]["tools"]
    names = {t["name"] for t in tools}
    assert names == {"get_status", "get_predictions"}
    get_status = next(t for t in tools if t["name"] == "get_status")
    assert get_status["description"]
    assert "inputSchema" in get_status


def test_tools_call_returns_text_content():
    resp = _call({
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "get_status", "arguments": {}},
    })
    result = resp["result"]
    assert result.get("isError", False) is False
    assert result["content"][0]["type"] == "text"
    assert json.loads(result["content"][0]["text"])["reconciles"] == 42


def test_tools_call_unknown_tool_is_error():
    resp = _call({
        "jsonrpc": "2.0", "id": 5, "method": "tools/call",
        "params": {"name": "nope"},
    })
    assert resp["error"]["code"] == -32602


def test_tools_call_handler_exception_is_caught():
    tools = {
        "boom": Tool("boom", "raises", {"type": "object"}, lambda args: (_ for _ in ()).throw(RuntimeError("x"))),
    }
    resp = _call({
        "jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "boom"},
    }, tools)
    assert resp["result"]["isError"] is True
    assert "x" in resp["result"]["content"][0]["text"]


def test_missing_jsonrpc_version_is_invalid_request():
    resp = _call({"id": 8, "method": "ping"})
    assert resp["error"]["code"] == -32600


def test_non_object_body_is_invalid_request():
    # [] is valid JSON but not a JSON-RPC object -> Invalid Request (-32600);
    # genuine parse errors (-32700) live at the transport layer (invalid raw
    # JSON text), so the core maps decoded non-objects to -32600.
    resp = _call([])
    assert resp["error"]["code"] == -32600

    scalar = _call("ping")
    assert scalar["error"]["code"] == -32600


def test_missing_method_is_invalid_request():
    resp = _call({"jsonrpc": "2.0", "id": 10, "params": {}})
    assert resp["error"]["code"] == -32600


def test_unknown_tool_is_invalid_params():
    resp = _call({
        "jsonrpc": "2.0", "id": 11, "method": "tools/call",
        "params": {"name": "missing"},
    })
    assert resp["error"]["code"] == -32602


def test_unknown_method_code():
    resp = _call({"jsonrpc": "2.0", "id": 12, "method": "bogus/method"})
    assert resp["error"]["code"] == -32601


def test_resources_and_prompts_are_empty_capabilities():
    for method, key in (("resources/list", "resources"), ("prompts/list", "prompts")):
        resp = _call({"jsonrpc": "2.0", "id": 9, "method": method})
        assert resp["result"][key] == []


# ── HTTP route (streamable-HTTP transport) ──────────────────────────────────


@pytest.fixture(scope="module")
def mcp_server():
    from predictive_agent.server import _context, start_server

    srv = start_server(18182, 18183)
    # Minimal injected state so tools return real-shaped payloads.
    _context.start_time = 0.0
    yield srv
    srv.shutdown()


def _post_mcp(body):
    req = urllib.request.Request(
        "http://localhost:18182/mcp",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode())


def test_mcp_http_initialize(mcp_server):
    status, resp = _post_mcp({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18",
                   "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
    })
    assert status == 200
    assert resp["result"]["serverInfo"]["name"] == "predictive-agent"


def test_mcp_http_tools_list_and_call(mcp_server):
    _, resp = _post_mcp({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {t["name"] for t in resp["result"]["tools"]}
    assert {"get_predictions", "get_status", "get_at_risk", "get_state",
            "get_backtest", "get_ceph_health", "get_alerts", "get_history"} <= names

    _, resp = _post_mcp({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "get_ceph_health", "arguments": {}},
    })
    result = resp["result"]
    assert result["isError"] is False
    # Either the deployed feed or the safe default — never raises.
    payload = json.loads(result["content"][0]["text"])
    # Safe default when the AlertWatch feed is not wired; never raises.
    assert "enabled" in payload


def test_mcp_http_notification_returns_202(mcp_server):
    req = urllib.request.Request(
        "http://localhost:18182/mcp",
        data=json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 202
    except urllib.error.HTTPError as e:
        assert e.code == 202
