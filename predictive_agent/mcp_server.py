"""Dependency-free MCP (Model Context Protocol) server core.

Implements just enough of the MCP spec over the Streamable HTTP transport for
an AI assistant (e.g. a coding agent, chat tool, or the in-cluster bridge) to
inspect this operator's state through stdio-less tool calls.

Transport / protocol surface implemented:

  * ``initialize``          — protocolVersion + capabilities + serverInfo
  * ``notifications/initialized`` — consumed, no reply (202)
  * ``ping``                — keepalive
  * ``tools/list``          — read-only tool registry
  * ``tools/call``          — invoke a tool; handler exceptions become
                              ``isError`` results, never a dead call
  * ``resources/list`` and ``prompts/list`` — advertised but empty
  * anything else           — ``-32601 Method not found``

Only JSON-RPC 2.0 requests with ``"jsonrpc": "2.0"`` are accepted; the server
answers ``application/json`` (it never streams, so no SSE is required by the
spec). Every tool exposed here is a *snapshot* of agent state — none of them
mutate the cluster. That invariant is load-bearing for exposing ``/mcp`` on
the operator's API port behind the ingress.

The core is pure and socket-free ("No globals, no IO") so it is trivially
unit-testable; ``server.py`` wires the registry built from its own handlers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

# MCP spec protocol version we speak (streamable HTTP, tools only).
PROTOCOL_VERSION = "2025-06-18"

SERVER_NAME = "predictive-agent"
SERVER_VERSION = "4.1.8"

# JSON-RPC error codes (MCP reuses the JSON-RPC 2.0 error space).
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class McpError(Exception):
    """Protocol-level error carrying a JSON-RPC error code."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class Tool:
    """A read-only MCP tool exposed by the operator.

    ``handler`` receives the parsed JSON arguments dict and returns a
    JSON-serializable snapshot (dict/list/str/scalar). Exceptions raised by a
    handler are converted into ``isError`` tool results upstream in
    ``handle_request`` — a broken tool never kills the JSON-RPC session.
    """

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema or {"type": "object", "properties": {}},
        }


def _result(_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": _id, "result": result}


def _error(_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": _id, "error": {"code": code, "message": message}}


def _call_tool(name: str, arguments: Dict[str, Any], tools: Dict[str, Tool]) -> Any:
    tool = tools.get(name)
    if tool is None:
        raise McpError(INVALID_PARAMS, f"Unknown tool: {name}")
    try:
        data = tool.handler(arguments or {})
    except Exception as exc:  # noqa: BLE001 - tool failure is a result, not a crash
        return {
            "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
            "isError": True,
        }
    return {
        "content": [{"type": "text", "text": _dump(data)}],
        "isError": False,
    }


def _dump(data: Any) -> str:
    """Serialise a tool result to text; fall back to repr on unserialisable data."""
    try:
        return json.dumps(data, default=str, sort_keys=True)
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        return repr(data)


def handle_request(payload: Any, tools: Dict[str, Tool]) -> Optional[Dict[str, Any]]:
    """Handle one raw JSON-RPC message.

    Args:
        payload: The decoded request body (already ``json.loads``-ed by the
            transport; ``None``/scalar/array here is a parse error).
        tools: Name -> Tool registry for ``tools/call``.

    Returns:
        A JSON-RPC response dict to write back, or ``None`` for notifications
        (the transport replies ``202 Accepted`` with an empty body).

    Raises:
        Never. Every failure mode is mapped to a JSON-RPC error result.
    """
    if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0":
        return _error(None, INVALID_REQUEST, "Invalid Request: expected JSON-RPC 2.0")

    method = payload.get("method")
    _id = payload.get("id")  # notifications carry no id
    params = payload.get("params") or {}

    if method == "initialize":
        return _result(_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": (
                "Read-only operator state endpoint. Tools return JSON snapshots of "
                "predictions, risk, state, backtest and health; nothing here mutates "
                "the cluster."
            ),
        })

    if method is None:
        return _error(_id, INVALID_REQUEST, "Invalid Request: missing method")

    if method.startswith("notifications/"):
        # Notifications have no id and expect NO response (transport: 202).
        return None

    if _id is None:
        # A non-notification request without an id cannot be answered.
        return _error(None, INVALID_REQUEST, "Invalid Request: missing id")

    if method == "ping":
        return _result(_id, {})

    if method == "tools/list":
        return _result(_id, {"tools": [t.to_dict() for t in tools.values()],
                             "nextCursor": None})

    if method == "tools/call":
        name = params.get("name") if isinstance(params, dict) else None
        try:
            if not isinstance(name, str):
                raise McpError(INVALID_PARAMS, "tools/call requires a string 'name'")
            return _result(_id, _call_tool(
                name, params.get("arguments") or {}, tools))
        except McpError as exc:
            return _error(_id, exc.code, exc.message)

    if method == "resources/list":
        return _result(_id, {"resources": [], "nextCursor": None})

    if method == "prompts/list":
        return _result(_id, {"prompts": [], "nextCursor": None})

    return _error(_id, METHOD_NOT_FOUND, f"Method not found: {method}")
