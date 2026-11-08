"""Edge-case tests for predictive_agent.server HTTP request handling.

These tests drive a :class:`predictive_agent.server.RequestHandler` directly
(no real sockets or threads): we feed it raw request bytes through the stdlib
request loop and capture the raw HTTP response so we can assert on status codes
and JSON bodies. This mirrors the coverage style of ``tests/test_server.py`` /
``tests/test_endpoints.py`` but exercises the handlers *in-process*.

Coverage:

1. POST endpoints with malformed / empty JSON bodies return 400 (never 500).
2. GET on an unknown path returns 404.
3. Unsupported HTTP methods return an HTTP error, consistently (never 200/500);
   method-mismatches on known routes resolve to 404 just like unknown paths.
4. ``/review`` GET with no remediation manager returns a safe payload (no raise).
5. ``/review`` POST with an unknown review id returns 404.
6. ``/calibration`` with no calibration monitor returns well-formed JSON.
7. Query-string parameters with invalid integers never crash the handlers.

The module-level ``_context`` is stubbed per test with a fresh
:class:`_ServerState` via ``unittest.mock.patch.object`` and restored
automatically afterwards.
"""

import io
import json
from types import SimpleNamespace
from unittest import mock

import pytest

import predictive_agent.server as server
from predictive_agent.server import RequestHandler, _ServerState


class _NoCloseBytesIO(io.BytesIO):
    """BytesIO whose ``close`` is a no-op so the response stays readable."""

    def close(self):
        pass


class _FakeSocket:
    """Minimal stand-in for a connected socket.

    Setting ``wbufsize > 0`` on the handler makes the stdlib build ``wfile``
    through ``connection.makefile('wb', ...)``, so outgoing bytes land in our
    capture buffer instead of ``sendall`` on a real socket.
    """

    def __init__(self, raw_request):
        self.rfile = io.BytesIO(raw_request)
        self.wfile = _NoCloseBytesIO()

    def makefile(self, mode, *args, **kwargs):
        return self.rfile if mode.startswith("r") else self.wfile

    def sendall(self, data):  # defensive; unused once wbufsize > 0
        self.wfile.write(data)

    def shutdown(self, how):
        pass

    def close(self):
        pass


class _TestRequestHandler(RequestHandler):
    wbufsize = 65536


def _dispatch(method, path, body=b"", headers=None):
    """Run one full HTTP request through RequestHandler; return (status, body_str).

    Any uncaught exception in the handler propagates out of the constructor and
    fails the test — which is exactly how we prove "does not crash".
    """
    lines = [f"{method} {path} HTTP/1.0"]
    headers = dict(headers or {})
    if body:
        headers.setdefault("Content-Length", str(len(body)))
    for name, value in headers.items():
        lines.append(f"{name}: {value}")
    raw = ("\r\n".join(lines) + "\r\n\r\n").encode() + body

    sock = _FakeSocket(raw)
    _TestRequestHandler(sock, ("127.0.0.1", 0), mock.Mock())

    response = sock.wfile.getvalue()
    head, _, payload = response.partition(b"\r\n\r\n")
    status_line = head.split(b"\r\n", 1)[0].decode("utf-8", "replace")
    status = int(status_line.split(" ", 2)[1])
    return status, payload.decode("utf-8", "replace")


def _json(result):
    """Return (status, parsed_json) for a ``_dispatch`` result."""
    status, body = result
    return status, json.loads(body)


@pytest.fixture
def ctx():
    """Stub the module-level server context and restore it after the test."""
    fresh = _ServerState()
    with mock.patch.object(server, "_context", fresh):
        yield fresh


# ── 2. Unknown paths ────────────────────────────────────────────────────────
def test_get_unknown_path_returns_404(ctx):
    status, body = _json(_dispatch("GET", "/no/such/endpoint"))
    assert status == 404
    assert isinstance(body, dict)
    assert "error" in body


def test_post_unknown_path_returns_404(ctx):
    status, body = _json(_dispatch("POST", "/no/such/endpoint", body=b"{}"))
    assert status == 404
    assert isinstance(body, dict)
    assert "error" in body


# ── 1. Malformed / empty JSON POST bodies -> 400, never 500 ─────────────────
def test_post_remediate_malformed_json_returns_400(ctx):
    ctx.remediation_manager = mock.Mock()
    status, body = _json(_dispatch(
        "POST", "/remediate",
        body=b'{"pod_name": "ns/pod", "risk_score": 90,',
        headers={"Content-Type": "application/json"},
    ))
    assert status == 400
    assert isinstance(body, dict)
    assert "error" in body


def test_post_remediate_empty_body_returns_400(ctx):
    ctx.remediation_manager = mock.Mock()
    status, body = _json(_dispatch("POST", "/remediate", body=b""))
    assert status == 400
    assert "pod_name" in body.get("error", "")


def test_post_remediate_invalid_json_without_manager_never_500(ctx):
    # Even with NO remediation manager configured the server must not 500 on
    # garbage: it rejects with a structured 503 instead.
    status, body = _json(_dispatch(
        "POST", "/remediate", body=b"{not json",
        headers={"Content-Type": "application/json"},
    ))
    assert status == 503
    assert isinstance(body, dict)
    assert "error" in body


def test_post_remediate_unknown_pod_returns_404(ctx):
    # Unknown identifier in a well-formed POST body -> 404, not 500.
    ctx.remediation_manager = mock.Mock()
    ctx.state_model = SimpleNamespace(pods={})
    status, body = _json(_dispatch(
        "POST", "/remediate",
        body=json.dumps({"pod_name": "ns/ghost", "risk_score": 80.0}).encode(),
        headers={"Content-Type": "application/json"},
    ))
    assert status == 404
    assert "not found" in body.get("error", "").lower()


# ── 4 & 5. /review without a remediation manager / unknown review id ────────
def test_review_get_without_manager_returns_safe_payload(ctx):
    # No remediation manager configured: must answer with a safe, well-formed
    # response (disabled payload, plain 404, or 503) — never a crash/500.
    status, body = _json(_dispatch("GET", "/review"))
    assert status in (200, 404, 503)
    assert isinstance(body, dict)


def test_remediate_get_without_manager_returns_disabled_payload(ctx):
    # The real "feature not configured" path answers with a structured
    # "disabled" payload instead of raising.
    status, body = _json(_dispatch("GET", "/remediate"))
    assert status == 200
    assert body.get("status") == "disabled"
    assert "message" in body


def test_review_post_unknown_review_id_returns_404(ctx):
    # Attach a minimal manager so the handler reaches the unknown-id branch
    # (without a manager the endpoint answers 503 "not initialized").
    ctx.remediation_manager = mock.MagicMock()
    ctx.remediation_manager.approve_review.return_value = None
    status, body = _json(_dispatch(
        "POST", "/review",
        body=json.dumps({"id": "missing-42"}).encode(),
        headers={"Content-Type": "application/json"},
    ))
    assert status == 404
    assert isinstance(body, dict)
    assert "error" in body


# ── 6. /calibration with no calibration monitor ─────────────────────────────
def test_calibration_without_monitor_returns_well_formed_json(ctx):
    # No calibration monitor attached: the server still answers with a
    # well-formed JSON body ({enabled: false, ...}), never a 500.
    status, body = _json(_dispatch("GET", "/calibration"))
    assert status != 500
    assert isinstance(body, dict)
    assert body.get("enabled") is False


# ── 3. Unknown HTTP methods ─────────────────────────────────────────────────
def test_unsupported_method_returns_error_consistently(ctx):
    put_known = _dispatch("PUT", "/status")
    put_unknown = _dispatch("PUT", "/no/such/route")
    delete_known = _dispatch("DELETE", "/status")

    # Unsupported methods must yield an HTTP error — never 200, never a 500
    # crash — and the result must be deterministic across paths/methods
    # (the stdlib default for unregistered methods here is 501).
    assert all(status >= 400 and status != 500
               for status in (put_known[0], put_unknown[0], delete_known[0]))
    assert put_known[0] == put_unknown[0] == delete_known[0]


def test_method_mismatch_on_known_path_matches_unknown_404(ctx):
    # A POST on a GET-only route resolves to 404, consistent with an unknown
    # path (an allowed outcome per the 405-or-404 contract).
    post_healthz = _dispatch("POST", "/healthz")[0]
    get_unknown = _dispatch("GET", "/definitely/not/here")[0]
    assert post_healthz == get_unknown == 404


# ── 7. Query strings with invalid integers ──────────────────────────────────
def test_query_string_invalid_integers_do_not_crash(ctx):
    # Endpoints tolerate query strings with non-integer values: routing strips
    # the query part before dispatch, so handlers never choke on bad ints.
    for path in ("/predictions?limit=abc",
                 "/state?top=not-an-int",
                 "/history?max=garbage",
                 "/status?uptime=NaN",
                 "/remediate?dry=maybe"):
        status, body = _json(_dispatch("GET", path))
        assert status == 200
        assert isinstance(body, (dict, list))
