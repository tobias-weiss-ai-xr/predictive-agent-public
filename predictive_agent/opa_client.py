"""
OPA Client for predictive-agent — **stdlib-only** (no third-party deps).

Critical Infrastructure Mode. Security features are implemented with the
Python standard library only, in keeping with the project's "stdlib-only
package" rule (see openspec/specs/predictive-agent/spec.md and
pyproject.toml):

- TLS 1.3 only (via ssl.SSLContext)
- Certificate verification + optional mutual TLS (client cert/key)
- Bearer-token authentication
- Request timeouts
- Retry logic with exponential backoff
- Circuit-breaker pattern
- Rate limiting

The public API (OPAClient, get_opa_client, OPASecurityError) is unchanged.
"""

import email.message
import http.client
import json
import logging
import ssl
import time
import urllib.error
from typing import Any, Dict, Optional
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class OPASecurityError(Exception):
    """Security-related OPA errors (circuit open, rate-limited, TLS, etc.)."""
    pass


class OPAClient:
    """Secure OPA client for critical-infrastructure mode (stdlib-only).

    Same constructor signature and behaviour as the previous ``requests``-based
    implementation, but uses ``http.client`` / ``ssl`` from the standard
    library instead of ``requests``/``urllib3``.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8181,
        use_tls: bool = True,
        ca_cert: Optional[str] = None,
        client_cert: Optional[str] = None,
        client_key: Optional[str] = None,
        auth_token: Optional[str] = None,
        timeout: float = 5.0,
        max_retries: int = 3,
        rate_limit: int = 100,
        rate_window: int = 60
    ):
        self.host = host
        self.port = port
        self.use_tls = use_tls
        self.ca_cert = ca_cert
        self.client_cert = client_cert
        self.client_key = client_key
        self.auth_token = auth_token
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit = rate_limit
        self.rate_window = rate_window

        # Build base URL (scheme only for readability/debugging)
        scheme = "https" if use_tls else "http"
        self.base_url = f"{scheme}://{host}:{port}"

        # Rate limiting state
        self._request_tokens = rate_limit
        self._last_rate_reset = time.time()

        # Circuit breaker state
        self._circuit_closed = True
        self._failure_count = 0
        self._circuit_reset_time = 0.0

        # TLS context (lazily built so construction never fails even if the
        # sidecar is absent — fail-open is the rule)
        self._ssl_context: Optional[ssl.SSLContext] = None

    # ────────────────────────────────────────────────────────────────────
    # Internal transport
    # ────────────────────────────────────────────────────────────────────

    def _build_ssl_context(self) -> ssl.SSLContext:
        """Build a hardened TLS context (TLS 1.3, cert verification, mTLS)."""
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3
        if self.ca_cert:
            ctx.load_verify_locations(self.ca_cert)  # CA-pinned verification
        else:
            ctx.check_hostname = True
            ctx.verify_mode = ssl.CERT_REQUIRED if self.use_tls else ssl.CERT_NONE
        if self.client_cert and self.client_key:
            ctx.load_cert_chain(self.client_cert, self.client_key)  # mTLS
        return ctx

    def _get_ssl_context(self) -> ssl.SSLContext:
        if self._ssl_context is None:
            self._ssl_context = self._build_ssl_context()
        return self._ssl_context

    def _connection(self) -> http.client.HTTPConnection:
        if self.use_tls:
            return http.client.HTTPSConnection(
                self.host, self.port,
                context=self._get_ssl_context(),
                timeout=self.timeout,
            )
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def _request(self, method: str, path: str, body: Optional[bytes] = None) -> Dict[str, Any]:
        """Execute an HTTP request to OPA with retry + circuit-breaker + rate limit.

        Returns the parsed JSON response. Raises ``OPASecurityError`` (caught
        upstream → fail-open) on any transport/security failure.
        """
        if not self._check_rate_limit():
            raise OPASecurityError("Rate limit exceeded")
        if not self._check_circuit_breaker():
            raise OPASecurityError("Circuit breaker open")

        url = urljoin(self.base_url, "") + path.lstrip("/")

        # Manual retry loop (replaces urllib3 Retry semantics)
        last_err: Optional[BaseException] = None
        for attempt in range(1, self.max_retries + 1):
            conn = self._connection()
            try:
                conn.request(method, path, body=body, headers=self._headers())
                resp = conn.getresponse()
                data = resp.read()
            except (TimeoutError, OSError, http.client.HTTPException, ssl.SSLError) as e:
                last_err = e
                if isinstance(e, ssl.SSLError):
                    logger.error("TLS error: %s", e)
                elif isinstance(e, (TimeoutError, OSError)):
                    logger.error("Timeout/request error: %s", e)
                else:
                    logger.error("Request failed: %s", e)
                self._record_failure()
                # exponential backoff before retrying
                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 5))
                continue
            finally:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass

            if 500 <= resp.status < 600:
                # Retry on OPA-side transient errors
                hdrs = email.message.Message()
                last_err = urllib.error.HTTPError(url, resp.status, resp.reason, hdrs, None)
                self._record_failure()
                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 5))
                continue

            if resp.status >= 400:
                # Non-retryable error from OPA — surface as a security error
                raise OPASecurityError(
                    f"OPA returned HTTP {resp.status}: {resp.reason}"
                )

            # Success — reset circuit breaker
            if not self._circuit_closed:
                self._circuit_closed = True
                self._failure_count = 0

            try:
                return json.loads(data) if data else {}
            except json.JSONDecodeError as e:
                raise OPASecurityError(f"Malformed OPA response: {e}")

        # All retries exhausted
        raise OPASecurityError(f"Request to OPA failed after {self.max_retries} attempts: {last_err}")

    def _record_failure(self) -> None:
        """Bump failure counters and trip the circuit breaker if needed."""
        self._failure_count += 1
        if self._failure_count >= 5:
            self._circuit_closed = False
            self._circuit_reset_time = time.time() + 30  # open for 30s

    def _check_rate_limit(self) -> bool:
        """Check rate limit before making a request"""
        now = time.time()
        if now - self._last_rate_reset > self.rate_window:
            self._request_tokens = self.rate_limit
            self._last_rate_reset = now
        if self._request_tokens <= 0:
            logger.warning("Rate limit exceeded, wait for next window")
            return False
        self._request_tokens -= 1
        return True

    def _check_circuit_breaker(self) -> bool:
        """Check circuit breaker state"""
        if not self._circuit_closed:
            if time.time() < self._circuit_reset_time:
                logger.warning("Circuit breaker open, requests blocked")
                return False
            else:
                self._circuit_closed = True
                self._failure_count = 0
                logger.info("Circuit breaker reset")
        return True

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    # ────────────────────────────────────────────────────────────────────
    # Public policy API (unchanged)
    # ────────────────────────────────────────────────────────────────────

    def evaluate_policy(
        self,
        package: str,
        input_data: Dict[str, Any],
        cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate an OPA policy package.

        Args:
            package: OPA package name (e.g. ``predictive_agent.pod``)
            input_data: Input data for policy evaluation
            cache_key: Optional cache key (handled by the caller cache)

        Returns:
            Policy evaluation result (parsed JSON body).
        """
        data = json.dumps({"input": input_data}).encode("utf-8")
        return self._request("POST", "/v1/data/" + package, body=data)

    def check_pod_security(self, pod_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check pod security using OPA"""
        return self.evaluate_policy(
            "predictive_agent.pod",
            {"kind": pod_data}
        )

    def check_llm_usage(self, llm_request: Dict[str, Any]) -> Dict[str, Any]:
        """Check LLM usage using OPA"""
        return self.evaluate_policy(
            "predictive_agent.llm",
            llm_request
        )

    def check_cache_config(self, cache_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check cache configuration using OPA"""
        return self.evaluate_policy(
            "predictive_agent.cache",
            {"cache_config": cache_config}
        )

    def health_check(self) -> bool:
        """Check OPA server health"""
        try:
            self._request("GET", "/health")
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("Health check failed: %s", e)
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get OPA server status"""
        try:
            return self._request("GET", "/v1/status")
        except Exception as e:  # noqa: BLE001
            logger.error("Status check failed: %s", e)
            return {"error": str(e)}

    def close(self):
        """Close the session (no persistent connection to close in stdlib mode)."""
        self._ssl_context = None


# Singleton instance
_opa_client: Optional[OPAClient] = None


def get_opa_client(**kwargs) -> OPAClient:
    """Get OPA client singleton"""
    global _opa_client
    if _opa_client is None:
        _opa_client = OPAClient(**kwargs)
    return _opa_client


def reset_opa_client():
    """Reset OPA client singleton (for testing)"""
    global _opa_client
    if _opa_client is not None:
        _opa_client.close()
    _opa_client = None
