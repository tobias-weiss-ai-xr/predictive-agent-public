"""
OPA Integration Module for Predictive-Agent

This module provides seamless integration between the predictive-agent
and Open Policy Agent (OPA) for policy-based decision making in critical
infrastructure environments.

Features:
- Policy-based pod evaluation
- LLM usage governance
- Cache configuration validation
- Security-aware decision making
- Performance monitoring
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from predictive_agent import config
from predictive_agent.opa_cache import OPAPolicyCache, get_opa_policy_cache
from predictive_agent.opa_client import OPAClient, get_opa_client

logger = logging.getLogger(__name__)


class OPAPolicyEngine:
    """
    Main policy engine for predictive-agent
    
    Integrates OPA decision making into predictive-agent workflows.
    OPA integration is ENABLED BY DEFAULT for critical infrastructure mode.
    """

    def __init__(
        self,
        opa_host: str = "localhost",
        opa_port: int = 8181,
        use_tls: bool = True,
        enable_caching: bool = True,
        enabled: bool = True
    ):
        """
        Initialize policy engine
        
        Args:
            opa_host: OPA server hostname
            opa_port: OPA server port
            use_tls: Enable TLS (MUST be True for production)
            enable_caching: Enable policy decision caching
            enabled: Enable OPA integration (DEFAULT: True)
        """
        self._opa_host = opa_host
        self._opa_port = opa_port
        self._use_tls = use_tls
        self._enable_caching = enable_caching
        self._enabled = enabled

        # Initialize clients
        self._opa_client: Optional[OPAClient] = None
        self._opa_cache: Optional[OPAPolicyCache] = None

        # Stats
        # Mixed int counters + datetime/bool flags — annotate as dict[str, Any]
        # so counter arithmetic (+=) type-checks.
        self._stats: Dict[str, Any] = {
            'total_requests': 0,
            'cached_requests': 0,
            'opa_requests': 0,
            'policy_violations': 0,
            'allowed': 0,
            'denied': 0,
            'errors': 0,
            'last_request': None,
            'enabled': self._enabled
        }

        # Initialize if enabled (DEFAULT: enabled)
        if self._enabled:
            self._init_clients()

    def _init_clients(self):
        """Initialize OPA client and cache"""
        self._opa_client = get_opa_client(
            host=self._opa_host,
            port=self._opa_port,
            use_tls=self._use_tls,
            timeout=config.OPA_TIMEOUT,
            max_retries=config.OPA_MAX_RETRIES,
            rate_limit=100,
            rate_window=60
        )

        if self._enable_caching:
            self._opa_cache = get_opa_policy_cache()

    def check_health(self) -> bool:
        """Check OPA server health"""
        if self._opa_client is None:
            self._init_clients()
        if self._opa_client is None:
            return False

        try:
            return self._opa_client.health_check()
        except Exception as e:
            logger.error(f"OPA health check failed: {e}")
            return False

    def check_pod_security(self, pod_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check pod security using OPA policies
        
        If OPA is disabled or unavailable, returns allow=True by default
        (fail-open mode for backward compatibility).
        
        Args:
            pod_data: Pod dictionary (Kubernetes format)
        
        Returns:
            Policy decision including allow/deny and violations
        """
        # If OPA is disabled, allow by default (fail-open)
        if not self._enabled:
            logger.debug("OPA integration disabled, allowing pod by default")
            return {'result': {'allow': True}, 'violation': []}

        self._stats['total_requests'] += 1
        self._stats['last_request'] = datetime.now()

        try:
            if self._enable_caching and self._opa_cache:
                cached, result, _ = self._opa_cache.evaluate_with_cache(
                    'predictive_agent.pod',
                    {'kind': pod_data},
                    'pod',
                    self._opa_client
                )
                if cached:
                    self._stats['cached_requests'] += 1
                    self._process_result(result)
                    return result

            if self._opa_client is None:
                self._init_clients()
            if self._opa_client is None:
                raise RuntimeError("OPA client unavailable")

            result = self._opa_client.check_pod_security(pod_data)
            self._stats['opa_requests'] += 1
            self._process_result(result)
            return result

        except Exception as e:
            self._stats['errors'] += 1
            logger.warning(f"OPA pod security check failed, allowing by default: {e}")
            # Fail-open: allow if OPA is unavailable
            return {
                'result': {'allow': True},
                'violation': [f'OPA check unavailable: {str(e)}'],
                'policy_check_failed': True
            }

    def check_llm_usage(self, llm_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check LLM usage using OPA policies
        
        If OPA is disabled or unavailable, returns allow=True by default
        (fail-open mode for backward compatibility).
        
        Args:
            llm_request: LLM request dictionary
                          Required keys: model, prompt_tokens, max_tokens, cost_per_request
        
        Returns:
            Policy decision including allow/deny and violations
        """
        # If OPA is disabled, allow by default (fail-open)
        if not self._enabled:
            logger.debug("OPA integration disabled, allowing LLM usage by default")
            return {'result': {'allow_llm': True}, 'violation': []}

        self._stats['total_requests'] += 1
        self._stats['last_request'] = datetime.now()

        try:
            if self._enable_caching and self._opa_cache:
                cached, result, _ = self._opa_cache.evaluate_with_cache(
                    'predictive_agent.llm',
                    llm_request,
                    'llm',
                    self._opa_client
                )
                if cached:
                    self._stats['cached_requests'] += 1
                    self._process_result(result)
                    return result

            if self._opa_client is None:
                self._init_clients()
            if self._opa_client is None:
                raise RuntimeError("OPA client unavailable")

            result = self._opa_client.check_llm_usage(llm_request)
            self._stats['opa_requests'] += 1
            self._process_result(result)
            return result

        except Exception as e:
            self._stats['errors'] += 1
            logger.warning(f"OPA LLM usage check failed, allowing by default: {e}")
            # Fail-open: allow if OPA is unavailable
            return {
                'result': {'allow_llm': True},
                'violation': [f'OPA check unavailable: {str(e)}'],
                'policy_check_failed': True
            }

    def check_cache_config(self, cache_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check cache configuration using OPA policies
        
        If OPA is disabled or unavailable, returns allow=True by default
        (fail-open mode for backward compatibility).
        
        Args:
            cache_config: Cache configuration dictionary
                          Required keys: layer, size, ttl
        
        Returns:
            Policy decision including allow/deny and violations
        """
        # If OPA is disabled, allow by default (fail-open)
        if not self._enabled:
            logger.debug("OPA integration disabled, allowing cache config by default")
            return {'result': {'allow_cache_action': True}, 'violation': []}

        self._stats['total_requests'] += 1
        self._stats['last_request'] = datetime.now()

        try:
            if self._enable_caching and self._opa_cache:
                cached, result, _ = self._opa_cache.evaluate_with_cache(
                    'predictive_agent.cache',
                    {'cache_config': cache_config},
                    'cache',
                    self._opa_client
                )
                if cached:
                    self._stats['cached_requests'] += 1
                    self._process_result(result)
                    return result

            if self._opa_client is None:
                self._init_clients()
            if self._opa_client is None:
                raise RuntimeError("OPA client unavailable")

            result = self._opa_client.check_cache_config(cache_config)
            self._stats['opa_requests'] += 1
            self._process_result(result)
            return result

        except Exception as e:
            self._stats['errors'] += 1
            logger.warning(f"OPA cache config check failed, allowing by default: {e}")
            # Fail-open: allow if OPA is unavailable
            return {
                'result': {'allow_cache_action': True},
                'violation': [f'OPA check unavailable: {str(e)}'],
                'policy_check_failed': True
            }

    def _process_result(self, result: Dict[str, Any]) -> None:
        """Process policy result and update stats"""
        if not isinstance(result, dict):
            return

        # Check for violations
        violations = result.get('violation', [])
        if violations:
            self._stats['policy_violations'] += 1

        # Check result
        res = result.get('result', {})
        if isinstance(res, dict):
            if res.get('allow', False):
                self._stats['allowed'] += 1
            elif res.get('allow', True):  # Default to allow if not specified
                pass
            else:
                self._stats['denied'] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get policy engine statistics"""
        stats = self._stats.copy()

        if self._opa_cache:
            stats['cache'] = self._opa_cache.get_all_stats()

        if self._opa_client:
            stats['opa_health'] = self.check_health()
        else:
            stats['opa_health'] = False

        return stats

    def reset_stats(self) -> None:
        """Reset statistics"""
        self._stats = {
            'total_requests': 0,
            'cached_requests': 0,
            'opa_requests': 0,
            'policy_violations': 0,
            'allowed': 0,
            'denied': 0,
            'errors': 0,
            'last_request': None
        }

        if self._opa_cache:
            self._opa_cache.clear_all()


# Global instance
_policy_engine: Optional[OPAPolicyEngine] = None


def get_policy_engine(
    opa_host: str = "localhost",
    opa_port: int = 8181,
    use_tls: bool = True,
    enable_caching: bool = True,
    enabled: bool = True
) -> OPAPolicyEngine:
    """
    Get or create the global policy engine
    
    OPA integration is ENABLED BY DEFAULT for critical infrastructure mode.
    
    Args:
        opa_host: OPA server hostname (default: localhost for sidecar)
        opa_port: OPA server port (default: 8181)
        use_tls: Enable TLS (DEFAULT: True)
        enable_caching: Enable caching (DEFAULT: True)
        enabled: Enable OPA integration (DEFAULT: True)
    
    Returns:
        Policy engine instance (singleton)
    """
    global _policy_engine
    if _policy_engine is None:
        _policy_engine = OPAPolicyEngine(
            opa_host=opa_host,
            opa_port=opa_port,
            use_tls=use_tls,
            enable_caching=enable_caching,
            enabled=enabled
        )
    return _policy_engine


def reset_policy_engine():
    """Reset the global policy engine (for testing)"""
    global _policy_engine
    if _policy_engine is not None:
        _policy_engine.reset_stats()
    _policy_engine = None


# Convenience functions for direct use
def check_pod_security(pod_data: Dict[str, Any]) -> Dict[str, Any]:
    """Check pod security using the global policy engine"""
    engine = get_policy_engine()
    return engine.check_pod_security(pod_data)


def check_llm_usage(llm_request: Dict[str, Any]) -> Dict[str, Any]:
    """Check LLM usage using the global policy engine"""
    engine = get_policy_engine()
    return engine.check_llm_usage(llm_request)


def check_cache_config(cache_config: Dict[str, Any]) -> Dict[str, Any]:
    """Check cache configuration using the global policy engine"""
    engine = get_policy_engine()
    return engine.check_cache_config(cache_config)


def get_policy_stats() -> Dict[str, Any]:
    """Get policy engine statistics"""
    engine = get_policy_engine()
    return engine.get_stats()
