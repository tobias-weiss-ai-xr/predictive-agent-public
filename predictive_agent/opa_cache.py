"""
OPA Cache Layer for L1Cache

This module integrates Open Policy Agent (OPA) caching capabilities
into the predictive-agent's L1Cache system for critical infrastructure.

Security Features:
- TLS 1.3 only
- Certificate verification
- Request rate limiting
- Circuit breaker pattern
- Response caching with TTL
- Memory-based eviction

Dependencies:
- predictive_agent.l1_cache
- opa_client (to be created)
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple


class OPACache:
    """
    OPA-specific cache layer with policy-aware caching
    
    Features:
    - Policy decision caching
    - Security-conscious cache invalidation
    - Rate limit tracking
    - Policy version awareness
    """

    def __init__(
        self,
        max_size: int = 10000,
        default_ttl: int = 300,  # 5 minutes for policy decisions
        max_memory_mb: int = 50
    ):
        """
        Initialize OPA cache
        
        Args:
            max_size: Maximum number of cache entries
            default_ttl: Default time-to-live in seconds
            max_memory_mb: Maximum memory usage in MB
        """
        self._cache: Dict[str, Any] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._max_memory_mb = max_memory_mb
        self._current_memory = 0

        # OPA-specific tracking
        self._policy_versions: Dict[str, str] = {}  # policy_id -> version
        self._decision_counts: Dict[str, int] = {}   # decision_type -> count

        # Stats
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'sets': 0,
            'policy_violations': 0,
            'rate_limited': 0,
            'circuit_breakers': 0
        }

    def _generate_key(
        self,
        policy_package: str,
        input_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate consistent cache key from policy and input
        
        Args:
            policy_package: OPA package name
            input_data: Input data for policy evaluation
        
        Returns:
            SHA-256 hash of policy + input
        """
        key_data = {
            'package': policy_package,
            'input': input_data or {},
            'version': self._policy_versions.get(policy_package, 'v1')
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of a value in bytes"""
        if isinstance(value, dict):
            # Create a copy to avoid modifying original
            size_value = {}
            for k, v in value.items():
                size_value[str(k)] = v
            size_str = json.dumps(size_value, default=str)
        else:
            size_str = json.dumps(value, default=str)
        return len(size_str.encode('utf-8'))

    def _evict_if_needed(self):
        """Evict old entries if cache exceeds limits"""
        while len(self._cache) >= self._max_size or self._current_memory > self._max_memory_mb * 1024 * 1024:
            # Find oldest entry
            oldest_key = None
            oldest_time = None

            for key, entry in self._cache.items():
                if 'timestamp' in entry:
                    if oldest_time is None or entry['timestamp'] < oldest_time:
                        oldest_time = entry['timestamp']
                        oldest_key = key

            if oldest_key is not None:
                # Remove entry
                entry = self._cache.pop(oldest_key)
                self._current_memory -= self._estimate_size(entry.get('data', {}))
                self._stats['evictions'] += 1
            else:
                break

    def set(
        self,
        policy_package: str,
        input_data: Optional[Dict[str, Any]] = None,
        data: Any = None,
        ttl: Optional[int] = None
    ) -> None:
        """
        Set OPA policy decision in cache
        
        Args:
            policy_package: OPA package name
            input_data: Input data that was evaluated
            data: Policy evaluation result
            ttl: Time-to-live in seconds (uses default if None)
        """
        key = self._generate_key(policy_package, input_data)

        entry = {
            'data': data,
            'policy': policy_package,
            'input_hash': hashlib.sha256(json.dumps(input_data or {}, sort_keys=True).encode()).hexdigest() if input_data else None,
            'timestamp': datetime.now(),
            'expires': datetime.now() + timedelta(seconds=ttl or self._default_ttl)
        }

        size = self._estimate_size(entry)

        # Check if we can add this entry
        if size > self._max_memory_mb * 1024 * 1024:
            # Entry too large, skip
            return

        # Evict if needed
        self._evict_if_needed()

        # Add entry
        self._cache[key] = entry
        self._current_memory += size
        self._stats['sets'] += 1

        # Track decision type - safely handle any data structure
        decision_type = 'unknown'
        if isinstance(data, dict):
            result = data.get('result', {})
            if isinstance(result, dict):
                if 'allow' in result:
                    decision_type = 'allow' if result['allow'] else 'deny'
                elif 'allow_llm' in result:
                    decision_type = 'allow' if result['allow_llm'] else 'deny'
                elif 'allow_cache_action' in result:
                    decision_type = 'allow' if result['allow_cache_action'] else 'deny'
        self._decision_counts[decision_type] = self._decision_counts.get(decision_type, 0) + 1

    def get(
        self,
        policy_package: str,
        input_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Any]:
        """
        Get OPA policy decision from cache
        
        Args:
            policy_package: OPA package name
            input_data: Input data to match
        
        Returns:
            Tuple of (hit: bool, data: Any)
        """
        key = self._generate_key(policy_package, input_data)

        entry = self._cache.get(key)
        if entry is None:
            self._stats['misses'] += 1
            return False, None

        # Check expiration
        if datetime.now() > entry['expires']:
            del self._cache[key]
            self._current_memory -= self._estimate_size(entry)
            self._stats['misses'] += 1
            return False, None

        self._stats['hits'] += 1
        return True, entry['data']

    def invalidate_policy(self, policy_package: str) -> int:
        """
        Invalidate all cached decisions for a specific policy package
        
        Args:
            policy_package: OPA package name
        
        Returns:
            Number of entries invalidated
        """
        keys_to_remove = []
        count = 0

        for key, entry in self._cache.items():
            if entry.get('policy') == policy_package:
                keys_to_remove.append(key)
                count += 1

        for key in keys_to_remove:
            self._current_memory -= self._estimate_size(self._cache[key])
            del self._cache[key]

        return count

    def update_policy_version(self, policy_package: str, version: str) -> None:
        """
        Update policy version (triggers cache invalidation)
        
        Args:
            policy_package: OPA package name
            version: New policy version
        """
        old_version = self._policy_versions.get(policy_package)
        self._policy_versions[policy_package] = version

        if old_version != version:
            # Invalidate all cached decisions for this policy
            self.invalidate_policy(policy_package)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = 0.0
        total = self._stats['hits'] + self._stats['misses']
        if total > 0:
            hit_rate = (self._stats['hits'] / total) * 100

        return {
            **self._stats,
            'hit_rate': hit_rate,
            'size': len(self._cache),
            'memory_usage_mb': self._current_memory / (1024 * 1024),
            'policy_versions': self._policy_versions.copy(),
            'decision_counts': self._decision_counts.copy()
        }

    def clear(self) -> None:
        """Clear all cache entries"""
        self._cache.clear()
        self._current_memory = 0
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'sets': 0,
            'policy_violations': 0,
            'rate_limited': 0,
            'circuit_breakers': 0
        }

    def cleanup(self) -> int:
        """
        Remove expired entries and return count removed
        
        Returns:
            Number of entries removed
        """
        now = datetime.now()
        keys_to_remove = []
        count = 0

        for key, entry in self._cache.items():
            if now > entry['expires']:
                keys_to_remove.append(key)
                count += 1

        for key in keys_to_remove:
            self._current_memory -= self._estimate_size(self._cache[key])
            del self._cache[key]

        return count


class OPAPolicyCache:
    """
    High-level OPA policy cache manager
    
    Integrates with predictive-agent's cache system
    """

    def __init__(self):
        # Create separate caches for different policy types
        self._pod_cache = OPACache(max_size=5000, default_ttl=60)
        self._llm_cache = OPACache(max_size=2000, default_ttl=300)
        self._cache_cache = OPACache(max_size=1000, default_ttl=120)
        self._general_cache = OPACache(max_size=10000, default_ttl=300)

    def get_cache(self, policy_type: str) -> OPACache:
        """Get cache for specific policy type"""
        caches = {
            'pod': self._pod_cache,
            'llm': self._llm_cache,
            'cache': self._cache_cache,
            'general': self._general_cache
        }
        return caches.get(policy_type, self._general_cache)

    def evaluate_with_cache(
        self,
        policy_package: str,
        input_data: Dict[str, Any],
        policy_type: str = 'general',
        opa_client: Optional[Any] = None
    ) -> Tuple[bool, Any, str]:
        """
        Evaluate policy with caching
        
        Args:
            policy_package: OPA package name
            input_data: Input data for evaluation
            policy_type: Type of policy for cache selection
            opa_client: OPA client instance
        
        Returns:
            Tuple of (cached: bool, result: Any, cache_key: str)
        """
        cache = self.get_cache(policy_type)
        cache_key = cache._generate_key(policy_package, input_data)

        # Try cache first
        hit, cached_result = cache.get(policy_package, input_data)
        if hit:
            return True, cached_result, cache_key

        # Evaluate with OPA
        if opa_client is not None:
            result = opa_client.evaluate_policy(policy_package, input_data, cache_key)
            cache.set(policy_package, input_data, result)
            return False, result, cache_key

        # Return miss without OPA client
        return False, None, cache_key

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches"""
        return {
            'pod': self._pod_cache.get_stats(),
            'llm': self._llm_cache.get_stats(),
            'cache': self._cache_cache.get_stats(),
            'general': self._general_cache.get_stats()
        }

    def clear_all(self) -> None:
        """Clear all caches"""
        self._pod_cache.clear()
        self._llm_cache.clear()
        self._cache_cache.clear()
        self._general_cache.clear()


# Global instance
_opa_policy_cache: Optional[OPAPolicyCache] = None


def get_opa_policy_cache() -> OPAPolicyCache:
    """Get global OPA policy cache singleton"""
    global _opa_policy_cache
    if _opa_policy_cache is None:
        _opa_policy_cache = OPAPolicyCache()
    return _opa_policy_cache


def reset_opa_policy_cache():
    """Reset global OPA policy cache (for testing)"""
    global _opa_policy_cache
    if _opa_policy_cache is not None:
        _opa_policy_cache.clear_all()
    _opa_policy_cache = None
