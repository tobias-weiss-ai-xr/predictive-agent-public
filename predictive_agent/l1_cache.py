"""L1 Cache module for predictive-agent.

Provides a hierarchical caching system:
- In-memory cache with TTL for fast access
- Request deduplication to avoid duplicate LLM calls
- Pod state caching
- Prediction result caching
- Token usage tracking

Inspired by:
- moe-sovereign's L0/L1 cache architecture
- pi-l1-cache's FNV-1a hashing and memory optimization

Features from pi-l1-cache:
- FNV-1a hash (~100x faster than SHA-256)
- Memory cap (default 50MB)
- CPU-aware auto-disable (>95% CPU)
- Size-based eviction
"""

import asyncio
import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, TypeVar

# CPU checking (inspired by pi-l1-cache)
try:
    import psutil
    CPU_CHECK_AVAILABLE = True
except ImportError:
    CPU_CHECK_AVAILABLE = False


T = TypeVar('T')


@dataclass
class CacheEntry:
    """A single cache entry with TTL, metadata, and size tracking.
    
    Inspired by pi-l1-cache's memory-aware design.
    """
    value: Any
    created_at: float = field(default_factory=time.time)
    ttl: float = 300.0  # 5 minutes default
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0  # Size in bytes (for memory tracking)

    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        return time.time() > (self.created_at + self.ttl)

    def touch(self) -> None:
        """Update last accessed time."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring.
    
    Inspired by pi-l1-cache's comprehensive stats tracking.
    """
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    sets: int = 0
    deduplicated_requests: int = 0
    estimated_tokens_saved: int = 0
    cpu_skips: int = 0  # From pi-l1-cache: cache disabled due to CPU load
    memory_evictions: int = 0  # Evictions due to memory cap

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'sets': self.sets,
            'deduplicated_requests': self.deduplicated_requests,
            'estimated_tokens_saved': self.estimated_tokens_saved,
            'cpu_skips': self.cpu_skips,
            'memory_evictions': self.memory_evictions,
            'hit_rate': f"{self.hit_rate:.2%}",
        }


class LRUCache:
    """Thread-safe LRU cache with TTL and memory tracking.
    
    Inspired by pi-l1-cache's memory-aware design.
    
    Implements a thread-safe LRU (Least Recently Used) cache with:
    - Maximum size limit
    - Maximum memory limit (from pi-l1-cache)
    - Per-entry TTL (time-to-live)
    - Automatic expiration
    - Access tracking
    - Size-based eviction (from pi-l1-cache)
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 300.0,
        max_memory_mb: int = 50,  # 50MB cap from pi-l1-cache
    ):
        """Initialize the cache.
        
        Args:
            max_size: Maximum number of entries to store
            default_ttl: Default time-to-live in seconds for new entries
            max_memory_mb: Maximum memory usage in MB (0 for unlimited)
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self._current_memory: int = 0
        self._stats = CacheStats()

    def get(self, key: str) -> Tuple[bool, Any]:
        """Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            Tuple of (hit: bool, value: Any)
        """
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return False, None

            entry = self._cache[key]

            if entry.is_expired():
                # Remove expired entry
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                return False, None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._stats.hits += 1
            return True, entry.value

    @staticmethod
    def _fast_hash(text: str) -> str:
        """FNV-1a hash - ~100x faster than SHA-256.
        
        Inspired by pi-l1-cache's fast hashing algorithm.
        """
        hash_val = 2166136261
        for char in text.encode():
            hash_val ^= char
            hash_val = (hash_val * 16777619) & 0xFFFFFFFFFFFFFFFF
        return hex(hash_val)

    @staticmethod
    def _estimate_size(obj: Any) -> int:
        """Estimate object size in bytes.
        
        Inspired by pi-l1-cache's size estimation.
        """
        try:
            # Rough estimate: JSON string length * 2 (UTF-16 overhead)
            return len(json.dumps(obj)) * 2
        except (TypeError, ValueError):
            return 1024  # Fallback size

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Optional TTL override in seconds
        """
        with self._lock:
            # Evict expired entries and enforce max size/memory
            self._evict_if_needed()

            # Calculate size
            size_bytes = LRUCache._estimate_size(value)

            # Check memory limit before adding
            if self._max_memory > 0:
                # Estimate new memory usage
                new_memory = self._current_memory + size_bytes
                if new_memory > self._max_memory:
                    # Need to evict to make room
                    self._evict_by_memory(size_bytes)

            effective_ttl = ttl if ttl is not None else self._default_ttl
            self._cache[key] = CacheEntry(
                value=value,
                ttl=effective_ttl,
                size_bytes=size_bytes
            )
            self._current_memory += size_bytes
            self._stats.sets += 1

    def delete(self, key: str) -> bool:
        """Delete a key from the cache.
        
        Args:
            key: The key to delete
            
        Returns:
            True if the key was found and deleted, False otherwise
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._current_memory -= entry.size_bytes
                del self._cache[key]
                return True
            return False

    def invalidate(self, prefix: str = "") -> int:
        """Invalidate all entries matching a prefix.
        
        Args:
            prefix: Key prefix to match (empty string matches all)
            
        Returns:
            Number of entries invalidated
        """
        with self._lock:
            keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
            for key in keys_to_remove:
                entry = self._cache[key]
                self._current_memory -= entry.size_bytes
                del self._cache[key]
            return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0

    def size(self) -> int:
        """Get the current number of entries."""
        with self._lock:
            return len(self._cache)

    def _evict_if_needed(self) -> None:
        """Evict expired entries and enforce max size. Must be called with lock held."""
        # Remove expired entries
        expired_keys = [
            k for k, v in self._cache.items()
            if v.is_expired()
        ]
        for key in expired_keys:
            entry = self._cache[key]
            self._current_memory -= entry.size_bytes
            del self._cache[key]
            self._stats.evictions += 1

        # Enforce max size
        while len(self._cache) >= self._max_size:
            # Pop the first item (least recently used)
            key, entry = self._cache.popitem(last=False)
            self._current_memory -= entry.size_bytes
            self._stats.evictions += 1

    def _evict_by_memory(self, needed_space: int) -> None:
        """Evict oldest entries until we have enough memory.
        
        Inspired by pi-l1-cache's size-based eviction.
        Must be called with lock held.
        """
        # Performance optimization: evict a batch at once
        entries_to_remove = []
        while self._current_memory + needed_space > self._max_memory and self._cache:
            # Get oldest entry (first in OrderedDict)
            key, entry = next(iter(self._cache.items()))
            entries_to_remove.append((key, entry))
            if len(entries_to_remove) >= max(1, self._max_size // 10):  # Batch eviction
                break

        for key, entry in entries_to_remove:
            self._cache.pop(key, None)
            self._current_memory -= entry.size_bytes
            self._stats.evictions += 1
            self._stats.memory_evictions += 1

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                sets=self._stats.sets,
                cpu_skips=self._stats.cpu_skips,
                memory_evictions=self._stats.memory_evictions,
            )

    def memory_usage(self) -> Tuple[int, int]:
        """Get current memory usage.
        
        Returns:
            Tuple of (current_bytes, max_bytes)
        """
        with self._lock:
            return self._current_memory, self._max_memory


class RequestDeduplicator:
    """Deduplicate concurrent identical requests.
    
    Prevents multiple identical requests from being processed simultaneously,
    ensuring each unique request is only executed once. Subsequent identical
    requests wait for the first one to complete.
    
    This is especially useful for LLM calls where multiple pods might have
    similar characteristics and would generate identical or similar prompts.
    """

    def __init__(self):
        self._in_progress: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self._stats = CacheStats()

    async def deduplicate(
        self,
        key: str,
        async_func: Callable[[], Any]
    ) -> Any:
        """Execute a function with deduplication.
        
        If another call with the same key is in progress, wait for it.
        If not, execute the function and store the result for others.
        
        Args:
            key: Unique key identifying the request
            async_func: Async function to execute
            
        Returns:
            The result of the async function
        """
        async with self._lock:
            if key in self._in_progress:
                # Another request with the same key is in progress
                self._stats.deduplicated_requests += 1
                future = self._in_progress[key]
                return await future

            # Create a new future for this request
            future = asyncio.ensure_future(async_func())
            self._in_progress[key] = future

        try:
            result = await future
            return result
        finally:
            # Clean up regardless of success or failure
            async with self._lock:
                if key in self._in_progress:
                    del self._in_progress[key]

    @property
    def stats(self) -> CacheStats:
        """Get deduplication statistics."""
        return CacheStats(
            deduplicated_requests=self._stats.deduplicated_requests,
        )


class L1Cache:
    """Main L1 cache for predictive-agent.
    
    Provides multiple cache layers:
    - General purpose cache (LRU with TTL)
    - Pod state cache
    - Prediction cache
    - LLM prompt cache
    - Request deduplication
    
    This is the primary caching interface for the predictive-agent system.
    """

    # Cache key prefixes for namespacing
    PREFIX_POD_STATE = "pod_state:"
    PREFIX_PREDICTION = "prediction:"
    PREFIX_LLM_PROMPT = "llm_prompt:"
    PREFIX_KUBECTL = "kubectl:"
    PREFIX_METRICS = "metrics:"

    def __init__(
        self,
        max_size: int = 10000,
        default_ttl: float = 300.0,
        pod_state_ttl: float = 60.0,
        prediction_ttl: float = 120.0,
        kubectl_ttl: float = 30.0,
        max_memory_mb: int = 50,  # From pi-l1-cache
        cpu_threshold: float = 95.0,  # From pi-l1-cache
    ):
        """Initialize L1 cache with multiple layers.
        
        Args:
            max_size: Maximum entries per cache layer
            default_ttl: Default TTL in seconds
            pod_state_ttl: TTL for pod state cache
            prediction_ttl: TTL for prediction cache
            kubectl_ttl: TTL for kubectl result cache
            max_memory_mb: Maximum memory usage per layer in MB (0 for unlimited)
            cpu_threshold: CPU percentage threshold to disable caching (0 for disabled)
        """
        self._max_memory_mb = max_memory_mb
        self._cpu_threshold = cpu_threshold
        self._cpu_overloaded = False
        self._last_cpu_check = 0.0
        self._cpu_check_interval = 5.0  # Check CPU every 5 seconds

        self._general_cache = LRUCache(max_size, default_ttl, max_memory_mb)
        self._pod_state_cache = LRUCache(max_size, pod_state_ttl, max_memory_mb)
        self._prediction_cache = LRUCache(max_size, prediction_ttl, max_memory_mb)
        self._kubectl_cache = LRUCache(max_size, kubectl_ttl, max_memory_mb)
        self._llm_prompt_cache = LRUCache(max_size * 2, default_ttl * 2, max_memory_mb)
        self._deduplicator = RequestDeduplicator()

    @staticmethod
    def make_key(prefix: str, *parts: Any, **kwargs: Any) -> str:
        """Create a consistent cache key from parts and kwargs.
        
        Args:
            prefix: Key prefix for namespacing
            *parts: Positional parts to include in key
            **kwargs: Keyword arguments to include in key
            
        Returns:
            A consistent string key
        """
        # Convert all parts to strings
        string_parts = [prefix] + [str(p) for p in parts]

        # Sort kwargs for consistent ordering
        sorted_kwargs = sorted(kwargs.items(), key=lambda x: x[0])
        string_parts.extend([f"{k}={v}" for k, v in sorted_kwargs])

        # Create hash for consistent length
        key_string = "|".join(string_parts)
        # Use FNV-1a for speed (from pi-l1-cache), fall back to SHA-256
        try:
            return LRUCache._fast_hash(key_string)
        except Exception:
            return hashlib.sha256(key_string.encode()).hexdigest()

    def _is_cpu_overloaded(self) -> bool:
        """Check if CPU is overloaded (from pi-l1-cache).
        
        Returns:
            True if CPU usage exceeds threshold
        """
        if self._cpu_threshold <= 0 or not CPU_CHECK_AVAILABLE:
            return False

        now = time.time()
        # Only check periodically to avoid overhead
        if now - self._last_cpu_check < self._cpu_check_interval:
            return self._cpu_overloaded

        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self._cpu_overloaded = cpu_percent > self._cpu_threshold
            self._last_cpu_check = now
            if self._cpu_overloaded:
                import logging
                logger = logging.getLogger('L1Cache')
                logger.warning(f"CPU overloaded ({cpu_percent:.1f}% > {self._cpu_threshold}%), cache operations may be skipped")
            return self._cpu_overloaded
        except Exception:
            return False

    def _check_cpu_and_get(self, cache: LRUCache, key: str) -> Tuple[bool, Any]:
        """Check CPU and conditionally get from cache.
        
        Inspired by pi-l1-cache's CPU-aware caching.
        """
        if self._is_cpu_overloaded():
            cache._stats.cpu_skips += 1
            return False, None
        return cache.get(key)

    # --- General Cache ---

    def get_general(self, key: str) -> Tuple[bool, Any]:
        """Get from general cache."""
        return self._check_cpu_and_get(self._general_cache, key)

    def set_general(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set in general cache."""
        self._general_cache.set(key, value, ttl)

    # --- Pod State Cache ---

    def get_pod_state(self, ns: str, name: str) -> Tuple[bool, Any]:
        """Get pod state from cache."""
        if self._is_cpu_overloaded():
            self._pod_state_cache._stats.cpu_skips += 1
            return False, None
        key = L1Cache.make_key(L1Cache.PREFIX_POD_STATE, ns, name)
        return self._pod_state_cache.get(key)

    def set_pod_state(self, ns: str, name: str, state: Dict[str, Any]) -> None:
        """Set pod state in cache."""
        if self._is_cpu_overloaded():
            return
        key = L1Cache.make_key(L1Cache.PREFIX_POD_STATE, ns, name)
        self._pod_state_cache.set(key, state)

    def invalidate_pod(self, ns: str, name: str) -> None:
        """Invalidate a specific pod from all caches."""
        # Invalidate pod state
        key = L1Cache.make_key(L1Cache.PREFIX_POD_STATE, ns, name)
        self._pod_state_cache.delete(key)

        # Invalidate predictions for this pod
        pred_key = L1Cache.make_key(L1Cache.PREFIX_PREDICTION, ns, name)
        self._prediction_cache.delete(pred_key)

        # Invalidate kubectl results for this pod
        kubectl_key = L1Cache.make_key(L1Cache.PREFIX_KUBECTL, ns, name)
        self._kubectl_cache.delete(kubectl_key)

    # --- Prediction Cache ---

    def get_prediction(self, ns: str, name: str) -> Tuple[bool, Any]:
        """Get prediction from cache."""
        if self._is_cpu_overloaded():
            self._prediction_cache._stats.cpu_skips += 1
            return False, None
        key = L1Cache.make_key(L1Cache.PREFIX_PREDICTION, ns, name)
        return self._prediction_cache.get(key)

    def set_prediction(self, ns: str, name: str, prediction: Any) -> None:
        """Set prediction in cache."""
        if self._is_cpu_overloaded():
            return
        key = L1Cache.make_key(L1Cache.PREFIX_PREDICTION, ns, name)
        self._prediction_cache.set(key, prediction)

    # --- Kubectl Cache ---

    def get_kubectl_result(self, cmd_hash: str) -> Tuple[bool, Any]:
        """Get kubectl command result from cache."""
        if self._is_cpu_overloaded():
            self._kubectl_cache._stats.cpu_skips += 1
            return False, None
        return self._kubectl_cache.get(cmd_hash)

    def set_kubectl_result(self, cmd: str, result: Any) -> None:
        """Set kubectl command result in cache."""
        if self._is_cpu_overloaded():
            return
        cmd_hash = hashlib.sha256(cmd.encode()).hexdigest()
        self._kubectl_cache.set(cmd_hash, result)

    # --- LLM Prompt Cache ---

    def get_llm_prompt(self, prompt: str) -> Tuple[bool, Any]:
        """Get LLM prompt response from cache.

        ``prompt`` may be the raw prompt text or any stable cache key; it is
        hashed exactly like :meth:`set_llm_prompt` so the pair is symmetric
        (previously only ``set`` hashed, so writes were invisible to reads).
        """
        if self._is_cpu_overloaded():
            self._llm_prompt_cache._stats.cpu_skips += 1
            return False, None
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        return self._llm_prompt_cache.get(prompt_hash)

    def set_llm_prompt(self, prompt: str, response: Any, estimated_tokens: int = 0) -> None:
        """Set LLM prompt response in cache.
        
        Args:
            prompt: The input prompt
            response: The LLM response
            estimated_tokens: Estimated tokens used (for stats)
        """
        if self._is_cpu_overloaded():
            return
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        self._llm_prompt_cache.set(prompt_hash, response)
        if estimated_tokens > 0:
            # This is a simplification - actual token savings depend on cache hits
            pass

    # --- Deduplication ---

    async def deduplicate_llm_call(
        self,
        prompt: str,
        async_func: Callable[[], Any],
        estimated_tokens: int = 0
    ) -> Any:
        """Deduplicate LLM calls with the same prompt.
        
        Args:
            prompt: The LLM prompt (used for deduplication key)
            async_func: Async function that makes the LLM call
            estimated_tokens: Estimated tokens for this call
            
        Returns:
            The LLM response
        """
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        return await self._deduplicator.deduplicate(
            f"llm:{prompt_hash[:16]}",  # Use first 16 chars for shorter key
            async_func
        )

    # --- Statistics ---

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            'general': self._general_cache.stats.to_dict(),
            'pod_state': self._pod_state_cache.stats.to_dict(),
            'prediction': self._prediction_cache.stats.to_dict(),
            'kubectl': self._kubectl_cache.stats.to_dict(),
            'llm_prompt': self._llm_prompt_cache.stats.to_dict(),
            'deduplication': self._deduplicator.stats.to_dict(),
            'sizes': {
                'general': self._general_cache.size(),
                'pod_state': self._pod_state_cache.size(),
                'prediction': self._prediction_cache.size(),
                'kubectl': self._kubectl_cache.size(),
                'llm_prompt': self._llm_prompt_cache.size(),
            }
        }

    def clear_all(self) -> None:
        """Clear all caches."""
        self._general_cache.clear()
        self._pod_state_cache.clear()
        self._prediction_cache.clear()
        self._kubectl_cache.clear()
        self._llm_prompt_cache.clear()

    def warm_cache_with_pod_list(self, pods: List[str]) -> None:
        """Pre-warm the cache with a list of pod namespaces/names.
        
        This can be used during startup to pre-cache known pods.
        
        Args:
            pods: List of pod identifiers in format "ns/name"
        """
        for pod_id in pods:
            if '/' in pod_id:
                ns, name = pod_id.split('/', 1)
                # Reserve space for pod state (will be filled by collector)
                key = L1Cache.make_key(L1Cache.PREFIX_POD_STATE, ns, name)
                self._pod_state_cache.set(key, None, ttl=10)  # Short TTL placeholder


# Global cache instance
_cache: Optional[L1Cache] = None
_cache_lock = threading.Lock()


def get_cache() -> L1Cache:
    """Get or create the global L1 cache instance."""
    global _cache
    with _cache_lock:
        if _cache is None:
            _cache = L1Cache()
        return _cache


def reset_cache() -> None:
    """Reset the global cache instance (useful for testing)."""
    global _cache
    with _cache_lock:
        if _cache is not None:
            _cache.clear_all()
            _cache = None
