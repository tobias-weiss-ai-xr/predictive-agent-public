"""Performance optimization module for predictive-agent.

This module provides:
- Unified interface for performance optimizations
- Automatic caching configuration
- Batch processing setup
- Performance monitoring and stats collection
- Easy integration with existing components

Usage:
    from predictive_agent.optimize import PerformanceOptimizer
    
    # Initialize optimizer
    optimizer = PerformanceOptimizer()
    
    # Configure optimizations
    optimizer.configure(
        llm_batching=True,
        kubectl_caching=True,
        l1_cache=True,
        parallel_llm=3,  # max concurrent LLM calls
    )
    
    # Get optimized collector
    collector = optimizer.get_optimized_collector()
    
    # Get optimized LLM processor
    llm_processor = optimizer.get_llm_processor(analyzer)
    
    # Get performance stats
    stats = optimizer.get_stats()
"""

import asyncio
import logging
import threading
import time
import types
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# Import local modules
from predictive_agent import collector as collector_module
from predictive_agent.l1_cache import L1Cache, get_cache, reset_cache
from predictive_agent.llm import LLMAnalyzer
from predictive_agent.llm_batch import LLMBatchProcessor, ParallelLLMProcessor

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for performance optimizations."""
    # Cache settings
    kubectl_cache_enabled: bool = True
    kubectl_cache_ttl: float = 30.0  # seconds
    l1_cache_enabled: bool = True
    l1_cache_max_size: int = 10000

    # LLM optimization settings
    llm_batching_enabled: bool = True
    llm_batch_max_size: int = 5
    llm_batch_wait_time: float = 2.0  # seconds
    llm_parallel_enabled: bool = True
    llm_max_concurrent: int = 3
    llm_rate_limit: float = 10.0  # requests per second
    llm_cache_enabled: bool = True
    llm_deduplication_enabled: bool = True

    # Token cost settings (for cost estimation)
    token_cost_input: float = 0.00001  # $ per 1k input tokens
    token_cost_output: float = 0.00002  # $ per 1k output tokens

    # Monitoring
    monitor_enabled: bool = True
    monitor_interval: float = 60.0  # seconds


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    # Cache stats
    kubectl_cache: Dict[str, Any] = field(default_factory=dict)
    l1_cache: Dict[str, Any] = field(default_factory=dict)

    # LLM stats
    llm_total_requests: int = 0
    llm_cached: int = 0
    llm_deduplicated: int = 0
    llm_tokens_used: int = 0
    llm_estimated_cost: float = 0.0
    llm_average_latency: float = 0.0

    # Collector stats
    collector_calls: int = 0
    collector_cached: int = 0

    # Timing
    start_time: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        uptime = time.time() - self.start_time
        return {
            'uptime': uptime,
            'cache': {
                'kubectl': self.kubectl_cache,
                'l1': self.l1_cache,
            },
            'llm': {
                'total_requests': self.llm_total_requests,
                'cached': self.llm_cached,
                'deduplicated': self.llm_deduplicated,
                'tokens_used': self.llm_tokens_used,
                'estimated_cost': self.llm_estimated_cost,
                'average_latency': self.llm_average_latency,
            },
            'collector': {
                'calls': self.collector_calls,
                'cached': self.collector_cached,
            },
        }


class PerformanceOptimizer:
    """Main optimization interface for predictive-agent.
    
    Provides a unified way to configure and access all performance optimizations:
    - L1 caching
    - Kubectl command caching
    - LLM batch processing
    - LLM parallel processing
    - Performance monitoring
    """

    _instance: Optional['PerformanceOptimizer'] = None
    _lock: Optional[threading.Lock] = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern for global optimization state."""
        if PerformanceOptimizer._lock is None:
            import threading
            PerformanceOptimizer._lock = threading.Lock()

        with PerformanceOptimizer._lock:
            if PerformanceOptimizer._instance is None:
                PerformanceOptimizer._instance = super().__new__(cls)
            return PerformanceOptimizer._instance

    _initialized: bool = False

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize the optimizer with optional configuration.
        
        Args:
            config: Optional OptimizationConfig, defaults to standard settings
        """
        if hasattr(self, '_initialized') and self._initialized:
            return  # Already initialized

        self._config = config or OptimizationConfig()
        self._stats = PerformanceStats()
        self._initialized = True

        # Initialize cache
        if self._config.l1_cache_enabled:
            # Ensure cache is initialized
            get_cache()

        # Initialize monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info("PerformanceOptimizer initialized with config: %s", self._config)

    @property
    def config(self) -> OptimizationConfig:
        """Get the current configuration."""
        return self._config

    def configure(self, **kwargs) -> None:
        """Update configuration with new settings.
        
        Args:
            **kwargs: Configuration options to update
        """
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                logger.warning(f"Unknown configuration option: {key}")

        logger.info("PerformanceOptimizer reconfigured: %s", kwargs)

    # --- Cache Management ---

    def get_l1_cache(self) -> L1Cache:
        """Get the L1 cache instance."""
        return get_cache()

    def clear_caches(self) -> None:
        """Clear all caches."""
        collector_module.clear_kubectl_cache()
        reset_cache()
        logger.info("All caches cleared")

    # --- Collector Optimization ---

    def get_optimized_collector(self) -> types.ModuleType:
        """Get the optimized collector module.
        
        The collector is already optimized with caching in the module itself,
        so we just return the module with cache enabled.
        """
        # Configure kubectl cache
        if self._config.kubectl_cache_enabled:
            # Cache is enabled by default in collector
            pass

        return collector_module

    def collect_with_cache(
        self,
        cmd: Any,
        timeout: int = 30,
        use_cache: bool = True,
        cache_ttl: Optional[float] = None
    ) -> Tuple[Any, bool]:
        """Run a kubectl command with optional caching.
        
        Args:
            cmd: Command to run (string or list)
            timeout: Command timeout
            use_cache: Whether to use caching
            cache_ttl: Optional cache TTL override
            
        Returns:
            Tuple of (result, cached)
        """
        ttl = cache_ttl if cache_ttl is not None else self._config.kubectl_cache_ttl

        rc, stdout, stderr = collector_module.run_cmd(
            cmd,
            timeout=timeout,
            use_cache=use_cache and self._config.kubectl_cache_enabled,
            cache_ttl=ttl
        )

        cached = use_cache and self._config.kubectl_cache_enabled
        if cached:
            # Check if it was a cache hit
            # We can't easily determine this without modifying run_cmd,
            # so we'll track it differently
            pass

        self._stats.collector_calls += 1

        return (rc, stdout, stderr), cached

    # --- LLM Optimization ---

    def get_batch_processor(self, analyzer: LLMAnalyzer) -> LLMBatchProcessor:
        """Get a batch processor for LLM analysis.
        
        Args:
            analyzer: LLMAnalyzer instance to use
            
        Returns:
            Configured LLMBatchProcessor
        """
        return LLMBatchProcessor(
            analyzer=analyzer,
            max_batch_size=self._config.llm_batch_max_size,
            max_wait_time=self._config.llm_batch_wait_time,
            cache_enabled=self._config.llm_cache_enabled and self._config.l1_cache_enabled,
            deduplication_enabled=self._config.llm_deduplication_enabled,
            token_cost_per_1k={
                'input': self._config.token_cost_input,
                'output': self._config.token_cost_output,
            }
        )

    def get_parallel_processor(self, analyzer: LLMAnalyzer) -> ParallelLLMProcessor:
        """Get a parallel processor for LLM analysis.
        
        Args:
            analyzer: LLMAnalyzer instance to use
            
        Returns:
            Configured ParallelLLMProcessor
        """
        return ParallelLLMProcessor(
            analyzer=analyzer,
            max_concurrent=self._config.llm_max_concurrent,
            rate_limit=self._config.llm_rate_limit,
            cache_enabled=self._config.llm_cache_enabled and self._config.l1_cache_enabled,
        )

    def get_llm_processor(self, analyzer: LLMAnalyzer) -> Any:
        """Get the appropriate LLM processor based on configuration.
        
        Args:
            analyzer: LLMAnalyzer instance to use
            
        Returns:
            Either LLMBatchProcessor or ParallelLLMProcessor
        """
        if self._config.llm_parallel_enabled:
            return self.get_parallel_processor(analyzer)
        elif self._config.llm_batching_enabled:
            return self.get_batch_processor(analyzer)
        else:
            # Return a simple wrapper
            return _SimpleLLMProcessor(analyzer, self)

    # --- Monitoring ---

    def start_monitoring(self) -> None:
        """Start performance monitoring in the background."""
        if self._running:
            return

        self._running = True

        async def monitor_loop():
            """Monitoring loop."""
            while self._running:
                await asyncio.sleep(self._config.monitor_interval)
                self._update_stats()

        self._monitoring_task = asyncio.create_task(monitor_loop())
        logger.info("Performance monitoring started")

    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if not self._running:
            return

        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            self._monitoring_task = None
        logger.info("Performance monitoring stopped")

    def _update_stats(self) -> None:
        """Update aggregated performance statistics."""
        # Update kubectl cache stats
        self._stats.kubectl_cache = collector_module.get_kubectl_cache_stats()

        # Update L1 cache stats
        if self._config.l1_cache_enabled:
            self._stats.l1_cache = get_cache().get_stats()

    def get_stats(self) -> PerformanceStats:
        """Get current performance statistics."""
        self._update_stats()
        return self._stats

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of performance optimizations and savings.
        
        Returns:
            Dictionary with optimization summary
        """
        stats = self.get_stats()

        # Calculate savings
        kubectl_savings = 0
        if stats.kubectl_cache.get('hits', 0) > 0:
            total_kubectl = stats.kubectl_cache.get('hits', 0) + stats.kubectl_cache.get('misses', 0)
            if total_kubectl > 0:
                hit_rate = stats.kubectl_cache.get('hits', 0) / total_kubectl
                kubectl_savings = hit_rate * 100

        llm_savings = 0
        llm_cache = stats.l1_cache.get('llm_prompt', {})
        if llm_cache.get('hits', 0) > 0:
            total_llm = llm_cache.get('hits', 0) + llm_cache.get('misses', 0)
            if total_llm > 0:
                llm_savings = (llm_cache.get('hits', 0) / total_llm) * 100

        return {
            'optimizations': {
                'kubectl_caching': self._config.kubectl_cache_enabled,
                'l1_cache': self._config.l1_cache_enabled,
                'llm_batching': self._config.llm_batching_enabled,
                'llm_parallel': self._config.llm_parallel_enabled,
            },
            'savings': {
                'kubectl_cache_hit_rate': f"{kubectl_savings:.1f}%",
                'llm_cache_hit_rate': f"{llm_savings:.1f}%",
                'estimated_llm_cost': stats.llm_estimated_cost,
                'estimated_tokens_saved': stats.llm_tokens_used,
            },
            'stats': stats.to_dict(),
        }

    def print_summary(self) -> None:
        """Print a human-readable performance summary."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("PERFORMANCE OPTIMIZATION SUMMARY")
        print("=" * 60)

        print("\n📊 Optimizations Enabled:")
        for opt, enabled in summary['optimizations'].items():
            status = "✓" if enabled else "✗"
            print(f"  {status} {opt}")

        print("\n💰 Savings:")
        print(f"  Kubectl cache hit rate: {summary['savings']['kubectl_cache_hit_rate']}")
        print(f"  LLM cache hit rate: {summary['savings']['llm_cache_hit_rate']}")
        print(f"  Estimated LLM cost: ${summary['savings']['estimated_llm_cost']:.6f}")

        print("\n📈 Statistics:")
        print(f"  Uptime: {summary['stats']['uptime']:.0f}s")

        kubectl_stats = summary['stats']['cache']['kubectl']
        print("\n  Kubectl Cache:")
        print(f"    Hits: {kubectl_stats.get('hits', 0)}")
        print(f"    Misses: {kubectl_stats.get('misses', 0)}")
        print(f"    Hit rate: {kubectl_stats.get('hit_rate', '0%')}")

        llm_stats = summary['stats']['cache']['l1'].get('llm_prompt', {})
        print("\n  LLM Cache:")
        print(f"    Hits: {llm_stats.get('hits', 0)}")
        print(f"    Misses: {llm_stats.get('misses', 0)}")
        print(f"    Size: {summary['stats']['cache']['l1']['sizes'].get('llm_prompt', 0)}")

        print("\n" + "=" * 60)


class _SimpleLLMProcessor:
    """Simple LLM processor without batching or parallelization."""

    def __init__(self, analyzer: LLMAnalyzer, optimizer: PerformanceOptimizer):
        self._analyzer = analyzer
        self._optimizer = optimizer

    async def analyze(
        self,
        prompt: str,
        context: Dict[str, Any],
        pod_key: str,
    ) -> Tuple[Dict[str, Any], bool]:
        """Simple analyze without optimization."""
        # Check cache
        cache = get_cache()
        cache_key = f"llm:{pod_key}"
        hit, cached = cache.get_llm_prompt(cache_key)
        if hit:
            return cached, True

        # Execute
        result = self._analyzer.analyze(
            issue=context.get('issue', 'Unknown'),
            context=context.get('context', ''),
            prediction=context.get('prediction'),
        )

        # Cache result
        cache.set_llm_prompt(cache_key, result)

        return result, False


# Convenience functions

def get_optimizer() -> PerformanceOptimizer:
    """Get the global optimizer instance."""
    return PerformanceOptimizer()


def configure_optimizations(**kwargs) -> PerformanceOptimizer:
    """Configure global optimizations."""
    optimizer = get_optimizer()
    optimizer.configure(**kwargs)
    return optimizer


def get_performance_stats() -> Dict[str, Any]:
    """Get global performance statistics."""
    return get_optimizer().get_summary()


def print_performance_summary() -> None:
    """Print global performance summary."""
    get_optimizer().print_summary()
