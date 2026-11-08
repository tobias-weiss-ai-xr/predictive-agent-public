"""Predictive Agent v4.0.8 — Predictive Kubernetes Health Monitor.

Performance Optimization Modules:
- l1_cache: L1 caching layer for pod states, predictions, and LLM prompts
- llm_batch: Batch processing and parallel execution for LLM calls
- optimize: Unified performance optimization interface
"""
__version__ = "4.0.8"

from predictive_agent.l1_cache import L1Cache, get_cache
from predictive_agent.llm_batch import LLMBatchProcessor, ParallelLLMProcessor
from predictive_agent.optimize import PerformanceOptimizer, configure_optimizations, get_optimizer

__all__ = [
    'get_cache', 'L1Cache',
    'LLMBatchProcessor', 'ParallelLLMProcessor',
    'PerformanceOptimizer', 'get_optimizer', 'configure_optimizations',
]
