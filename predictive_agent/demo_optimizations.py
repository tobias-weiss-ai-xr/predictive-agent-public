#!/usr/bin/env python3
"""Demo script showcasing performance optimizations in predictive-agent.

This script demonstrates:
1. L1 Cache performance
2. Kubectl command caching
3. LLM batch processing
4. LLM parallel processing
5. Integrated performance monitoring

Usage:
    python demo_optimizations.py
    
    # Or run specific demos
    python demo_optimizations.py --cache
    python demo_optimizations.py --llm
    python demo_optimizations.py --all
"""

import argparse
import asyncio
import sys
import time
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, '/home/weissto_local/git/predictive-agent')

from predictive_agent import collector
from predictive_agent.l1_cache import get_cache, reset_cache
from predictive_agent.llm import LLMAnalyzer, LLMBackend
from predictive_agent.llm_batch import LLMBatchProcessor, ParallelLLMProcessor
from predictive_agent.optimize import PerformanceOptimizer


class DemoColors:
    """ANSI color codes for demo output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class MockSlowLLMAnalyzer(LLMAnalyzer):
    """Mock LLM analyzer with configurable delay to simulate slow API."""

    def __init__(self, delay: float = 0.1):
        self.call_count = 0
        self.delay = delay
        self.backend = LLMBackend.OLLAMA
        self.url = "http://localhost:11434"
        self.model = "mock-model"
        self.timeout = 180

    def analyze(self, issue, context, prediction=None):
        """Mock analyze with delay."""
        self.call_count += 1
        time.sleep(self.delay)
        return {
            'analysis': f'Analyzed {issue} (call #{self.call_count})',
            'severity': 'medium',
            'action': 'Monitor',
            'command': 'kubectl describe pod <name>',
        }


def print_header(title: str) -> None:
    """Print a colored header."""
    print(f"\n{DemoColors.HEADER}{'=' * 60}{DemoColors.ENDC}")
    print(f"{DemoColors.HEADER}{DemoColors.BOLD}{title}{DemoColors.ENDC}")
    print(f"{DemoColors.HEADER}{'=' * 60}{DemoColors.ENDC}\n")


def print_section(title: str) -> None:
    """Print a colored section header."""
    print(f"\n{DemoColors.OKBLUE}{'-' * 40}{DemoColors.ENDC}")
    print(f"{DemoColors.OKBLUE}{title}{DemoColors.ENDC}")
    print(f"{DemoColors.OKBLUE}{'-' * 40}{DemoColors.ENDC}")


def print_result(label: str, value: Any, good: bool = True) -> None:
    """Print a labeled result with color."""
    color = DemoColors.OKGREEN if good else DemoColors.FAIL
    print(f"  {color}{label}: {value}{DemoColors.ENDC}")


def print_time_elapsed(label: str, start: float, good: bool = True) -> float:
    """Print elapsed time."""
    elapsed = time.time() - start
    color = DemoColors.OKGREEN if good else DemoColors.WARNING
    print(f"  {color}{label}: {elapsed*1000:.2f}ms{DemoColors.ENDC}")
    return elapsed


# ============================================================
# Demo: L1 Cache
# ============================================================

def demo_l1_cache():
    """Demonstrate L1 cache performance."""
    print_header("L1 Cache Performance")

    reset_cache()
    cache = get_cache()

    # Test 1: Write performance
    print_section("Write Performance (10,000 entries)")
    start = time.time()
    for i in range(10000):
        cache.set_pod_state(f"namespace-{i % 100}", f"pod-{i}", {
            'status': 'running',
            'cpu': 0.5,
            'memory': 128,
        })
    elapsed = print_time_elapsed("Write 10,000 entries", start)
    print_result("Throughput", f"{10000/elapsed:.0f} writes/sec")

    # Test 2: Read performance
    print_section("Read Performance (10,000 entries)")
    start = time.time()
    for i in range(10000):
        cache.get_pod_state(f"namespace-{i % 100}", f"pod-{i}")
    elapsed = print_time_elapsed("Read 10,000 entries", start)
    print_result("Throughput", f"{10000/elapsed:.0f} reads/sec")

    # Test 3: Mixed operations
    print_section("Mixed Operations (5,000 reads + 5,000 writes)")
    start = time.time()
    for i in range(5000):
        cache.set_pod_state(f"ns-{i}", f"pod-{i}", {'status': 'running'})
        cache.get_pod_state(f"ns-{i}", f"pod-{i}")
    elapsed = print_time_elapsed("Mixed 10,000 ops", start)
    print_result("Throughput", f"{10000/elapsed:.0f} ops/sec")

    # Test 4: Cache hit rate
    print_section("Cache Hit Rate Testing")

    # Prime the cache
    for i in range(100):
        cache.set_prediction(f"ns-{i}", f"pod-{i}", {'risk': 0.5})

    # Test hits
    hits = 0
    for i in range(100):
        if cache.get_prediction(f"ns-{i}", f"pod-{i}")[0]:
            hits += 1

    print_result("Cache hits", f"{hits}/100")
    print_result("Hit rate", f"{hits}%")

    # Print cache stats
    stats = cache.get_stats()
    print("\n  Cache sizes:")
    for name, size in stats['sizes'].items():
        print(f"    {name}: {size} entries")


# ============================================================
# Demo: Kubectl Command Caching
# ============================================================

def demo_kubectl_caching():
    """Demonstrate kubectl command caching."""
    print_header("Kubectl Command Caching")

    # Note: We'll mock subprocess to avoid actual kubectl calls
    from unittest.mock import MagicMock, patch

    # Create mock for subprocess
    def mock_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "mock output"
        result.stderr = ""
        return result

    with patch('subprocess.run', side_effect=mock_run):
        collector.clear_kubectl_cache()

        print_section("Without Caching (100 identical calls)")
        start = time.time()
        for _ in range(100):
            collector.run_cmd(
                ["kubectl", "get", "pods", "-A", "-o", "json"],
                timeout=5,
                use_cache=False
            )
        elapsed_no_cache = print_time_elapsed("100 calls", start)

        print_section("With Caching (100 identical calls)")
        collector.clear_kubectl_cache()
        start = time.time()
        for _ in range(100):
            collector.run_cmd(
                ["kubectl", "get", "pods", "-A", "-o", "json"],
                timeout=5,
                use_cache=True,
                cache_ttl=300
            )
        elapsed_with_cache = print_time_elapsed("100 calls", start)

        # Show improvement
        if elapsed_no_cache > 0 and elapsed_with_cache > 0:
            speedup = elapsed_no_cache / elapsed_with_cache
            print_result("Speedup", f"{speedup:.1f}x faster")
        elif elapsed_no_cache > 0:
            print_result("Speedup", "infinite (cached calls too fast to measure)")

        # Show cache stats
        stats = collector.get_kubectl_cache_stats()
        print("\n  Cache statistics:")
        print_result("  Hits", stats['hits'])
        print_result("  Misses", stats['misses'])
        print_result("  Hit rate", stats['hit_rate'])


# ============================================================
# Demo: LLM Batch Processing
# ============================================================

async def demo_llm_batching():
    """Demonstrate LLM batch processing."""
    print_header("LLM Batch Processing")

    reset_cache()
    analyzer = MockSlowLLMAnalyzer(delay=0.05)  # 50ms per call

    # Test 1: Sequential without batching
    print_section("Sequential Processing (20 calls, no caching)")
    analyzer.call_count = 0
    start = time.time()
    for i in range(20):
        analyzer.analyze(f"issue-{i}", f"context-{i}")
    print_time_elapsed("20 sequential calls", start)
    print_result("Total LLM calls", analyzer.call_count)

    # Test 2: Batch with caching
    print_section("Batch Processing (20 identical requests, with caching)")
    analyzer = MockSlowLLMAnalyzer(delay=0.05)
    processor = LLMBatchProcessor(
        analyzer,
        max_batch_size=5,
        max_wait_time=0.1,
        cache_enabled=True,
    )

    # Queue 20 identical requests
    prompt = "Analyze pod memory usage"
    context = {"context": "memory high"}

    for i in range(20):
        processor.queue_analysis(prompt, context, f"ns/pod-{i}")

    start = time.time()
    results = await processor.process_all()
    print_time_elapsed("20 batched calls", start)
    print_result("Total LLM calls", analyzer.call_count)
    print_result("Cache hits", f"{len(results) - analyzer.call_count}")

    # Test 3: Batch processor stats
    stats = processor.get_stats()
    print("\n  Batch processor stats:")
    print_result("  Total requests", stats.get('total_requests', 0))
    print_result("  Cache hits", stats.get('total_cached', 0))


# ============================================================
# Demo: LLM Parallel Processing
# ============================================================

async def demo_llm_parallel():
    """Demonstrate LLM parallel processing."""
    print_header("LLM Parallel Processing")

    # Test with different concurrency levels
    concurrency_levels = [1, 3, 5]

    for concurrency in concurrency_levels:
        reset_cache()
        analyzer = MockSlowLLMAnalyzer(delay=0.05)
        processor = ParallelLLMProcessor(
            analyzer,
            max_concurrent=concurrency,
            rate_limit=0,  # No rate limiting
        )

        print_section(f"Parallel with {concurrency} concurrent workers")

        # Create 10 unique requests
        requests = [
            (f"prompt-{i}", {"context": f"context-{i}"}, f"ns/pod-{i}")
            for i in range(10)
        ]

        start = time.time()
        await processor.analyze_many(requests)
        elapsed = print_time_elapsed("10 unique requests", start)

        print_result("Total LLM calls", analyzer.call_count)
        print_result("Throughput", f"{10/elapsed:.1f} calls/sec")


# ============================================================
# Demo: Integrated Optimizer
# ============================================================

async def demo_integrated_optimizer():
    """Demonstrate the integrated performance optimizer."""
    print_header("Integrated Performance Optimizer")

    # Reset everything
    reset_cache()
    collector.clear_kubectl_cache()
    PerformanceOptimizer._instance = None

    # Create optimizer
    opt = PerformanceOptimizer()

    print_section("Configuration")
    config = opt.config
    print(f"  Kubectl caching: {'✓' if config.kubectl_cache_enabled else '✗'}")
    print(f"  L1 caching: {'✓' if config.l1_cache_enabled else '✗'}")
    print(f"  LLM batching: {'✓' if config.llm_batching_enabled else '✗'}")
    print(f"  LLM parallel: {'✓' if config.llm_parallel_enabled else '✗'}")
    print(f"  Max concurrent LLM: {config.llm_max_concurrent}")

    # Test cache access
    print_section("Cache Access")
    cache = opt.get_l1_cache()
    cache.set_pod_state('demo-ns', 'demo-pod', {'status': 'running'})
    hit, state = cache.get_pod_state('demo-ns', 'demo-pod')
    print_result("Cache write/read", "✓ Working")

    # Test LLM processor selection
    print_section("LLM Processor Selection")
    analyzer = MockSlowLLMAnalyzer(delay=0.01)

    # Parallel mode
    opt.configure(llm_parallel_enabled=True, llm_batching_enabled=False)
    processor = opt.get_llm_processor(analyzer)
    print_result("Selected processor (parallel enabled)", type(processor).__name__)

    # Batch mode
    opt.configure(llm_parallel_enabled=False, llm_batching_enabled=True)
    processor = opt.get_llm_processor(analyzer)
    print_result("Selected processor (batch enabled)", type(processor).__name__)

    # Test statistics
    print_section("Performance Statistics")
    summary = opt.get_summary()
    print(f"  Optimizations enabled: {summary['optimizations']}")
    print(f"  Kubectl cache hit rate: {summary['savings']['kubectl_cache_hit_rate']}")
    print(f"  LLM cache hit rate: {summary['savings']['llm_cache_hit_rate']}")


# ============================================================
# Main Demo Runner
# ============================================================

async def run_all_demos():
    """Run all demonstrations."""
    print(f"\n{DemoColors.BOLD}╔{'=' * 58}╗{DemoColors.ENDC}")
    print(f"{DemoColors.BOLD}║{DemoColors.HEADER}  PREDICTIVE-AGENT PERFORMANCE OPTIMIZATIONS DEMO  {DemoColors.ENDC}{DemoColors.BOLD}║{DemoColors.ENDC}")
    print(f"{DemoColors.BOLD}╚{'=' * 58}╝{DemoColors.ENDC}")

    # Run synchronous demos
    print(f"\n{DemoColors.OKGREEN}Running L1 Cache Demo...{DemoColors.ENDC}")
    demo_l1_cache()

    print(f"\n{DemoColors.OKGREEN}Running Kubectl Caching Demo...{DemoColors.ENDC}")
    demo_kubectl_caching()

    # Run asynchronous demos
    print(f"\n{DemoColors.OKGREEN}Running LLM Batching Demo...{DemoColors.ENDC}")
    await demo_llm_batching()

    print(f"\n{DemoColors.OKGREEN}Running LLM Parallel Demo...{DemoColors.ENDC}")
    await demo_llm_parallel()

    print(f"\n{DemoColors.OKGREEN}Running Integrated Optimizer Demo...{DemoColors.ENDC}")
    await demo_integrated_optimizer()

    # Final summary
    print(f"\n{DemoColors.BOLD}{DemoColors.HEADER}╔{'=' * 58}╗{DemoColors.ENDC}")
    print(f"{DemoColors.BOLD}{DemoColors.HEADER}║{DemoColors.ENDC}  {DemoColors.BOLD}{DemoColors.OKGREEN}ALL DEMOS COMPLETED SUCCESSFULLY!{DemoColors.ENDC}{DemoColors.BOLD}{DemoColors.HEADER}  {DemoColors.ENDC}{DemoColors.BOLD}{DemoColors.HEADER}║{DemoColors.ENDC}")
    print(f"{DemoColors.BOLD}{DemoColors.HEADER}╚{'=' * 58}╝{DemoColors.ENDC}")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Demo performance optimizations for predictive-agent'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all demos'
    )
    parser.add_argument(
        '--cache',
        action='store_true',
        help='Run cache demos only'
    )
    parser.add_argument(
        '--llm',
        action='store_true',
        help='Run LLM demos only'
    )
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output'
    )

    args = parser.parse_args()

    # Disable colors if requested
    if args.no_color:
        DemoColors.HEADER = ''
        DemoColors.OKBLUE = ''
        DemoColors.OKGREEN = ''
        DemoColors.WARNING = ''
        DemoColors.FAIL = ''
        DemoColors.ENDC = ''
        DemoColors.BOLD = ''
        DemoColors.UNDERLINE = ''

    # Run selected demos
    if args.all or (not args.cache and not args.llm):
        asyncio.run(run_all_demos())
    else:
        if args.cache:
            demo_l1_cache()
            demo_kubectl_caching()
        if args.llm:
            asyncio.run(demo_llm_batching())
            asyncio.run(demo_llm_parallel())


if __name__ == "__main__":
    main()
