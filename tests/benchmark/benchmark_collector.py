"""Benchmark tests for collector performance.

Tests:
- kubectl command caching
- Pod discovery speed
- Metrics collection performance
- Cache hit rates
"""

import time
import unittest
from unittest.mock import patch, MagicMock
import subprocess

from predictive_agent import collector
from predictive_agent.l1_cache import get_cache, reset_cache


class BenchmarkCollectorKubectlCaching(unittest.TestCase):
    """Benchmark kubectl command caching performance."""

    def setUp(self):
        """Reset cache before each test."""
        collector.clear_kubectl_cache()
        reset_cache()

    def tearDown(self):
        """Clean up after each test."""
        collector.clear_kubectl_cache()
        reset_cache()

    @patch('subprocess.run')
    def test_run_cmd_without_cache(self, mock_run):
        """Test run_cmd without caching (baseline)."""
        # Setup mock
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "test output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # Run without cache
        cmd = ["kubectl", "get", "pods"]
        start = time.perf_counter()

        for _ in range(100):
            collector.run_cmd(cmd, timeout=1, use_cache=False)

        elapsed = time.perf_counter() - start

        # Should have called subprocess 100 times
        self.assertEqual(mock_run.call_count, 100)

        print(f"\n✓ Without cache: 100 calls in {elapsed:.4f}s ({elapsed/100*1000:.2f}ms per call)")
        return elapsed

    @patch('subprocess.run')
    def test_run_cmd_with_cache(self, mock_run):
        """Test run_cmd with caching."""
        # Setup mock
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "test output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # Run with cache
        cmd = ["kubectl", "get", "pods"]
        start = time.perf_counter()

        for _ in range(100):
            collector.run_cmd(cmd, timeout=1, use_cache=True)

        elapsed = time.perf_counter() - start

        # Should have called subprocess only once (first call)
        self.assertEqual(mock_run.call_count, 1)

        # Check cache stats
        stats = collector.get_kubectl_cache_stats()
        self.assertEqual(stats['hits'], 99)
        self.assertEqual(stats['misses'], 1)

        print(f"\n✓ With cache: 100 calls in {elapsed:.4f}s ({elapsed/100*1000:.2f}ms per call)")
        print(f"  Cache stats: {stats['hits']} hits, {stats['misses']} misses ({stats['hit_rate']})")

        return elapsed

    @patch('subprocess.run')
    def test_cache_expires(self, mock_run):
        """Test that cache entries expire correctly."""
        # Setup mock
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "test output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        cmd = ["kubectl", "get", "pods"]

        # First call - should hit subprocess
        collector.run_cmd(cmd, timeout=1, use_cache=True, cache_ttl=0.1)
        self.assertEqual(mock_run.call_count, 1)

        # Second call immediately - should hit cache
        collector.run_cmd(cmd, timeout=1, use_cache=True, cache_ttl=0.1)
        self.assertEqual(mock_run.call_count, 1)

        # Wait for cache to expire
        time.sleep(0.11)

        # Third call - should hit subprocess again
        collector.run_cmd(cmd, timeout=1, use_cache=True, cache_ttl=0.1)
        self.assertEqual(mock_run.call_count, 2)

        print("\n✓ Cache expiration works correctly")


class BenchmarkCollectorL1Cache(unittest.TestCase):
    """Benchmark L1 cache performance."""

    def setUp(self):
        """Reset cache before each test."""
        reset_cache()

    def test_cache_throughput(self):
        """Test L1 cache read/write throughput."""
        cache = get_cache()

        # Write test
        start = time.perf_counter()
        for i in range(1000):
            cache.set_pod_state(f"ns-{i}", f"pod-{i}", {"status": "running"})
        write_time = time.perf_counter() - start

        # Read test (all hits)
        start = time.perf_counter()
        for i in range(1000):
            cache.get_pod_state(f"ns-{i}", f"pod-{i}")
        read_time = time.perf_counter() - start

        print(f"\n✓ L1 Cache Performance:")
        print(f"  Write: 1000 entries in {write_time:.4f}s ({1000/write_time:.0f} writes/sec)")
        print(f"  Read: 1000 entries in {read_time:.4f}s ({1000/read_time:.0f} reads/sec)")

        # Combined operations
        start = time.perf_counter()
        for i in range(1000):
            cache.set_pod_state(f"ns-{i}", f"pod-{i}", {"status": f"updated-{i}"})
            cache.get_pod_state(f"ns-{i}", f"pod-{i}")
        combined_time = time.perf_counter() - start
        print(f"  Combined: 2000 ops in {combined_time:.4f}s ({2000/combined_time:.0f} ops/sec)")


class BenchmarkCollectorE2E(unittest.TestCase):
    """End-to-end benchmarks for collector performance."""

    def setUp(self):
        """Reset cache before each test."""
        collector.clear_kubectl_cache()
        reset_cache()

    @patch('subprocess.run')
    def test_collect_top_pods_batching(self, mock_run):
        """Test top pods collection with and without batching."""
        # Mock kubectl top pods output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = """NAMESPACE     NAME      CPU(cores)   MEMORY(bytes)
default       pod-1     10m          128Mi
default       pod-2     20m          256Mi
kube-system   pod-3     5m           64Mi"""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # Without batching (individual calls)
        start = time.perf_counter()
        for _ in range(10):
            collector.run_cmd(
                ["kubectl", "top", "pods", "-A"],
                timeout=5,
                use_cache=False
            )
        no_batch_time = time.perf_counter() - start

        # With batching (single call, cached)
        collector.clear_kubectl_cache()
        start = time.perf_counter()
        for _ in range(10):
            collector.run_cmd(
                ["kubectl", "top", "pods", "-A"],
                timeout=5,
                use_cache=True
            )
        batch_time = time.perf_counter() - start

        print(f"\n✓ kubectl top pods collection:")
        print(f"  Without batching: 10 calls in {no_batch_time:.4f}s")
        print(f"  With caching: 10 calls in {batch_time:.4f}s")
        print(f"  Speedup: {no_batch_time/batch_time:.1f}x")


if __name__ == "__main__":
    # Run benchmarks
    print("=" * 60)
    print("COLLECTOR PERFORMANCE BENCHMARKS")
    print("=" * 60)

    # Kubectl caching
    suite = unittest.TestLoader()

    print("\n" + "-" * 40)
    print("Testing kubectl command caching")
    print("-" * 40)
    runner = unittest.TextTestRunner(verbosity=0)
    runner.run(suite.loadTestsFromTestCase(BenchmarkCollectorKubectlCaching))

    print("\n" + "-" * 40)
    print("Testing L1 cache performance")
    print("-" * 40)
    runner.run(suite.loadTestsFromTestCase(BenchmarkCollectorL1Cache))

    print("\n" + "-" * 40)
    print("Testing end-to-end collector performance")
    print("-" * 40)
    runner.run(suite.loadTestsFromTestCase(BenchmarkCollectorE2E))

    print("\n" + "=" * 60)
    print("BENCHMARKS COMPLETE")
    print("=" * 60)
