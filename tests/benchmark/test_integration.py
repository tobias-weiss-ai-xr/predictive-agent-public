"""Integration test for performance optimizations.

Tests that all optimization modules work together correctly.
"""

import asyncio
import time
import unittest

from predictive_agent.llm import LLMAnalyzer, LLMBackend
from predictive_agent.l1_cache import get_cache, reset_cache
from predictive_agent.llm_batch import LLMBatchProcessor, ParallelLLMProcessor
from predictive_agent.optimize import PerformanceOptimizer, get_optimizer


class MockAnalyzer(LLMAnalyzer):
    """Mock analyzer for testing."""

    def __init__(self):
        self.call_count = 0
        self.delay = 0.01

    def analyze(self, issue, context, prediction=None):
        self.call_count += 1
        time.sleep(self.delay)
        return {
            'analysis': f'Analyzed {issue}',
            'severity': 'medium',
            'action': 'Monitor',
            'command': '',
        }


class TestOptimizerIntegration(unittest.TestCase):
    """Test that the optimizer integrates all components correctly."""

    def setUp(self):
        reset_cache()
        # Reset singleton
        PerformanceOptimizer._instance = None

    def test_optimizer_singleton(self):
        """Test that optimizer is a singleton."""
        opt1 = PerformanceOptimizer()
        opt2 = get_optimizer()
        self.assertIs(opt1, opt2)

    def test_get_cache(self):
        """Test that cache is accessible through optimizer."""
        opt = PerformanceOptimizer()
        cache = opt.get_l1_cache()
        self.assertIsNotNone(cache)

        # Test cache operations
        cache.set_pod_state('test-ns', 'test-pod', {'status': 'running'})
        hit, state = cache.get_pod_state('test-ns', 'test-pod')
        self.assertTrue(hit)
        self.assertEqual(state['status'], 'running')

    def test_batch_processor_creation(self):
        """Test batch processor can be created through optimizer."""
        opt = PerformanceOptimizer()
        analyzer = MockAnalyzer()

        processor = opt.get_batch_processor(analyzer)
        self.assertIsInstance(processor, LLMBatchProcessor)

    def test_parallel_processor_creation(self):
        """Test parallel processor can be created through optimizer."""
        opt = PerformanceOptimizer()
        analyzer = MockAnalyzer()

        processor = opt.get_parallel_processor(analyzer)
        self.assertIsInstance(processor, ParallelLLMProcessor)

    def test_llm_processor_selection(self):
        """Test that optimizer selects the right processor based on config."""
        opt = PerformanceOptimizer()
        analyzer = MockAnalyzer()

        # Test parallel (default when enabled)
        opt.configure(llm_parallel_enabled=True, llm_batching_enabled=False)
        processor = opt.get_llm_processor(analyzer)
        self.assertIsInstance(processor, ParallelLLMProcessor)

        # Test batching
        opt.configure(llm_parallel_enabled=False, llm_batching_enabled=True)
        processor = opt.get_llm_processor(analyzer)
        self.assertIsInstance(processor, LLMBatchProcessor)

        # Test simple (both disabled)
        opt.configure(llm_parallel_enabled=False, llm_batching_enabled=False)
        processor = opt.get_llm_processor(analyzer)
        # Should return simple processor
        self.assertIsNotNone(processor)

    def test_stats_collection(self):
        """Test that optimizer collects statistics."""
        opt = PerformanceOptimizer()
        stats = opt.get_stats()

        self.assertIsNotNone(stats)
        self.assertIsInstance(stats.to_dict(), dict)

    def test_summary_generation(self):
        """Test that optimizer generates summary."""
        opt = PerformanceOptimizer()
        summary = opt.get_summary()

        self.assertIn('optimizations', summary)
        self.assertIn('savings', summary)
        self.assertIn('stats', summary)

    async def test_batch_processor_working(self):
        """Test that batch processor works with mock analyzer."""
        opt = PerformanceOptimizer()
        analyzer = MockAnalyzer()
        processor = opt.get_batch_processor(analyzer)

        # Queue some requests
        processor.queue_analysis("test prompt", {}, "ns/pod-1")
        processor.queue_analysis("test prompt", {}, "ns/pod-2")

        # Process them
        results = await processor.process_all()

        self.assertEqual(len(results), 2)
        self.assertEqual(analyzer.call_count, 2)  # Different pod keys = different calls

    async def test_parallel_processor_working(self):
        """Test that parallel processor works with mock analyzer."""
        analyzer = MockAnalyzer()
        processor = ParallelLLMProcessor(
            analyzer,
            max_concurrent=2,
        )

        # Process multiple requests
        requests = [
            ("prompt 1", {}, "ns/pod-1"),
            ("prompt 2", {}, "ns/pod-2"),
            ("prompt 3", {}, "ns/pod-3"),
        ]

        results = await processor.analyze_many(requests)

        self.assertEqual(len(results), 3)
        self.assertEqual(analyzer.call_count, 3)

    def test_cache_clearing(self):
        """Test that optimizer can clear all caches."""
        opt = PerformanceOptimizer()
        cache = get_cache()

        # Add some data
        cache.set_pod_state('ns1', 'pod1', {'status': 'running'})
        cache.set_prediction('ns1', 'pod1', {'risk': 0.5})

        # Verify data is there
        self.assertTrue(cache.get_pod_state('ns1', 'pod1')[0])
        self.assertTrue(cache.get_prediction('ns1', 'pod1')[0])

        # Clear all caches
        opt.clear_caches()

        # Verify data is gone
        self.assertFalse(cache.get_pod_state('ns1', 'pod1')[0])
        self.assertFalse(cache.get_prediction('ns1', 'pod1')[0])

    def test_configuration_updates(self):
        """Test that configuration can be updated."""
        opt = PerformanceOptimizer()

        # Default values
        self.assertTrue(opt.config.kubectl_cache_enabled)
        self.assertTrue(opt.config.l1_cache_enabled)

        # Update config
        opt.configure(
            kubectl_cache_enabled=False,
            l1_cache_enabled=False,
        )

        # Verify changes
        self.assertFalse(opt.config.kubectl_cache_enabled)
        self.assertFalse(opt.config.l1_cache_enabled)


if __name__ == "__main__":
    unittest.main()
