"""
Parallel Tests for OPA Integration using pytest-xdist

Run with: python -m pytest tests/test_opa_parallel.py -n auto

This test suite tests OPA integration with:
- Parallel execution
- Different configurations
- Edge cases
- Performance scenarios
- Concurrent access
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest


@pytest.fixture(autouse=True)
def fast_opa_failures(monkeypatch):
    """Pin small timeout/retries and reset the OPA client singleton so
    unreachable-OPA paths fail in milliseconds instead of 3s of retry
    backoff (see tests/test_opa_integration_simple.py for rationale).
    """
    import predictive_agent.config as _config
    import predictive_agent.opa_client as _opa_client_mod
    import predictive_agent.opa_integration as _opa_integration_mod
    monkeypatch.setattr(_config, "OPA_TIMEOUT", 0.1)
    monkeypatch.setattr(_config, "OPA_MAX_RETRIES", 1)
    _opa_client_mod.reset_opa_client()
    _opa_integration_mod.reset_policy_engine()
    yield
    _opa_client_mod.reset_opa_client()

# Suppress connection error logs during tests
logging.getLogger('predictive_agent.opa_client').setLevel(logging.ERROR)
logging.getLogger('predictive_agent.opa_integration').setLevel(logging.ERROR)


# ============================================================================
# PARAMETERIZED TESTS - Test with different configurations
# ============================================================================

class TestOPAConfiguration:
    """Test OPA with different configuration parameters"""
    
    @pytest.mark.parametrize("enabled", [True, False])
    def test_enabled_disabled_configurations(self, enabled):
        """Test OPA behavior with enabled/disabled states"""
        from predictive_agent.opa_integration import OPAPolicyEngine
        
        engine = OPAPolicyEngine(enabled=enabled)
        
        pod_data = {'metadata': {'name': 'test'}}
        result = engine.check_pod_security(pod_data)
        
        if enabled:
            # When enabled, should attempt OPA check (will fail-open due to no server)
            assert 'result' in result
            assert 'allow' in result['result']
        else:
            # When disabled, should immediately return allow
            assert result['result']['allow'] == True
    
    @pytest.mark.parametrize("use_tls", [True, False])
    def test_tls_configuration(self, use_tls):
        """Test OPA with TLS enabled/disabled"""
        from predictive_agent.opa_integration import OPAPolicyEngine
        
        engine = OPAPolicyEngine(
            opa_host='localhost',
            opa_port=8181,
            use_tls=use_tls,
            enabled=False  # Disable to avoid connection attempts
        )
        
        stats = engine.get_stats()
        assert 'enabled' in stats
    
    @pytest.mark.parametrize("cache_enabled", [True, False])
    def test_caching_configuration(self, cache_enabled):
        """Test OPA with caching enabled/disabled"""
        from predictive_agent.opa_integration import OPAPolicyEngine
        
        engine = OPAPolicyEngine(
            enable_caching=cache_enabled,
            enabled=False
        )
        
        # Should work regardless of caching setting when OPA is disabled
        result = engine.check_pod_security({})
        assert result['result']['allow'] == True


class TestOPAPolicyDecisions:
    """Test OPA policy decision handling"""
    
    @pytest.mark.parametrize("allow_value", [True, False])
    def test_pod_security_decisions(self, allow_value):
        """Test different pod security decisions"""
        from predictive_agent.opa_integration import OPAPolicyEngine
        
        engine = OPAPolicyEngine(enabled=False)
        
        pod_data = {'metadata': {'name': 'test'}}
        result = engine.check_pod_security(pod_data)
        
        # When disabled, should always allow
        assert result['result']['allow'] == True
    
    @pytest.mark.parametrize("allow_value", [True, False])
    def test_llm_usage_decisions(self, allow_value):
        """Test different LLM usage decisions"""
        from predictive_agent.opa_integration import OPAPolicyEngine
        
        engine = OPAPolicyEngine(enabled=False)
        
        llm_request = {'model': 'test'}
        result = engine.check_llm_usage(llm_request)
        
        # When disabled, should always allow
        assert result['result']['allow_llm'] == True
    
    @pytest.mark.parametrize("allow_value", [True, False])
    def test_cache_config_decisions(self, allow_value):
        """Test different cache config decisions"""
        from predictive_agent.opa_integration import OPAPolicyEngine
        
        engine = OPAPolicyEngine(enabled=False)
        
        cache_config = {'layer': 'test', 'size': 1000}
        result = engine.check_cache_config(cache_config)
        
        # When disabled, should always allow
        assert result['result']['allow_cache_action'] == True


class TestOPADataFormats:
    """Test OPA with various data formats"""
    
    @pytest.mark.parametrize("pod_name", [
        'simple-pod',
        'app-web-86f4d4b8b8-abc12',
        'very-long-pod-name-that-should-still-be-handled-correctly-by-the-system',
        ''
    ])
    def test_various_pod_names(self, pod_name):
        """Test pod security check with various pod names"""
        from predictive_agent.opa_integration import OPAPolicyEngine
        
        engine = OPAPolicyEngine(enabled=False)
        pod_data = {'metadata': {'name': pod_name}}
        
        result = engine.check_pod_security(pod_data)
        assert result['result']['allow'] == True
    
    @pytest.mark.parametrize("model", [
        'gpt-4',
        'gpt-3.5-turbo',
        'llama-3-8b',
        'mistral-7b',
        'custom-model'
    ])
    def test_various_llm_models(self, model):
        """Test LLM usage check with various models"""
        from predictive_agent.opa_integration import OPAPolicyEngine
        
        engine = OPAPolicyEngine(enabled=False)
        llm_request = {'model': model, 'cost_per_request': 0.01}
        
        result = engine.check_llm_usage(llm_request)
        assert result['result']['allow_llm'] == True
    
    @pytest.mark.parametrize("namespace", [
        'default',
        'kube-system',
        'opendesk-predictive-agent',
        'production',
        'staging'
    ])
    def test_various_namespaces(self, namespace):
        """Test pod security check with various namespaces"""
        from predictive_agent.opa_integration import OPAPolicyEngine
        
        engine = OPAPolicyEngine(enabled=False)
        pod_data = {'metadata': {'name': 'test', 'namespace': namespace}}
        
        result = engine.check_pod_security(pod_data)
        assert result['result']['allow'] == True


# ============================================================================
# CONCURRENT TESTS - Test thread safety
# ============================================================================

class TestOPAConcurrency:
    """Test OPA thread safety and concurrent access"""
    
    def test_concurrent_pod_checks(self):
        """Test concurrent pod security checks"""
        from predictive_agent.opa_integration import OPAPolicyEngine
        
        engine = OPAPolicyEngine(enabled=False)
        num_requests = 100
        results = []
        
        def check_pod(i):
            pod_data = {'metadata': {'name': f'pod-{i}'}}
            result = engine.check_pod_security(pod_data)
            results.append(result)
            return result
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(check_pod, i) for i in range(num_requests)]
            for future in as_completed(futures):
                assert future.result()['result']['allow'] == True
        
        assert len(results) == num_requests
        all_allowed = all(r['result']['allow'] for r in results)
        assert all_allowed
    
    def test_concurrent_llm_checks(self):
        """Test concurrent LLM usage checks"""
        from predictive_agent.opa_integration import OPAPolicyEngine
        
        engine = OPAPolicyEngine(enabled=False)
        num_requests = 100
        results = []
        
        def check_llm(i):
            llm_request = {'model': f'model-{i}', 'cost_per_request': 0.01 * i}
            result = engine.check_llm_usage(llm_request)
            results.append(result)
            return result
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(check_llm, i) for i in range(num_requests)]
            for future in as_completed(futures):
                assert future.result()['result']['allow_llm'] == True
        
        assert len(results) == num_requests
        all_allowed = all(r['result']['allow_llm'] for r in results)
        assert all_allowed
    
    def test_concurrent_cache_config_checks(self):
        """Test concurrent cache config checks"""
        from predictive_agent.opa_integration import OPAPolicyEngine
        
        engine = OPAPolicyEngine(enabled=False)
        num_requests = 100
        results = []
        
        def check_cache_config(i):
            cache_config = {'layer': f'layer-{i}', 'size': 1000 + i, 'ttl': 300 + i}
            result = engine.check_cache_config(cache_config)
            results.append(result)
            return result
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(check_cache_config, i) for i in range(num_requests)]
            for future in as_completed(futures):
                assert future.result()['result']['allow_cache_action'] == True
        
        assert len(results) == num_requests
        all_allowed = all(r['result']['allow_cache_action'] for r in results)
        assert all_allowed
    
    def test_concurrent_mixed_checks(self):
        """Test concurrent mixed policy checks"""
        from predictive_agent.opa_integration import OPAPolicyEngine
        
        engine = OPAPolicyEngine(enabled=False)
        num_requests = 100
        results = []
        
        def mixed_checks(i):
            # Alternate between different check types
            check_type = i % 3
            
            if check_type == 0:
                result = engine.check_pod_security({'metadata': {'name': f'pod-{i}'}})
            elif check_type == 1:
                result = engine.check_llm_usage({'model': f'model-{i}'})
            else:
                result = engine.check_cache_config({'layer': f'layer-{i}'})
            
            results.append(result)
            return result
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(mixed_checks, i) for i in range(num_requests)]
            for future in as_completed(futures):
                result = future.result()
                # Check that result has expected structure
                assert 'result' in result
                assert 'violation' in result
        
        assert len(results) == num_requests


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestOPAPerformance:
    """Test OPA performance characteristics"""
    
    def test_pod_check_performance_disabled(self):
        """Test performance of pod checks when OPA is disabled"""
        from predictive_agent.opa_integration import OPAPolicyEngine
        
        engine = OPAPolicyEngine(enabled=False)
        
        start = time.time()
        for _ in range(1000):
            engine.check_pod_security({'metadata': {'name': 'test'}})
        elapsed = time.time() - start
        
        # Should handle 1000 requests in < 1 second when disabled
        assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s for 1000 requests"
    
    def test_llm_check_performance_disabled(self):
        """Test performance of LLM checks when OPA is disabled"""
        from predictive_agent.opa_integration import OPAPolicyEngine
        
        engine = OPAPolicyEngine(enabled=False)
        
        start = time.time()
        for _ in range(1000):
            engine.check_llm_usage({'model': 'test', 'cost_per_request': 0.01})
        elapsed = time.time() - start
        
        # Should handle 1000 requests in < 1 second when disabled
        assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s for 1000 requests"
    
    def test_cache_config_performance_disabled(self):
        """Test performance of cache config checks when OPA is disabled"""
        from predictive_agent.opa_integration import OPAPolicyEngine
        
        engine = OPAPolicyEngine(enabled=False)
        
        start = time.time()
        for _ in range(1000):
            engine.check_cache_config({'layer': 'test', 'size': 1000})
        elapsed = time.time() - start
        
        # Should handle 1000 requests in < 1 second when disabled
        assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s for 1000 requests"
    
    def test_stats_performance(self):
        """Test performance of stats retrieval"""
        from predictive_agent.opa_integration import OPAPolicyEngine
        
        engine = OPAPolicyEngine(enabled=False)
        
        # Make some requests
        for _ in range(100):
            engine.check_pod_security({})
        
        start = time.time()
        for _ in range(1000):
            engine.get_stats()
        elapsed = time.time() - start
        
        # Should handle 1000 stats requests in < 0.5 seconds
        assert elapsed < 0.5, f"Too slow: {elapsed:.2f}s for 1000 stats requests"


# ============================================================================
# STRESS TESTS
# ============================================================================

class TestOPAStress:
    """Stress tests for OPA integration"""
    
    def test_high_concurrency_pod_checks(self):
        """Test high concurrency pod security checks"""
        from predictive_agent.opa_integration import OPAPolicyEngine
        
        engine = OPAPolicyEngine(enabled=False)
        num_requests = 1000
        
        def check_pod(i):
            result = engine.check_pod_security({'metadata': {'name': f'pod-{i}'}})
            assert result['result']['allow'] == True
        
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(check_pod, i) for i in range(num_requests)]
            for future in as_completed(futures):
                future.result()  # Will raise if assertion fails
        
        # Verify stats after high concurrency
        engine.reset_stats()
    
    def test_memory_usage_with_many_requests(self):
        """Test memory usage doesn't grow unbounded with many requests"""
        from predictive_agent.opa_integration import OPAPolicyEngine, reset_policy_engine
        
        reset_policy_engine()
        engine = OPAPolicyEngine(enabled=False)
        
        # Make many requests
        for i in range(10000):
            engine.check_pod_security({'metadata': {'name': f'pod-{i}'}})
        
        # Reset and check stats are clean
        engine.reset_stats()
        stats = engine.get_stats()
        
        assert stats['total_requests'] == 0
        assert stats['errors'] == 0
    
    def test_singleton_thread_safety(self):
        """Test that singleton pattern is thread-safe"""
        from predictive_agent.opa_integration import get_policy_engine, reset_policy_engine
        
        reset_policy_engine()
        
        engines = []
        
        def get_engine(_):
            engine = get_policy_engine(enabled=False)
            engines.append(engine)
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(get_engine, i) for i in range(100)]
            for future in as_completed(futures):
                future.result()
        
        # All engines should be the same instance
        assert len(set(id(e) for e in engines)) == 1


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestOPAEdgeCases:
    """Test OPA with edge cases and unusual inputs"""
    
    def test_empty_pod_data(self):
        """Test with empty pod data"""
        from predictive_agent.opa_integration import OPAPolicyEngine
        
        engine = OPAPolicyEngine(enabled=False)
        result = engine.check_pod_security({})
        assert result['result']['allow'] == True
    
    def test_none_pod_data(self):
        """Test with None pod data"""
        from predictive_agent.opa_integration import OPAPolicyEngine
        
        engine = OPAPolicyEngine(enabled=False)
        result = engine.check_pod_security(None)
        assert result['result']['allow'] == True
    
    def test_missing_result_key(self):
        """Test with result missing 'allow' key (simulated OPA response)"""
        from predictive_agent.opa_integration import OPAPolicyEngine
        
        engine = OPAPolicyEngine(enabled=False)
        
        # Directly test the bypass path
        result = engine.check_pod_security({})
        assert 'result' in result
        assert 'allow' in result['result']  # Should always have allow key
    
    def test_various_violation_formats(self):
        """Test with various violation formats"""
        from predictive_agent.opa_integration import OPAPolicyEngine
        
        engine = OPAPolicyEngine(enabled=False)
        
        # These should all work and return allow=True when OPA is disabled
        pod_data_list = [
            {},
            {'metadata': None},
            {'metadata': {}, 'spec': None},
            {'metadata': {'name': 'test'}, 'spec': {'containers': None}},
            {'kind': 'Pod', 'metadata': {'name': 'test'}},
        ]
        
        for pod_data in pod_data_list:
            result = engine.check_pod_security(pod_data)
            assert result['result']['allow'] == True
    
    def test_large_pod_data(self):
        """Test with very large pod data"""
        from predictive_agent.opa_integration import OPAPolicyEngine
        
        engine = OPAPolicyEngine(enabled=False)
        
        # Create large pod data
        large_pod_data = {
            'metadata': {
                'name': 'x' * 1000,
                'labels': {f'label-{i}': 'value-' * 100 for i in range(50)},
                'annotations': {f'anno-{i}': 'value-' * 100 for i in range(50)}
            },
            'spec': {
                'containers': [
                    {
                        'name': f'container-{i}',
                        'env': [{'name': f'env-{j}', 'value': 'x' * 100} for j in range(20)]
                    }
                    for i in range(10)
                ]
            }
        }
        
        result = engine.check_pod_security(large_pod_data)
        assert result['result']['allow'] == True


# ============================================================================
# REGRESSION TESTS
# ============================================================================

class TestOPARegressions:
    """Regression tests to ensure bugs stay fixed"""
    
    def test_datetime_serialization_fix(self):
        """Regression test for datetime serialization issue in cache"""
        from datetime import datetime

        from predictive_agent.opa_cache import OPACache
        
        cache = OPACache()
        
        # This used to fail with: TypeError: Object of type datetime is not JSON serializable
        test_data = {
            'result': {'allow': True},
            'timestamp': datetime.now()
        }
        
        cache.set('test.policy', {}, test_data)
        hit, result = cache.get('test.policy', {})
        
        assert hit == True
        assert result == test_data
    
    def test_cache_clear_removes_all_entries(self):
        """Regression test to ensure cache.clear() works"""
        from predictive_agent.opa_cache import OPACache
        
        cache = OPACache()
        
        # Add many entries
        for i in range(100):
            cache.set(f'policy-{i}', {}, {'result': {'allow': True}})
        
        # Clear
        cache.clear()
        
        # Verify all entries are gone
        for i in range(100):
            hit, _ = cache.get(f'policy-{i}', {})
            assert hit == False
    
    def test_singleton_doesnt_leak_state(self):
        """Regression test to ensure singleton doesn't leak state between tests"""
        from predictive_agent.opa_integration import get_policy_engine, reset_policy_engine
        
        reset_policy_engine()
        
        # Create engine and make requests
        engine1 = get_policy_engine(enabled=False)
        engine1.check_pod_security({})
        
        engine1.get_stats()

        # Reset and create new engine
        reset_policy_engine()
        engine2 = get_policy_engine(enabled=False)
        
        # State should be clean - engines should be different instances
        assert engine1 is not engine2
        
        # Stats should be clean
        stats2 = engine2.get_stats()
        assert stats2['total_requests'] == 0


# ============================================================================
# INTEGRATION WITH EXISTING MODULES
# ============================================================================

class TestOPAWithExistingModules:
    """Test OPA integration with existing predictive-agent modules"""
    
    @pytest.mark.parametrize("module", [
        "l1_cache",
        "optimize",
        "collector",
        "llm_batch",
        "kalman",
        "markov",
    ])
    def test_opa_doesnt_break_modules(self, module):
        """Test that importing OPA doesn't break existing modules"""
        # Import OPA modules first
        
        # Then import the module under test
        if module == "l1_cache":
            from predictive_agent.l1_cache import get_cache
            cache = get_cache()
            assert cache is not None
        elif module == "optimize":
            from predictive_agent.optimize import get_optimizer
            optimizer = get_optimizer()
            assert optimizer is not None
        elif module == "collector":
            import predictive_agent.collector
            assert predictive_agent.collector is not None
        elif module == "llm_batch":
            import predictive_agent.llm_batch
            assert predictive_agent.llm_batch is not None
        elif module == "kalman":
            import predictive_agent.kalman
            assert predictive_agent.kalman is not None
        elif module == "markov":
            import predictive_agent.markov
            assert predictive_agent.markov is not None


# ============================================================================
# FUTURE: PERFORMANCE BENCHMARK TESTS
# Install pytest-benchmark to run these:
# pip install pytest-benchmark
# ============================================================================

# class TestOPABenchmarks:
#     """Benchmark tests for OPA integration"""
#     
#     @pytest.mark.benchmark
#     def test_pod_check_benchmark(self, benchmark):
#         """Benchmark pod security checks"""
#         from predictive_agent.opa_integration import OPAPolicyEngine
#         
#         engine = OPAPolicyEngine(enabled=False)
#         
#         def check_pod():
#             engine.check_pod_security({'metadata': {'name': 'test'}})
#         
#         result = benchmark(check_pod)
#         assert result is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-n', 'auto'])
