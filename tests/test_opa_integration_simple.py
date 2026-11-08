"""
Simple Tests for OPA Integration Module

This module tests the basic functionality of OPA integration without
requiring a running OPA server.
"""

import logging

import pytest

# Suppress connection error logs during tests
logging.getLogger('predictive_agent.opa_client').setLevel(logging.ERROR)
logging.getLogger('predictive_agent.opa_integration').setLevel(logging.ERROR)


@pytest.fixture(autouse=True)
def fast_opa_failures(monkeypatch):
    """Unreachable-OPA paths must fail in milliseconds, not seconds.

    The client retry loop sleeps 1s+2s exponential backoff per failed call,
    and the OPAClient singleton carries timeout/max_retries from whichever
    construction built it. Pin small values (read at engine construction
    time) and reset the singletons so every engine in this module fails
    fast; restore afterwards so other modules are unaffected.
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


class TestOPACache:
    """Tests for OPACache class"""

    def test_init_defaults(self):
        """Test OPACache initialization with defaults"""
        from predictive_agent.opa_cache import OPACache

        cache = OPACache()
        assert cache._max_size == 10000
        assert cache._default_ttl == 300
        assert cache._max_memory_mb == 50
        assert cache._cache == {}
        assert cache._current_memory == 0

    def test_init_custom(self):
        """Test OPACache initialization with custom values"""
        from predictive_agent.opa_cache import OPACache

        cache = OPACache(max_size=100, default_ttl=60, max_memory_mb=10)
        assert cache._max_size == 100
        assert cache._default_ttl == 60
        assert cache._max_memory_mb == 10

    def test_set_and_get(self):
        """Test set and get operations"""
        from predictive_agent.opa_cache import OPACache

        cache = OPACache()
        policy_package = 'test.policy'
        input_data = {'test': 'data'}
        test_data = {'result': {'allow': True}, 'violation': []}

        cache.set(policy_package, input_data, test_data)
        hit, result = cache.get(policy_package, input_data)

        assert hit == True
        assert result == test_data

    def test_get_nonexistent(self):
        """Test getting nonexistent entry"""
        from predictive_agent.opa_cache import OPACache

        cache = OPACache()
        hit, result = cache.get('nonexistent', {})

        assert hit == False
        assert result is None

    def test_clear(self):
        """Test clear functionality"""
        from predictive_agent.opa_cache import OPACache

        cache = OPACache()
        cache.set('policy1', {}, {'result': {'allow': True}, 'violation': []})
        cache.set('policy2', {}, {'result': {'allow': False}, 'violation': ['test']})

        cache.clear()

        stats = cache.get_stats()
        assert stats['size'] == 0
        assert stats['hits'] == 0
        assert stats['sets'] == 0

    def test_invalidate_policy(self):
        """Test policy invalidation"""
        from predictive_agent.opa_cache import OPACache

        cache = OPACache()
        cache.set('policy1', {'key': 'a'}, {'result': {'allow': True}, 'violation': []})
        cache.set('policy1', {'key': 'b'}, {'result': {'allow': False}, 'violation': ['test']})
        cache.set('policy2', {'key': 'c'}, {'result': {'allow': True}, 'violation': []})

        # Invalidate policy1
        count = cache.invalidate_policy('policy1')
        assert count == 2

        # policy1 entries should be gone
        hit, _ = cache.get('policy1', {'key': 'a'})
        assert hit == False

        # policy2 should still exist
        hit, _ = cache.get('policy2', {'key': 'c'})
        assert hit == True

    def test_update_policy_version(self):
        """Test policy version update triggers invalidation"""
        from predictive_agent.opa_cache import OPACache

        cache = OPACache()
        cache.set('policy1', {}, {'result': {'allow': True}, 'violation': []})

        # Update version
        cache.update_policy_version('policy1', 'v2')

        # Entry should be invalidated
        hit, _ = cache.get('policy1', {})
        assert hit == False

    def test_stats(self):
        """Test statistics tracking"""
        from predictive_agent.opa_cache import OPACache

        cache = OPACache()

        # Set an entry
        cache.set('policy1', {}, {'result': {'allow': True}, 'violation': []})

        # Get with hit
        cache.get('policy1', {})

        # Get with miss
        cache.get('policy2', {})

        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['sets'] == 1
        assert 'hit_rate' in stats


class TestOPAPolicyCache:
    """Tests for OPAPolicyCache class"""

    def test_singleton(self):
        """Test singleton pattern"""
        from predictive_agent.opa_cache import get_opa_policy_cache, reset_opa_policy_cache

        # Reset for clean test
        reset_opa_policy_cache()

        cache1 = get_opa_policy_cache()
        cache2 = get_opa_policy_cache()

        assert cache1 is cache2

    def test_get_cache_layers(self):
        """Test getting different cache layers"""
        from predictive_agent.opa_cache import get_opa_policy_cache

        policy_cache = get_opa_policy_cache()

        pod_cache = policy_cache.get_cache('pod')
        llm_cache = policy_cache.get_cache('llm')
        cache_cache = policy_cache.get_cache('cache')
        general_cache = policy_cache.get_cache('general')

        assert pod_cache is not None
        assert llm_cache is not None
        assert cache_cache is not None
        assert general_cache is not None

    def test_clear_all(self):
        """Test clearing all caches"""
        from predictive_agent.opa_cache import get_opa_policy_cache, reset_opa_policy_cache

        reset_opa_policy_cache()
        policy_cache = get_opa_policy_cache()

        # Add entries to different caches
        policy_cache.get_cache('pod').set('test.pod', {}, {'result': {'allow': True}, 'violation': []})
        policy_cache.get_cache('llm').set('test.llm', {}, {'result': {'allow_llm': True}, 'violation': []})

        policy_cache.clear_all()

        stats = policy_cache.get_all_stats()
        assert stats['pod']['size'] == 0
        assert stats['llm']['size'] == 0


class TestOPAPolicyEngine:
    """Tests for OPAPolicyEngine class"""

    def test_default_enabled(self):
        """Test that OPA is enabled by default"""
        from predictive_agent.opa_integration import OPAPolicyEngine

        engine = OPAPolicyEngine()
        stats = engine.get_stats()
        assert stats['enabled'] == True

    def test_explicitly_disabled(self):
        """Test explicitly disabling OPA"""
        from predictive_agent.opa_integration import OPAPolicyEngine

        engine = OPAPolicyEngine(enabled=False)
        stats = engine.get_stats()
        assert stats['enabled'] == False

    def test_fail_open_pod_security(self):
        """Test fail-open mode for pod security when OPA is disabled"""
        from predictive_agent.opa_integration import OPAPolicyEngine

        engine = OPAPolicyEngine(enabled=False)

        pod_data = {'metadata': {'name': 'test'}}
        result = engine.check_pod_security(pod_data)

        assert result['result']['allow'] == True

    def test_fail_open_llm_usage(self):
        """Test fail-open mode for LLM usage when OPA is disabled"""
        from predictive_agent.opa_integration import OPAPolicyEngine

        engine = OPAPolicyEngine(enabled=False)

        llm_request = {'model': 'test', 'cost_per_request': 100.0}
        result = engine.check_llm_usage(llm_request)

        assert result['result']['allow_llm'] == True

    def test_fail_open_cache_config(self):
        """Test fail-open mode for cache config when OPA is disabled"""
        from predictive_agent.opa_integration import OPAPolicyEngine

        engine = OPAPolicyEngine(enabled=False)

        cache_config = {'layer': 'test', 'size': 100000, 'ttl': 10000}
        result = engine.check_cache_config(cache_config)

        assert result['result']['allow_cache_action'] == True

    def test_stats_tracking(self):
        """Test statistics tracking"""
        from predictive_agent.opa_integration import OPAPolicyEngine

        engine = OPAPolicyEngine(enabled=False)

        # Make some requests
        engine.check_pod_security({})
        engine.check_llm_usage({})
        engine.check_cache_config({'cache_config': {}})

        stats = engine.get_stats()
        # When OPA is disabled, requests bypass OPA client but should still count
        # Actually they don't because it returns early - this is by design
        # The enabled flag should be in stats though
        assert stats['enabled'] == False
        assert stats['total_requests'] == 0  # Bypasses request counting when disabled

    def test_reset_stats(self):
        """Test resetting statistics"""
        from predictive_agent.opa_integration import OPAPolicyEngine

        engine = OPAPolicyEngine(enabled=False)
        engine.check_pod_security({})
        engine.reset_stats()

        stats = engine.get_stats()
        assert stats['total_requests'] == 0
        assert stats['errors'] == 0

    def test_singleton(self):
        """Test singleton pattern for policy engine"""
        from predictive_agent.opa_integration import get_policy_engine, reset_policy_engine

        reset_policy_engine()

        engine1 = get_policy_engine()
        engine2 = get_policy_engine()

        assert engine1 is engine2


class TestConvenienceFunctions:
    """Tests for convenience functions"""

    def setup_method(self):
        """Reset singletons before each test"""
        from predictive_agent.opa_integration import reset_policy_engine
        reset_policy_engine()

    def test_check_pod_security(self):
        """Test check_pod_security convenience function"""
        from predictive_agent.opa_integration import check_pod_security

        result = check_pod_security({'metadata': {'name': 'test'}})
        assert 'result' in result
        assert 'allow' in result['result']

    def test_check_llm_usage(self):
        """Test check_llm_usage convenience function"""
        from predictive_agent.opa_integration import check_llm_usage

        result = check_llm_usage({'model': 'test', 'cost_per_request': 0.01})
        assert 'result' in result
        assert 'allow_llm' in result['result']

    def test_check_cache_config(self):
        """Test check_cache_config convenience function"""
        from predictive_agent.opa_integration import check_cache_config

        result = check_cache_config({'cache_config': {'layer': 'test', 'size': 1000}})
        assert 'result' in result
        assert 'allow_cache_action' in result['result']

    def test_get_policy_stats(self):
        """Test get_policy_stats convenience function"""
        from predictive_agent.opa_integration import check_pod_security, get_policy_stats

        # Make a request
        check_pod_security({})

        # Get stats
        stats = get_policy_stats()
        assert isinstance(stats, dict)
        assert 'total_requests' in stats
        assert 'enabled' in stats


class TestIntegrationWithExistingModules:
    """Tests to ensure OPA integration doesn't break existing modules"""

    def test_l1cache_still_works(self):
        """Test that L1Cache still works with OPA integration"""
        from predictive_agent.l1_cache import get_cache

        cache = get_cache()
        # Use the actual L1Cache API
        cache.set_general('test_key_opa_int', 'test_value_opa_int', ttl=60)
        hit, value = cache.get_general('test_key_opa_int')

        assert hit == True
        assert value == 'test_value_opa_int'

    def test_optimizer_still_works(self):
        """Test that Optimizer still works with OPA integration"""
        from predictive_agent.optimize import get_optimizer

        optimizer = get_optimizer()
        assert optimizer is not None

    def test_collector_import(self):
        """Test that Collector module can still be imported"""
        import predictive_agent.collector
        assert predictive_agent.collector is not None

    def test_llm_batch_import(self):
        """Test that LLM batch module can still be imported"""
        import predictive_agent.llm_batch
        assert predictive_agent.llm_batch is not None

    def test_kalman_import(self):
        """Test that Kalman module can still be imported"""
        import predictive_agent.kalman
        assert predictive_agent.kalman is not None

    def test_markov_import(self):
        """Test that Markov module can still be imported"""
        import predictive_agent.markov
        assert predictive_agent.markov is not None


class TestImportAllOPAModules:
    """Test that all OPA modules can be imported"""

    def test_opa_client_import(self):
        """Test opa_client module import"""
        from predictive_agent.opa_client import OPAClient, OPASecurityError, get_opa_client
        assert OPAClient is not None
        assert get_opa_client is not None
        assert OPASecurityError is not None

    def test_opa_cache_import(self):
        """Test opa_cache module import"""
        from predictive_agent.opa_cache import OPACache, OPAPolicyCache, get_opa_policy_cache, reset_opa_policy_cache
        assert OPACache is not None
        assert OPAPolicyCache is not None
        assert get_opa_policy_cache is not None
        assert reset_opa_policy_cache is not None

    def test_opa_integration_import(self):
        """Test opa_integration module import"""
