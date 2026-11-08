"""Edge tests for predictive_agent.l1_cache.

Locks in the L1 cache contracts the rest of the system relies on:
LRU/TTL/eviction semantics, per-layer accessors, the (symmetric!)
LLM prompt cache, CPU-overload bypass and request deduplication.

The module's ``time`` source is patched (never slept) for TTL tests and
the module-global singleton is reset by an autouse fixture so tests stay
order-independent.
"""

import asyncio
import hashlib
import threading
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from predictive_agent.l1_cache import (
    L1Cache,
    LRUCache,
    get_cache,
    reset_cache,
)

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _fresh_global_cache():
    """Reset the module-level singleton before and after every test."""
    reset_cache()
    yield
    reset_cache()


# ── LRUCache: basics, TTL, eviction ─────────────────────────────────────────

def test_lru_set_get_roundtrip_and_miss():
    cache = LRUCache(max_size=10, default_ttl=300.0)
    cache.set("a", {"v": 1})
    hit, value = cache.get("a")
    assert hit is True
    assert value == {"v": 1}
    # Unknown key: miss with None value, no exception.
    hit2, value2 = cache.get("missing")
    assert hit2 is False
    assert value2 is None


def test_lru_ttl_expiry_is_honored():
    cache = LRUCache(max_size=10, default_ttl=5.0)
    cache.set("k", "v")
    hit, _ = cache.get("k")
    assert hit is True

    # Age the entry past its TTL by rewinding its creation stamp (the entry
    # captures time.time via a bound default_factory, so we age the entry
    # itself instead of patching the module clock).
    cache._cache["k"].created_at -= 6.0
    hit, value = cache.get("k")
    assert hit is False
    assert value is None
    assert cache.size() == 0  # expired entry is dropped on access


def test_lru_ttl_override_per_entry():
    cache = LRUCache(max_size=10, default_ttl=100.0)
    cache.set("short", "v", ttl=1.0)
    cache.set("long", "v")
    # Age only the short-TTL entry past its deadline.
    cache._cache["short"].created_at -= 2.0
    hit, _ = cache.get("short")
    assert hit is False  # per-entry TTL beat the default
    hit2, _ = cache.get("long")
    assert hit2 is True  # default-TTL sibling survives


def test_lru_evicts_oldest_when_full():
    cache = LRUCache(max_size=3, default_ttl=300.0)
    for i in range(4):
        cache.set(f"k{i}", i)
    assert cache.size() == 3
    assert cache.get("k0") == (False, None)  # oldest evicted
    assert cache.get("k3") == (True, 3)      # newest retained


def test_lru_get_refreshes_recency():
    cache = LRUCache(max_size=3, default_ttl=300.0)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    cache.get("a")          # touch "a" -> "b" becomes the LRU victim
    cache.set("d", 4)
    assert cache.get("a") == (True, 1)
    assert cache.get("b") == (False, None)


def test_lru_delete_returns_found_flag():
    cache = LRUCache(max_size=5)
    cache.set("x", 1)
    assert cache.delete("x") is True
    assert cache.delete("x") is False  # second delete: nothing to remove
    assert cache.get("x") == (False, None)


def test_lru_invalidate_prefix_returns_count():
    cache = LRUCache(max_size=10)
    cache.set("pod:ns1/a", 1)
    cache.set("pod:ns1/b", 2)
    cache.set("pod:ns2/a", 3)
    cache.set("other:1", 4)
    removed = cache.invalidate("pod:ns1")
    assert removed == 2
    assert cache.get("pod:ns2/a") == (True, 3)
    assert cache.get("other:1") == (True, 4)
    # Empty prefix invalidates everything.
    assert cache.invalidate("") == 2


def test_lru_stats_track_hits_and_misses():
    cache = LRUCache(max_size=5)
    cache.set("k", 1)
    cache.get("k")    # hit
    cache.get("nope")  # miss
    stats = cache.stats
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.sets == 1
    payload = stats.to_dict()
    assert payload["hits"] == 1 and payload["misses"] == 1


# ── L1Cache: namespaced layers ──────────────────────────────────────────────

def test_l1_pod_state_roundtrip_and_invalidate():
    cache = L1Cache()
    cache.set_pod_state("ns1", "web", {"phase": "Running"})
    hit, state = cache.get_pod_state("ns1", "web")
    assert hit is True
    assert state == {"phase": "Running"}

    cache.invalidate_pod("ns1", "web")
    hit2, value = cache.get_pod_state("ns1", "web")
    assert hit2 is False and value is None


def test_l1_prediction_roundtrip():
    cache = L1Cache()
    cache.set_prediction("ns1", "db", {"risk": 0.42})
    hit, prediction = cache.get_prediction("ns1", "db")
    assert hit is True
    assert prediction == {"risk": 0.42}


def test_l1_kubectl_result_and_make_key_stability():
    cache = L1Cache()
    cmd = "kubectl get pods -n ns1"
    cache.set_kubectl_result(cmd, {"items": []})
    # Contract: set() hashes the raw command internally; get() expects the
    # caller to pass that same sha256 hex digest.
    key = hashlib.sha256(cmd.encode()).hexdigest()
    hit, result = cache.get_kubectl_result(key)
    assert hit is True
    assert result == {"items": []}
    # make_key is order-independent for kwargs and stable across calls.
    k1 = L1Cache.make_key("p", "a", b=2, a=1)
    k2 = L1Cache.make_key("p", "a", a=1, b=2)
    assert k1 == k2


def test_l1_llm_prompt_cache_is_symmetric():
    """set_llm_prompt/get_llm_prompt must agree on key derivation.

    Regression: set() hashed the key while get() looked the raw key up,
    so the LLM response cache never hit. Both sides now hash identically.
    """
    cache = L1Cache()
    cache.set_llm_prompt("analyze pod cpu", {"severity": "low"})
    hit, response = cache.get_llm_prompt("analyze pod cpu")
    assert hit is True
    assert response == {"severity": "low"}
    # A different prompt must not collide.
    hit2, _ = cache.get_llm_prompt("analyze pod memory")
    assert hit2 is False


def test_l1_llm_prompt_overwrite_replaces_value():
    cache = L1Cache()
    cache.set_llm_prompt("p", {"v": 1})
    cache.set_llm_prompt("p", {"v": 2})
    hit, response = cache.get_llm_prompt("p")
    assert hit is True
    assert response == {"v": 2}


def test_l1_cpu_overload_bypasses_llm_cache():
    """Under CPU overload the cache must fail closed: no reads, no writes."""
    cache = L1Cache()
    cache.set_llm_prompt("p", {"v": 1})  # populate while healthy
    with patch.object(cache, "_is_cpu_overloaded", return_value=True):
        cache.set_llm_prompt("p", {"v": 99})  # must be ignored
        hit, value = cache.get_llm_prompt("p")  # must not serve stale data
    assert hit is False
    assert value is None
    # The original entry is still there once the overload clears.
    hit2, value2 = cache.get_llm_prompt("p")
    assert hit2 is True
    assert value2 == {"v": 1}


def test_l1_clear_all_and_get_stats():
    cache = L1Cache()
    cache.set_pod_state("ns", "web", {})
    cache.set_prediction("ns", "web", {})
    cache.set_llm_prompt("p", {})
    cache.clear_all()
    assert cache.get_pod_state("ns", "web") == (False, None)
    assert cache.get_prediction("ns", "web") == (False, None)
    assert cache.get_llm_prompt("p") == (False, None)

    stats = cache.get_stats()
    for layer in ("general", "pod_state", "prediction", "kubectl", "llm_prompt"):
        assert layer in stats["sizes"]
        assert stats["sizes"][layer] == 0


# ── Deduplication ───────────────────────────────────────────────────────────

def test_deduplicate_llm_call_collapses_concurrent_calls():
    cache = L1Cache()
    calls: List[int] = []

    async def slow_llm() -> Dict[str, Any]:
        calls.append(1)
        await asyncio.sleep(0.05)
        return {"analysis": "ok"}

    async def scenario() -> List[Any]:
        return await asyncio.gather(*[
            cache.deduplicate_llm_call("same prompt", slow_llm)
            for _ in range(3)
        ])

    results = asyncio.run(scenario())
    assert results == [{"analysis": "ok"}] * 3
    assert len(calls) == 1  # one underlying LLM call served all three
    assert cache._deduplicator.stats.deduplicated_requests == 2


def test_deduplicate_llm_call_distinct_prompts_run_separately():
    cache = L1Cache()
    calls: List[str] = []

    def make_func(tag: str):
        async def func() -> Dict[str, Any]:
            calls.append(tag)
            return {"tag": tag}
        return func

    async def scenario() -> List[Any]:
        return await asyncio.gather(*[
            cache.deduplicate_llm_call(f"prompt-{i}", make_func(f"t{i}"))
            for i in range(3)
        ])

    results = asyncio.run(scenario())
    assert len(calls) == 3  # nothing shared -> nothing deduplicated
    assert [r["tag"] for r in results] == ["t0", "t1", "t2"]


# ── Global singleton ────────────────────────────────────────────────────────

def test_get_cache_returns_same_instance_until_reset():
    first = get_cache()
    assert get_cache() is first
    reset_cache()
    assert get_cache() is not first


def test_lru_thread_safety_smoke():
    """Concurrent set/get from many threads must not corrupt state."""
    cache = LRUCache(max_size=1000, default_ttl=300.0)
    errors: List[BaseException] = []

    def worker(base: int) -> None:
        try:
            for i in range(200):
                key = f"k{base}-{i % 50}"
                cache.set(key, i)
                cache.get(key)
        except BaseException as exc:  # noqa: BLE001 - recorded, not raised
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert errors == []
    assert cache.size() <= 1000
