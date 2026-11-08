"""Edge tests (round 2) for predictive_agent.l1_cache.

Covers the failure and memory paths not locked in by
``tests/test_l1_cache_edges.py``:

- memory-capped eviction (``_evict_by_memory``) with tiny caps and large
  values, plus ``memory_usage()`` accounting
- size-estimation fallback for non-serializable values
- ``_evict_if_needed`` interplay with TTL expiry on write
- ``make_key`` FNV fallback to SHA-256 when ``_fast_hash`` raises
- ``warm_cache_with_pod_list`` placeholder semantics
- ``RequestDeduplicator`` error propagation to concurrent waiters
  (the ``finally`` cleanup branch that runs when async_func raises)
- ``cpu_skips`` stat counters for every cache layer under CPU overload

TTL tests age entries by rewinding ``entry.created_at`` (no clock patching,
no sleeps). The module-global singleton is reset by an autouse fixture so
tests stay order-independent.
"""

import asyncio
import hashlib
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

import predictive_agent.l1_cache as _l1_cache_mod
from predictive_agent.l1_cache import (
    L1Cache,
    LRUCache,
    RequestDeduplicator,
    reset_cache,
)

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _fresh_global_cache():
    """Reset the module-level singleton before and after every test."""
    reset_cache()
    yield
    reset_cache()


# ── Memory-capped eviction ──────────────────────────────────────────────────

def test_memory_cap_evicts_oldest_entries():
    """With a 1MB cap and over-cap entries, the oldest entries must be
    evicted (one batch per call with a small max_size) before adding."""
    cache = LRUCache(max_size=10, max_memory_mb=1)  # batch = max(1, 10//10)
    big = "x" * 600_000  # ~1.2MB serialized > 1MB cap
    cache.set("a", big)
    cache.set("b", big)  # forces _evict_by_memory -> evicts "a"
    assert cache.size() == 1
    assert cache.get("a") == (False, None)  # oldest evicted by memory
    assert cache.get("b") == (True, big)
    assert cache.stats.evictions == 1
    assert cache.stats.memory_evictions == 1


def test_memory_cap_within_limit_keeps_entries():
    """Small values under the memory cap must never be evicted."""
    cache = LRUCache(max_size=10, max_memory_mb=50)
    for i in range(6):
        cache.set(f"k{i}", {"v": i})
    assert cache.size() == 6
    for i in range(6):
        assert cache.get(f"k{i}") == (True, {"v": i})
    assert cache.stats.memory_evictions == 0


def test_memory_evictions_tracked_in_stats_dict():
    """memory_evictions must flow through stats.to_dict()."""
    cache = LRUCache(max_size=5, max_memory_mb=1)
    big = "y" * 600_000
    cache.set("a", big)
    cache.set("b", big)
    payload = cache.stats.to_dict()
    assert payload["memory_evictions"] == 1
    assert payload["evictions"] == 1


def test_memory_usage_reports_current_and_max():
    """memory_usage() must return (current_bytes, max_bytes)."""
    cache = LRUCache(max_size=10, max_memory_mb=2)
    current, maximum = cache.memory_usage()
    assert current == 0
    assert maximum == 2 * 1024 * 1024

    big = "x" * 600_000
    cache.set("a", big)
    current, maximum = cache.memory_usage()
    assert current > 0
    assert maximum == 2 * 1024 * 1024


def test_memory_usage_updates_after_eviction():
    """After a memory-triggered eviction, current memory must shrink to only
    the surviving entries."""
    cache = LRUCache(max_size=10, max_memory_mb=1)
    big = "x" * 600_000
    cache.set("a", big)
    cache.set("b", big)  # evicts "a"
    current, _ = cache.memory_usage()
    assert current == cache._estimate_size(big)  # only "b" remains
    assert cache.get("a") == (False, None)


def test_estimate_size_falls_back_for_non_serializable():
    """Non-JSON-serializable values get the 1024-byte fallback size."""
    class NotSerializable:
        pass

    cache = LRUCache(max_size=10)
    cache.set("k", NotSerializable())  # must not raise
    assert cache._cache["k"].size_bytes == 1024
    current, _ = cache.memory_usage()
    assert current == 1024
    hit, value = cache.get("k")
    assert hit is True
    assert isinstance(value, NotSerializable)
    cache.delete("k")
    assert cache.memory_usage() == (0, 50 * 1024 * 1024)
    assert cache.get("k") == (False, None)


# ── _evict_if_needed interplay ──────────────────────────────────────────────

def test_set_evicts_expired_entries_first():
    """set() must purge expired entries during _evict_if_needed."""
    cache = LRUCache(max_size=10, default_ttl=5.0)
    cache.set("a", 1)
    cache.set("b", 2)
    cache._cache["a"].created_at -= 10.0  # expired
    cache.set("c", 3)
    assert cache.get("a") == (False, None)  # expired and dropped
    assert cache.get("b") == (True, 2)
    assert cache.get("c") == (True, 3)
    assert cache.size() == 2
    assert cache.stats.evictions == 1


def test_set_when_full_evicts_lru_and_tracks_evictions():
    """A write to a full cache evicts the LRU entry and counts it."""
    cache = LRUCache(max_size=2, default_ttl=300.0)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)  # over capacity -> evict "a"
    assert cache.size() == 2
    assert cache.get("a") == (False, None)
    assert cache.get("c") == (True, 3)
    assert cache.stats.evictions == 1


# ── make_key FNV fallback ───────────────────────────────────────────────────

def test_make_key_falls_back_to_sha256_when_fnv_raises(monkeypatch):
    """If _fast_hash raises, make_key must fall back to SHA-256 instead of
    propagating the error."""

    def boom(text: str) -> str:
        raise RuntimeError("FNV unavailable")

    monkeypatch.setattr(LRUCache, "_fast_hash", boom)
    key = L1Cache.make_key("p", "a", b=2, a=1)
    expected = hashlib.sha256(b"p|a|a=1|b=2").hexdigest()
    assert key == expected


def test_make_key_switches_hash_without_changing_shape(monkeypatch):
    """The fallback key is deterministic and repeatable per call."""

    def boom(text: str) -> str:
        raise RuntimeError("FNV unavailable")

    monkeypatch.setattr(LRUCache, "_fast_hash", boom)
    k1 = L1Cache.make_key("pod_state:", "ns1", "web")
    k2 = L1Cache.make_key("pod_state:", "ns1", "web")
    assert k1 == k2
    assert isinstance(k1, str) and len(k1) == 64  # sha256 hex digest


def test_fast_hash_is_consistent_and_distinct():
    """Direct check on the FNV path used by make_key."""
    h1 = LRUCache._fast_hash("same input")
    h2 = LRUCache._fast_hash("same input")
    assert h1 == h2
    assert LRUCache._fast_hash("other input") != h1


# ── warm_cache_with_pod_list ────────────────────────────────────────────────

def test_warm_cache_with_pod_list_places_placeholders():
    """Valid ns/name pods get placeholder entries; malformed ones are skipped."""
    cache = L1Cache()
    cache.warm_cache_with_pod_list(["ns1/web", "ns1/db", "malformed-without-slash", ""])
    # valid entries are present with a None placeholder value
    hit, value = cache.get_pod_state("ns1", "web")
    assert hit is True and value is None
    hit2, value2 = cache.get_pod_state("ns1", "db")
    assert hit2 is True and value2 is None
    # malformed entries were skipped entirely
    assert cache._pod_state_cache.size() == 2


def test_warm_cache_placeholders_have_short_ttl():
    """Placeholder entries use the 10s TTL and must expire quickly."""
    cache = L1Cache()
    cache.warm_cache_with_pod_list(["ns1/web"])
    assert cache.get_pod_state("ns1", "web") == (True, None)

    key = L1Cache.make_key(L1Cache.PREFIX_POD_STATE, "ns1", "web")
    cache._pod_state_cache._cache[key].created_at -= 11.0
    assert cache.get_pod_state("ns1", "web") == (False, None)


# ── RequestDeduplicator: error propagation ──────────────────────────────────

def test_deduplicate_propagates_error_to_all_waiters():
    """When async_func raises, the owner and every concurrent waiter must
    receive the exception, exactly one underlying call happens, and the
    in-progress slot is cleaned up afterward."""
    dedup = RequestDeduplicator()
    calls: List[int] = []

    async def failing() -> None:
        calls.append(1)
        raise ValueError("llm down")

    async def scenario() -> List[Any]:
        return await asyncio.gather(
            dedup.deduplicate("k", failing),
            dedup.deduplicate("k", failing),
            dedup.deduplicate("k", failing),
            return_exceptions=True,
        )

    results = asyncio.run(scenario())
    assert [type(r) for r in results] == [ValueError, ValueError, ValueError]
    assert len(calls) == 1  # one underlying call served all three waiters
    assert dedup.stats.deduplicated_requests == 2
    assert dedup._in_progress == {}  # finally-cleanup ran


def test_deduplicate_cleans_up_and_allows_retry_after_failure():
    """After an exception, the same key can be used again (fresh run)."""
    dedup = RequestDeduplicator()
    calls: List[int] = []

    async def failing() -> None:
        calls.append(1)
        raise RuntimeError("boom")

    async def run() -> None:
        with pytest.raises(RuntimeError):
            await dedup.deduplicate("k", failing)

    asyncio.run(run())
    assert dedup._in_progress == {}
    assert len(calls) == 1
    # Retry must execute fresh: the in-progress slot was fully released.
    asyncio.run(run())
    assert len(calls) == 2


def test_deduplicate_owner_exception_still_cleans_up():
    """A lone caller whose function raises must not leak the in-progress slot."""
    dedup = RequestDeduplicator()

    async def failing() -> None:
        raise OSError("connection reset")

    async def run() -> None:
        with pytest.raises(OSError):
            await dedup.deduplicate("k", failing)

    asyncio.run(run())
    assert dedup._in_progress == {}
    assert dedup.stats.deduplicated_requests == 0


# ── cpu_skips stat counters ─────────────────────────────────────────────────

# These tests monkeypatch l1.psutil, which only exists when psutil is
# installed (the module degrades gracefully without it).
_requires_psutil = pytest.mark.skipif(
    not _l1_cache_mod.CPU_CHECK_AVAILABLE,
    reason="psutil not installed; l1_cache runs without host-metric checks",
)


@_requires_psutil
def test_cpu_overload_detection_crosses_threshold(monkeypatch):
    """The real _is_cpu_overloaded measurement body: a high psutil reading flips
    the flag and logs a warning."""
    import predictive_agent.l1_cache as l1

    monkeypatch.setattr(l1.psutil, "cpu_percent", lambda interval=0.1: 99.0)
    cache = L1Cache(cpu_threshold=95.0)
    assert cache._is_cpu_overloaded() is True
    # Second call inside the check interval reuses the cached decision.
    assert cache._is_cpu_overloaded() is True
    assert cache._cpu_overloaded is True
    assert cache._last_cpu_check > 0


@_requires_psutil
def test_cpu_overload_detection_below_threshold(monkeypatch):
    """A normal psutil reading must not flip the overload flag."""
    import predictive_agent.l1_cache as l1

    monkeypatch.setattr(l1.psutil, "cpu_percent", lambda interval=0.1: 10.0)
    cache = L1Cache(cpu_threshold=95.0)
    assert cache._is_cpu_overloaded() is False
    assert cache._cpu_overloaded is False
    # threshold <= 0 permanently disables CPU checking
    cache2 = L1Cache(cpu_threshold=0)
    assert cache2._is_cpu_overloaded() is False


@_requires_psutil
def test_cpu_overload_detection_swallows_psutil_errors(monkeypatch):
    """If psutil measurement raises, _is_cpu_overloaded must fail closed."""
    import predictive_agent.l1_cache as l1

    def boom(interval: float = 0.1) -> float:
        raise OSError("procfs unavailable")

    monkeypatch.setattr(l1.psutil, "cpu_percent", boom)
    cache = L1Cache(cpu_threshold=95.0)
    assert cache._is_cpu_overloaded() is False


def test_deduplicate_success_returns_value_and_cleans_up():
    """The owner's success path must return the value and still release the
    in-progress slot via the finally block."""
    dedup = RequestDeduplicator()
    calls: List[int] = []

    async def work() -> Dict[str, Any]:
        calls.append(1)
        return {"ok": True}

    async def run() -> Any:
        # non-deduplicated direct success path
        return await dedup.deduplicate("k", work)

    assert asyncio.run(run()) == {"ok": True}
    assert dedup._in_progress == {}
    assert len(calls) == 1
    assert dedup.stats.deduplicated_requests == 0


def test_general_cache_cpu_overload_increments_cpu_skips():
    """_check_cpu_and_get must bump cpu_skips (general layer)."""
    cache = L1Cache()
    cache.set_general("k", 1)
    with patch.object(cache, "_is_cpu_overloaded", return_value=True):
        hit, value = cache.get_general("k")
    assert (hit, value) == (False, None)
    assert cache._general_cache._stats.cpu_skips == 1
    assert cache.get_stats()["general"]["cpu_skips"] == 1


def test_kubectl_cache_cpu_overload_skips_read_and_write():
    """Under CPU overload kubectl cache must fail closed: no reads, no stale
    writes, and cpu_skips counts the skipped read."""
    cache = L1Cache()
    cmd = "kubectl get pods -n ns1"
    cache.set_kubectl_result(cmd, {"items": []})  # populate while healthy
    key = hashlib.sha256(cmd.encode()).hexdigest()

    with patch.object(cache, "_is_cpu_overloaded", return_value=True):
        cache.set_kubectl_result(cmd, {"items": ["stale"]})  # ignored
        hit, value = cache.get_kubectl_result(key)            # skipped
    assert (hit, value) == (False, None)
    assert cache._kubectl_cache._stats.cpu_skips == 1

    # Once healthy again, the original (pre-overload) data is still served.
    hit2, value2 = cache.get_kubectl_result(key)
    assert hit2 is True
    assert value2 == {"items": []}


def test_pod_state_cpu_overload_increments_cpu_skips():
    cache = L1Cache()
    with patch.object(cache, "_is_cpu_overloaded", return_value=True):
        assert cache.get_pod_state("ns", "web") == (False, None)
    assert cache._pod_state_cache._stats.cpu_skips == 1
    assert cache.get_stats()["pod_state"]["cpu_skips"] == 1


def test_prediction_cpu_overload_increments_cpu_skips():
    cache = L1Cache()
    with patch.object(cache, "_is_cpu_overloaded", return_value=True):
        assert cache.get_prediction("ns", "db") == (False, None)
    assert cache._prediction_cache._stats.cpu_skips == 1
    assert cache.get_stats()["prediction"]["cpu_skips"] == 1


def test_llm_prompt_cpu_overload_increments_cpu_skips():
    cache = L1Cache()
    with patch.object(cache, "_is_cpu_overloaded", return_value=True):
        assert cache.get_llm_prompt("analyze pod") == (False, None)
    assert cache._llm_prompt_cache._stats.cpu_skips == 1
    assert cache.get_stats()["llm_prompt"]["cpu_skips"] == 1


def test_set_writes_dropped_under_cpu_overload_except_general():
    """Under CPU overload, layer-set writes are dropped silently — except
    set_general, which is not CPU-guarded (observed asymmetry vs. the other
    layers and vs. get_general which is CPU-guarded)."""
    cache = L1Cache()
    with patch.object(cache, "_is_cpu_overloaded", return_value=True):
        cache.set_pod_state("ns", "web", {"phase": "Running"})
        cache.set_prediction("ns", "db", {"risk": 0.5})
        cache.set_general("g", 1)
        cache.set_llm_prompt("p", {"v": 1})
        cache.set_kubectl_result("cmd", {"r": 1})
    # guarded layers dropped their writes entirely
    assert cache._pod_state_cache.size() == 0
    assert cache._prediction_cache.size() == 0
    assert cache._llm_prompt_cache.size() == 0
    assert cache._kubectl_cache.size() == 0
    # set_general is not CPU-guarded, so its write landed
    assert cache._general_cache.size() == 1
    assert cache.get_general("g") == (True, 1)
    # writes never count as reads; cpu_skips stays 0 on every layer
    assert cache._pod_state_cache._stats.cpu_skips == 0
    assert cache._general_cache._stats.cpu_skips == 0
