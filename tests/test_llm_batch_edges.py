"""Edge tests for predictive_agent.llm_batch.

Covers batching/deduplication/caching of ``LLMBatchProcessor`` and the
concurrency + failure semantics of ``ParallelLLMProcessor``. All LLM
calls go through fakes — the module never touches the network here.
"""

import asyncio
import time as _time
from typing import Any, Dict, List, Optional, Tuple

import pytest

from predictive_agent.l1_cache import reset_cache
from predictive_agent.llm_batch import (
    BatchRequest,
    BatchResult,
    LLMBatchProcessor,
    LLMCostEstimate,
    ParallelLLMProcessor,
)


@pytest.fixture(autouse=True)
def _fresh_global_cache():
    """The processors use the global L1 cache; keep tests isolated."""
    reset_cache()
    yield
    reset_cache()


class FakeAnalyzer:
    """Records analyze() calls and returns deterministic results."""

    def __init__(self, delay: float = 0.0, fail_on: Optional[str] = None) -> None:
        self.calls: List[Tuple[str, str, Optional[Dict[str, Any]]]] = []
        self.delay = delay
        self.fail_on = fail_on

    def analyze(self, issue: str, context: str,
                prediction: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.calls.append((issue, context, prediction))
        if self.fail_on and self.fail_on in issue:
            raise RuntimeError(f"LLM backend exploded on {issue}")
        if self.delay:
            _time.sleep(self.delay)
        return {"analysis": f"done:{issue}", "severity": "low", "action": "none"}


# ── Dataclasses ─────────────────────────────────────────────────────────────

def test_cost_estimate_to_dict_roundtrip():
    est = LLMCostEstimate(input_tokens=100, output_tokens=50, total_tokens=150,
                          estimated_cost=0.002, estimated_time=1.0)
    d = est.to_dict()
    assert d == {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "estimated_cost": 0.002,
        "estimated_time": 1.0,
    }


def test_batch_result_to_dict_includes_cached_flag():
    result = BatchResult(
        pod_key="ns/web",
        response={"a": 1},
        cost_estimate=LLMCostEstimate(1, 1, 2),
        processing_time=0.5,
        cached=True,
    )
    d = result.to_dict()
    assert d["pod_key"] == "ns/web"
    assert d["cached"] is True
    assert d["cost_estimate"]["total_tokens"] == 2


def test_simplify_prompt_normalizes_pod_names_and_numbers():  # noqa: C901
    simplify = BatchRequest._simplify_prompt
    # Pod names and numbers collapse to placeholders; case is normalized.
    a = simplify("Check pod web-1 cpu 85.5 percent")
    b = simplify("check pod db-2 cpu 12 percent")
    assert "85" not in a and "12" not in b
    assert a != b  # structure still differs
    # Placeholders come out lowercased (the simplifier lower()s at the end);
    # the word-regex consumes digit runs too, so no raw digits remain.
    assert "pod_name" in a
    # The digit regex is unreachable: the word-regex consumes digit runs
    # first, so numbers collapse into pod_name placeholders.
    assert not any(ch.isdigit() for ch in a)
    assert a == a.lower()


def test_batch_request_get_key_stable_for_similar_prompts():
    r1 = BatchRequest(prompt="Analyze pod web-1 at 85%", context={}, pod_key="ns/web")
    r2 = BatchRequest(prompt="Analyze pod web-2 at 90%", context={}, pod_key="ns/db")
    assert r1.get_key() == r2.get_key()  # names/numbers simplified away
    r3 = BatchRequest(prompt="Completely different task", context={}, pod_key="ns/x")
    assert r1.get_key() != r3.get_key()


# ── LLMBatchProcessor ───────────────────────────────────────────────────────

def _make_processor(analyzer: FakeAnalyzer, **kwargs: Any) -> LLMBatchProcessor:
    return LLMBatchProcessor(analyzer, **kwargs)


def test_process_all_preserves_input_order():
    analyzer = FakeAnalyzer()
    proc = _make_processor(analyzer)
    for i in range(3):
        proc.queue_analysis(f"prompt {i}", {"issue": f"iss{i}"}, f"ns/p{i}")
    results = asyncio.run(proc.process_all())
    assert [r.pod_key for r in results] == ["ns/p0", "ns/p1", "ns/p2"]
    assert all(r.cached is False for r in results)
    assert len(analyzer.calls) == 3


def test_process_batch_orders_by_priority_then_arrival():
    analyzer = FakeAnalyzer()
    proc = _make_processor(analyzer)
    t0 = _time.time()
    low = BatchRequest("p-low", {"issue": "a"}, "ns/low", priority=0, created_at=t0)
    high = BatchRequest("p-high", {"issue": "b"}, "ns/high", priority=5, created_at=t0 + 1)
    mid = BatchRequest("p-mid", {"issue": "c"}, "ns/mid", priority=1, created_at=t0 + 2)
    results = asyncio.run(proc.process_batch([low, high, mid]))
    assert [r.pod_key for r in results] == ["ns/high", "ns/mid", "ns/low"]


def test_second_identical_prompt_is_served_from_cache():
    analyzer = FakeAnalyzer()
    proc = _make_processor(analyzer)
    r1 = asyncio.run(proc.analyze_immediate("same prompt", {"issue": "x"}, "ns/a"))
    r2 = asyncio.run(proc.analyze_immediate("same prompt", {"issue": "x"}, "ns/b"))
    assert r1.cached is False and r2.cached is True
    assert r1.response == r2.response
    assert len(analyzer.calls) == 1  # only the first hit the analyzer
    assert proc.get_stats()["total_cached"] == 1


def test_cache_disabled_always_calls_analyzer():
    analyzer = FakeAnalyzer()
    proc = _make_processor(analyzer, cache_enabled=False)
    asyncio.run(proc.analyze_immediate("p", {}, "ns/a"))
    asyncio.run(proc.analyze_immediate("p", {}, "ns/a"))
    assert len(analyzer.calls) == 2
    assert proc.get_stats()["total_cached"] == 0


def test_deduplication_collapses_concurrent_same_prompt_calls():
    analyzer = FakeAnalyzer(delay=0.05)
    proc = _make_processor(analyzer, cache_enabled=False)  # isolate dedup path

    async def scenario() -> List[BatchResult]:
        return await asyncio.gather(*[
            proc.analyze_immediate("shared", {"issue": "x"}, f"ns/p{i}")
            for i in range(3)
        ])

    results = asyncio.run(scenario())
    assert len(results) == 3
    assert len(analyzer.calls) == 1  # deduplicated into one backend call


def test_deduplication_disabled_calls_analyzer_per_request():
    analyzer = FakeAnalyzer(delay=0.05)
    proc = _make_processor(analyzer, cache_enabled=False, deduplication_enabled=False)

    async def scenario() -> List[BatchResult]:
        return await asyncio.gather(*[
            proc.analyze_immediate("shared", {"issue": "x"}, f"ns/p{i}")
            for i in range(3)
        ])

    asyncio.run(scenario())
    assert len(analyzer.calls) == 3


def test_custom_token_cost_drives_cost_estimate():
    analyzer = FakeAnalyzer()
    proc = _make_processor(analyzer, token_cost_per_1k={"input": 0.01, "output": 0.03})
    result = asyncio.run(proc.analyze_immediate("abcdefgh" * 10, {}, "ns/a"))
    cost = result.cost_estimate
    assert cost.input_tokens == 20  # len(prompt) // 4
    assert cost.estimated_cost > 0
    assert cost.estimated_time > 0
    # Math check: cost = in/1000*0.01 + out/1000*0.03
    expected = (cost.input_tokens / 1000 * 0.01) + (cost.output_tokens / 1000 * 0.03)
    assert abs(cost.estimated_cost - expected) < 1e-12


def test_processor_stats_reflect_activity():
    analyzer = FakeAnalyzer()
    proc = _make_processor(analyzer)
    for i in range(5):
        proc.queue_analysis(f"p{i}", {"issue": f"i{i}"}, f"ns/{i}")
    stats_before = proc.get_stats()
    assert stats_before["queue_size"] == 5
    assert stats_before["total_requests"] == 5
    asyncio.run(proc.process_all())
    stats = proc.get_stats()
    assert stats["total_batches"] == 1
    assert stats["running"] is False


def test_start_and_stop_lifecycle():
    analyzer = FakeAnalyzer()
    proc = _make_processor(analyzer)

    async def scenario() -> None:
        proc.start(auto_flush_interval=0.05)
        assert proc.get_stats()["running"] is True
        proc.start(auto_flush_interval=0.05)  # second start is a no-op
        proc.queue_analysis("p", {"issue": "x"}, "ns/a")
        await asyncio.sleep(0.3)  # let the background loop flush
        proc.stop()
        assert proc.get_stats()["running"] is False
        proc.stop()  # second stop is a no-op

    asyncio.run(scenario())
    assert len(analyzer.calls) == 1  # background loop flushed the queue


# ── ParallelLLMProcessor ────────────────────────────────────────────────────

def test_parallel_analyze_success_then_cache_hit():
    analyzer = FakeAnalyzer()
    proc = ParallelLLMProcessor(analyzer, max_concurrent=2, rate_limit=0)

    result, cached = asyncio.run(proc.analyze("p1", {"issue": "cpu"}, "ns/a"))
    assert cached is False
    assert result["analysis"] == "done:cpu"

    result2, cached2 = asyncio.run(proc.analyze("p1", {"issue": "cpu"}, "ns/a"))
    assert cached2 is True
    assert result2 == result
    assert len(analyzer.calls) == 1

    stats = proc.get_stats()
    assert stats["total_success"] == 2  # cache hits count as success
    assert stats["total_errors"] == 0


def test_parallel_analyze_failure_returns_safe_fallback():
    analyzer = FakeAnalyzer(fail_on="doomed")
    proc = ParallelLLMProcessor(analyzer, rate_limit=0)
    result, cached = asyncio.run(proc.analyze("p", {"issue": "doomed"}, "ns/a"))
    assert cached is False
    assert result["severity"] == "high"
    assert "failed" in result["analysis"].lower()
    assert proc.get_stats()["total_errors"] == 1


def test_analyze_many_pairs_results_with_pod_keys():
    analyzer = FakeAnalyzer(delay=0.02)
    proc = ParallelLLMProcessor(analyzer, max_concurrent=3, rate_limit=0)
    requests = [(f"prompt-{i}", {"issue": f"i{i}"}, f"ns/pod{i}") for i in range(5)]
    results = asyncio.run(proc.analyze_many(requests))
    assert [r[2] for r in results] == [f"ns/pod{i}" for i in range(5)]
    assert all(r[1] is False for r in results)
    assert len(analyzer.calls) == 5


def test_parallel_analyze_cache_disabled():
    analyzer = FakeAnalyzer()
    proc = ParallelLLMProcessor(analyzer, rate_limit=0, cache_enabled=False)
    asyncio.run(proc.analyze("p", {}, "ns/a"))
    result, cached = asyncio.run(proc.analyze("p", {}, "ns/a"))
    assert cached is False
    assert len(analyzer.calls) == 2
