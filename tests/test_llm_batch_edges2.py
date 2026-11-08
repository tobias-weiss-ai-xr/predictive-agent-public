"""Round-2 edge tests for predictive_agent.llm_batch.

Covers the branches the first edge suite (``tests/test_llm_batch_edges.py``)
left untouched:

- ``LLMBatchProcessor.process_all``'s inner drain-wait loop: the
  ``QueueEmpty`` race, the empty-batch ``asyncio.sleep`` branch, and the
  ``TimeoutError`` abort path (lines 376-381).
- ``LLMBatchProcessor._background_processor``'s exception-logging branch:
  a failing ``process_all`` raises once, is logged, and the loop keeps
  running — and the woken (re-queued) request is still processed instead of
  being dropped (lines 457-460).
- ``ParallelLLMProcessor._rate_limit_request``'s wait branch when the 1s
  window is full (``rate_limit=2`` + 3 concurrent calls; total < 2s).

All LLM calls go through fakes — this module never touches the network.
"""

import asyncio
import logging
import time as _time
from typing import Any, Dict, List, Optional, Tuple

import pytest

import predictive_agent.llm_batch as llm_batch
from predictive_agent.l1_cache import reset_cache
from predictive_agent.llm_batch import LLMBatchProcessor, ParallelLLMProcessor


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


class _TrickyQueue(asyncio.Queue):
    """Simulates a drain race inside ``process_all``.

    Claims items are ready for the first few ``get_nowait()`` attempts (which
    always raise ``QueueEmpty``), then reports genuinely empty so the outer
    ``while not queue.empty()`` loop can terminate.
    """

    def __init__(self, races: int = 1) -> None:
        super().__init__()
        self._races = races
        self._attempts = 0

    def empty(self) -> bool:
        return self._attempts >= self._races

    def get_nowait(self):
        self._attempts += 1
        raise asyncio.QueueEmpty


# ── LLMBatchProcessor.process_all drain-wait loop ──────────────────────────

def test_process_all_handles_drain_race_and_empty_batch():
    """QueueEmpty race + empty-batch sleep, then a clean empty result."""
    analyzer = FakeAnalyzer()
    proc = LLMBatchProcessor(analyzer, max_wait_time=0.2)
    # The queue claims an item is ready for one get_nowait() (which raises
    # QueueEmpty), then reports empty: the inner wait loop falls into its
    # empty-batch sleep branch and eventually gives up with no work done.
    proc._queue = _TrickyQueue(races=1)

    started = _time.time()
    results = asyncio.run(proc.process_all())
    elapsed = _time.time() - started

    assert results == []
    assert elapsed >= 0.15  # had to wait out the empty-batch sleep loop
    assert len(analyzer.calls) == 0
    assert proc.get_stats()["total_batches"] == 0


def test_process_all_aborts_wait_loop_on_timeout(monkeypatch):
    """The inner wait loop's ``except TimeoutError`` swallows a timeout."""

    def _raising_sleep(*args, **kwargs):  # noqa: ARG001
        raise TimeoutError("simulated wait timeout")

    # The module only awaits asyncio.sleep in the empty-batch branch; make
    # that await explode like a wait_for(..., timeout=...) would.
    monkeypatch.setattr(llm_batch.asyncio, "sleep", _raising_sleep)

    analyzer = FakeAnalyzer()
    proc = LLMBatchProcessor(analyzer, max_wait_time=5.0)
    proc._queue = _TrickyQueue(races=1)

    results = asyncio.run(proc.process_all())

    assert results == []
    assert len(analyzer.calls) == 0
    assert proc.get_stats()["total_batches"] == 0


# ── LLMBatchProcessor._background_processor ────────────────────────────────

def test_background_processor_logs_error_and_keeps_running(caplog):
    """A failing process_all is logged; the loop survives and keeps flushing."""
    analyzer = FakeAnalyzer(fail_on="two")  # issue "two" makes the backend throw
    proc = LLMBatchProcessor(analyzer)

    async def scenario() -> List[str]:
        proc.start(auto_flush_interval=0.05)
        proc.queue_analysis("req1", {"issue": "one"}, "ns/one")
        await asyncio.sleep(0)
        # First flush: the woken request is re-queued before the drain and
        # must be processed exactly once (the no-re-queue bug dropped it).
        for _ in range(200):
            if len(analyzer.calls) >= 1:
                break
            await asyncio.sleep(0.02)
        assert len(analyzer.calls) == 1, analyzer.calls

        # This request raises inside process_all during the background loop.
        proc.queue_analysis("req2", {"issue": "two"}, "ns/two")
        for _ in range(300):
            if len(analyzer.calls) >= 2:
                break
            await asyncio.sleep(0.02)
        assert len(analyzer.calls) == 2, analyzer.calls

        # The loop survived the exception -> new work is still processed.
        proc.queue_analysis("req3", {"issue": "three"}, "ns/three")
        for _ in range(300):
            if len(analyzer.calls) >= 3:
                break
            await asyncio.sleep(0.02)
        proc.stop()
        return [c[0] for c in analyzer.calls]

    with caplog.at_level(logging.ERROR, logger="LLMBatchProcessor"):
        analyzed_issues = asyncio.run(scenario())

    assert analyzed_issues == ["one", "two", "three"]  # loop kept running
    messages = [r.getMessage() for r in caplog.records]
    assert any("Error in batch processor" in msg and "two" in msg
               for msg in messages)


def test_background_processor_processes_woken_request_exactly_once():
    """The re-queued woken request is flushed once (not dropped, not doubled)."""
    analyzer = FakeAnalyzer()
    proc = LLMBatchProcessor(analyzer)

    async def scenario() -> None:
        proc.start(auto_flush_interval=0.5)
        proc.queue_analysis("woken", {"issue": "woken"}, "ns/a")
        for _ in range(400):
            if len(analyzer.calls) >= 1:
                break
            await asyncio.sleep(0.02)
        proc.stop()

    asyncio.run(scenario())

    assert [c[0] for c in analyzer.calls] == ["woken"]
    assert proc.get_stats()["total_requests"] == 1


# ── ParallelLLMProcessor._rate_limit_request wait branch ───────────────────

def test_rate_limit_waits_when_1s_window_full():
    """Window full of recent requests -> sleep for the remainder of 1s."""
    analyzer = FakeAnalyzer()
    proc = ParallelLLMProcessor(analyzer, max_concurrent=10, rate_limit=2)
    now = _time.time()
    # Two recent requests already fill the 1s allowance window.
    proc._last_request_times = [now - 0.5, now - 0.45]

    started = _time.time()
    result, cached = asyncio.run(proc.analyze("p", {"issue": "x"}, "ns/a"))
    elapsed = _time.time() - started

    assert cached is False
    assert result["analysis"] == "done:x"
    assert elapsed >= 0.45  # waited out the rest of the 1s window
    # Oldest entry fell out of the window (age >= 1s) and was pruned.
    assert len(proc._last_request_times) == 2


def test_rate_limit_zero_never_blocks():
    """rate_limit=0 disables rate limiting entirely."""
    analyzer = FakeAnalyzer()
    proc = ParallelLLMProcessor(analyzer, rate_limit=0)

    started = _time.time()
    result, cached = asyncio.run(proc.analyze("p", {"issue": "free"}, "ns/a"))
    elapsed = _time.time() - started

    assert cached is False
    assert result["analysis"] == "done:free"
    # rate-limit window is 1s — any real wait would be >= 1s, so a bound below
    # 1s proves the request was NOT blocked. Generous margin for a loaded runner
    # (the old < 0.2 flaked at ~0.28s on a busy mac).
    assert elapsed < 0.9
    assert proc._last_request_times == []  # nothing recorded


def test_rate_limit_2_with_three_concurrent_calls_under_2s():
    """rate_limit=2 + 3 concurrent calls -> third waits, all succeed, < 2s."""
    analyzer = FakeAnalyzer()
    proc = ParallelLLMProcessor(analyzer, max_concurrent=3, rate_limit=2)

    async def scenario() -> List[Tuple[Dict[str, Any], bool]]:
        return await asyncio.gather(*[
            proc.analyze(f"rl-{i}", {"issue": f"i{i}"}, f"ns/p{i}")
            for i in range(3)
        ])

    started = _time.time()
    results = asyncio.run(scenario())
    elapsed = _time.time() - started

    assert len(results) == 3
    assert all(cached is False for _, cached in results)
    stats = proc.get_stats()
    assert stats["total_requests"] == 3
    assert stats["total_success"] == 3
    assert stats["total_errors"] == 0
    assert elapsed < 2.0  # hard cap: the rate-limit wait is ~1s
    assert elapsed >= 0.8  # at least one call hit the full 1s window


def test_rate_limit_ignores_entries_outside_1s_window():
    """Entries older than the 1s window never block the next request."""
    analyzer = FakeAnalyzer()
    proc = ParallelLLMProcessor(analyzer, max_concurrent=10, rate_limit=2)
    now = _time.time()
    # Both recorded timestamps already fell out of the 1s window.
    proc._last_request_times = [now - 2.0, now - 1.5]

    started = _time.time()
    result, cached = asyncio.run(proc.analyze("p", {"issue": "x"}, "ns/a"))
    elapsed = _time.time() - started

    assert cached is False
    # window is 1s => a real wait would be >= 1s; < 0.9 proves no sleep while
    # leaving margin for a loaded runner (old < 0.2 flaked at ~0.25s on mac).
    assert elapsed < 0.9  # no sleep needed
    assert len(proc._last_request_times) == 1  # only the fresh one kept
