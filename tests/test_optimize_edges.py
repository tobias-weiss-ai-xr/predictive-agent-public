"""Edge tests for the performance-optimization facade (predictive_agent.optimize).

Covers:
1. get_optimizer() singleton identity and reconfiguration
2. enable/disable toggles that change the behavior of wrapped operations
3. L1 cache / batching / parallel / simple LLM integration with fakes (no real LLM calls)
4. Monitor start/stop lifecycle behavior
5. Reset-to-defaults path (reconfigure back to OptimizationConfig defaults)
6. demo_optimizations importability (no heavy side effects on import)

Only stdlib + pytest are used. Every test restores module globals (the optimizer
singleton and the L1 cache) via the ``optimizer`` fixture — never relies on test order.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

import predictive_agent.demo_optimizations as demo_optimizations
from predictive_agent import collector as collector_module
from predictive_agent.l1_cache import get_cache, reset_cache
from predictive_agent.llm_batch import LLMBatchProcessor, ParallelLLMProcessor
from predictive_agent.optimize import (
    OptimizationConfig,
    PerformanceOptimizer,
    _SimpleLLMProcessor,
    configure_optimizations,
    get_optimizer,
    get_performance_stats,
)


class FakeAnalyzer:
    """Dumb LLM analyzer stand-in; never touches the network."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, str, Optional[Dict[str, Any]]]] = []

    def analyze(
        self,
        issue: str,
        context: str,
        prediction: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record the call and return a deterministic fake result."""
        self.calls.append((issue, context, prediction))
        return {
            "analysis": f"analyzed:{issue}",
            "severity": "medium",
            "action": "Monitor",
            "command": "",
        }


@pytest.fixture()
def optimizer() -> PerformanceOptimizer:
    """Provide a pristine singleton and restore all module globals afterwards."""
    PerformanceOptimizer._instance = None
    PerformanceOptimizer._lock = None
    collector_module.clear_kubectl_cache()
    reset_cache()
    opt = PerformanceOptimizer()
    yield opt
    # Restore globals so later tests (and other files) start from a clean slate.
    PerformanceOptimizer._instance = None
    PerformanceOptimizer._lock = None
    collector_module.clear_kubectl_cache()
    reset_cache()


# --------------------------------------------------------------------------- #
# 1. Singleton identity + reconfiguration
# --------------------------------------------------------------------------- #

def test_get_optimizer_singleton_identity() -> None:
    """get_optimizer(), PerformanceOptimizer() and configure_optimizations() all
    return the same singleton instance."""
    first = get_optimizer()
    second = get_optimizer()
    assert first is second
    assert isinstance(first, PerformanceOptimizer)
    assert PerformanceOptimizer() is first
    reconfigured = configure_optimizations(llm_max_concurrent=7)
    assert reconfigured is first
    assert first.config.llm_max_concurrent == 7


def test_configure_known_key_updates_config(optimizer: PerformanceOptimizer) -> None:
    """configure() with a real knob mutates the singleton's shared config."""
    assert optimizer.config.kubectl_cache_ttl == 30.0
    optimizer.configure(kubectl_cache_ttl=7.5)
    assert optimizer.config.kubectl_cache_ttl == 7.5
    optimizer.configure(llm_max_concurrent=9)
    assert optimizer.config.llm_max_concurrent == 9


def test_configure_unknown_key_ignored(optimizer: PerformanceOptimizer) -> None:
    """configure() with a bogus knob neither sets it nor touches existing ones."""
    before_ttl = optimizer.config.kubectl_cache_ttl
    optimizer.configure(no_such_knob=123)
    assert not hasattr(optimizer.config, "no_such_knob")
    assert optimizer.config.kubectl_cache_ttl == before_ttl


def test_optimization_config_defaults() -> None:
    """The canonical defaults act as the baseline / reset target."""
    cfg = OptimizationConfig()
    assert cfg.kubectl_cache_enabled is True
    assert cfg.kubectl_cache_ttl == 30.0
    assert cfg.l1_cache_enabled is True
    assert cfg.llm_batching_enabled is True
    assert cfg.llm_parallel_enabled is True
    assert cfg.llm_max_concurrent == 3
    assert cfg.llm_rate_limit == 10.0


# --------------------------------------------------------------------------- #
# 2. Enable/disable toggles change wrapped behaviour
# --------------------------------------------------------------------------- #

def test_kubectl_cache_toggle_affects_collect_with_cache(
    optimizer: PerformanceOptimizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabling kubectl caching strips use_cache before it reaches run_cmd."""
    seen: List[Dict[str, Any]] = []

    def fake_run_cmd(cmd: Any, timeout: int = 30, use_cache: bool = True,
                     cache_ttl: Optional[float] = None) -> Tuple[int, str, str]:
        seen.append({"cmd": cmd, "use_cache": use_cache, "cache_ttl": cache_ttl})
        return 0, "out", ""

    monkeypatch.setattr(collector_module, "run_cmd", fake_run_cmd)

    # Enabled (default) -> use_cache forwarded as-is.
    (rc, out, err), cached = optimizer.collect_with_cache(
        ["kubectl", "get", "pods"], use_cache=True, cache_ttl=10.0
    )
    assert rc == 0
    assert cached is True
    assert seen[-1]["use_cache"] is True
    assert seen[-1]["cache_ttl"] == 10.0

    # Disabled -> cache flag stripped before the wrapped operation.
    optimizer.configure(kubectl_cache_enabled=False)
    (rc, out, err), cached = optimizer.collect_with_cache(
        ["kubectl", "get", "pods"], use_cache=True, cache_ttl=10.0
    )
    assert rc == 0
    assert cached is False
    assert seen[-1]["use_cache"] is False

    # Re-enabled -> behaviour restored.
    optimizer.configure(kubectl_cache_enabled=True)
    (rc, out, err), cached = optimizer.collect_with_cache(
        ["kubectl", "get", "pods"], use_cache=True, cache_ttl=10.0
    )
    assert cached is True
    assert seen[-1]["use_cache"] is True


def test_collect_with_cache_primes_and_hits_kubectl_cache(
    optimizer: PerformanceOptimizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The facade plumbs into the real kubectl cache (miss then hit)."""
    def fake_subprocess(cmd: Any, **kwargs: Any) -> MagicMock:
        result = MagicMock()
        result.returncode = 0
        result.stdout = "pod ready"
        result.stderr = ""
        return result

    monkeypatch.setattr("subprocess.run", fake_subprocess)
    collector_module.clear_kubectl_cache()

    (rc1, out1, err1), cached1 = optimizer.collect_with_cache(
        ["kubectl", "get", "pods", "-A"], use_cache=True
    )
    assert rc1 == 0
    assert cached1 is True

    (rc2, out2, err2), cached2 = optimizer.collect_with_cache(
        ["kubectl", "get", "pods", "-A"], use_cache=True
    )
    assert rc2 == 0
    assert cached2 is True
    assert out2 == "pod ready"

    stats = collector_module.get_kubectl_cache_stats()
    assert stats["misses"] == 1
    assert stats["hits"] >= 1
    assert stats["size"] >= 1


def test_clear_caches_resets_kubectl_and_l1(
    optimizer: PerformanceOptimizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """clear_caches() wipes both the kubectl cache and the L1 cache."""
    def fake_subprocess(cmd: Any, **kwargs: Any) -> MagicMock:
        result = MagicMock()
        result.returncode = 0
        result.stdout = "x"
        result.stderr = ""
        return result

    monkeypatch.setattr("subprocess.run", fake_subprocess)

    cache = get_cache()
    cache.set_pod_state("ns", "pod", {"status": "running"})
    collector_module.run_cmd(["kubectl", "get", "pods"], use_cache=True, cache_ttl=300)
    assert collector_module.get_kubectl_cache_stats()["size"] >= 1

    optimizer.clear_caches()

    assert collector_module.get_kubectl_cache_stats()["size"] == 0
    assert collector_module.get_kubectl_cache_stats()["hits"] == 0
    hit, state = get_cache().get_pod_state("ns", "pod")
    assert hit is False
    assert state is None


# --------------------------------------------------------------------------- #
# 3. L1 cache + LLM integration points with fakes
# --------------------------------------------------------------------------- #

def test_get_l1_cache_returns_global_instance(optimizer: PerformanceOptimizer) -> None:
    """optimizer.get_l1_cache() is the process-global L1 cache."""
    assert optimizer.get_l1_cache() is get_cache()
    optimizer.get_l1_cache().set_pod_state("ns", "pod", {"status": "running"})
    hit, state = get_cache().get_pod_state("ns", "pod")
    assert hit is True
    assert state["status"] == "running"


def test_get_optimized_collector_returns_collector_module(
    optimizer: PerformanceOptimizer,
) -> None:
    """The optimized collector is the collector module itself."""
    assert optimizer.get_optimized_collector() is collector_module


def test_llm_processor_selection_parallel(optimizer: PerformanceOptimizer) -> None:
    """Default config (parallel enabled) selects the ParallelLLMProcessor."""
    processor = optimizer.get_llm_processor(FakeAnalyzer())
    assert isinstance(processor, ParallelLLMProcessor)


def test_llm_processor_selection_batch(optimizer: PerformanceOptimizer) -> None:
    """Parallel off + batching on selects the LLMBatchProcessor."""
    optimizer.configure(llm_parallel_enabled=False, llm_batching_enabled=True)
    processor = optimizer.get_llm_processor(FakeAnalyzer())
    assert isinstance(processor, LLMBatchProcessor)


def test_llm_processor_selection_simple(optimizer: PerformanceOptimizer) -> None:
    """Both batching and parallel off selects the plain wrapper."""
    optimizer.configure(llm_parallel_enabled=False, llm_batching_enabled=False)
    processor = optimizer.get_llm_processor(FakeAnalyzer())
    assert isinstance(processor, _SimpleLLMProcessor)


def test_batch_processor_integration_with_fake_analyzer(
    optimizer: PerformanceOptimizer,
) -> None:
    """Batching runs distinct prompts once and serves repeats from cache."""
    reset_cache()
    analyzer = FakeAnalyzer()
    processor = optimizer.get_batch_processor(analyzer)

    processor.queue_analysis(
        "Analyze pod cpu", {"issue": "CPU high", "context": "ns/pod-1"}, "ns/pod-1"
    )
    processor.queue_analysis(
        "Analyze pod mem", {"issue": "Memory high", "context": "ns/pod-2"}, "ns/pod-2"
    )
    results = asyncio.run(processor.process_all())
    assert len(results) == 2
    assert len(analyzer.calls) == 2

    # Same prompt again -> served from the L1 cache, no new LLM call.
    processor.queue_analysis(
        "Analyze pod cpu", {"issue": "CPU high", "context": "ns/pod-1"}, "ns/pod-1"
    )
    results2 = asyncio.run(processor.process_all())
    assert len(results2) == 1
    assert results2[0].cached is True
    assert len(analyzer.calls) == 2

    # Stats reflect the requests.
    stats = processor.get_stats()
    assert stats["total_requests"] == 3
    assert stats["total_cached"] >= 1


def test_parallel_processor_integration_with_fake_analyzer(
    optimizer: PerformanceOptimizer,
) -> None:
    """Parallel processor dedupes identical prompts via the L1 cache."""
    reset_cache()
    analyzer = FakeAnalyzer()
    processor = optimizer.get_parallel_processor(analyzer)

    async def scenario() -> Tuple[Dict[str, Any], bool, Dict[str, Any], bool,
                                  Dict[str, Any], bool]:
        r1, c1 = await processor.analyze(
            "prompt-a", {"issue": "CPU", "context": "ns/pod-1"}, "ns/pod-1"
        )
        r2, c2 = await processor.analyze(
            "prompt-b", {"issue": "Mem", "context": "ns/pod-2"}, "ns/pod-2"
        )
        # Identical prompt to the first -> cache hit, no new LLM call.
        r3, c3 = await processor.analyze(
            "prompt-a", {"issue": "CPU", "context": "ns/pod-1"}, "ns/pod-1"
        )
        return r1, c1, r2, c2, r3, c3

    r1, c1, r2, c2, r3, c3 = asyncio.run(scenario())

    assert c1 is False and c2 is False and c3 is True
    assert r1 == r3
    assert len(analyzer.calls) == 2

    stats = processor.get_stats()
    assert stats["total_requests"] == 3  # every analyze() call counts
    assert stats["total_success"] == 3  # cache hits count as successes too


def test_simple_llm_processor_integration_with_fake_analyzer(
    optimizer: PerformanceOptimizer,
) -> None:
    """The simple processor caches per pod_key through the L1 cache."""
    optimizer.configure(llm_parallel_enabled=False, llm_batching_enabled=False)
    reset_cache()
    analyzer = FakeAnalyzer()
    processor = optimizer.get_llm_processor(analyzer)
    assert isinstance(processor, _SimpleLLMProcessor)

    async def scenario() -> Tuple[Dict[str, Any], bool, Dict[str, Any], bool]:
        r1, c1 = await processor.analyze(
            "prompt", {"issue": "CPU", "context": "some-context"}, "ns/pod-1"
        )
        r2, c2 = await processor.analyze(
            "prompt", {"issue": "CPU", "context": "some-context"}, "ns/pod-1"
        )
        return r1, c1, r2, c2

    r1, c1, r2, c2 = asyncio.run(scenario())
    assert c1 is False
    assert c2 is True
    assert r1 == r2
    assert len(analyzer.calls) == 1


# --------------------------------------------------------------------------- #
# 4. Lifecycle / monitoring behavior
# --------------------------------------------------------------------------- #

def test_monitor_start_stop_lifecycle(optimizer: PerformanceOptimizer) -> None:
    """start_monitoring/stop_monitoring are idempotent and clean up the task."""
    assert optimizer._running is False

    async def scenario() -> None:
        optimizer.start_monitoring()
        assert optimizer._running is True
        assert optimizer._monitoring_task is not None
        first_task = optimizer._monitoring_task

        # Second start is a no-op (same task, no duplicate).
        optimizer.start_monitoring()
        assert optimizer._monitoring_task is first_task

        optimizer.stop_monitoring()
        assert optimizer._running is False
        assert optimizer._monitoring_task is None

        # Stop when not running is a safe no-op.
        optimizer.stop_monitoring()

    asyncio.run(scenario())


# --------------------------------------------------------------------------- #
# 5. Reset-to-defaults path + stats
# --------------------------------------------------------------------------- #

def test_reset_to_defaults_path(optimizer: PerformanceOptimizer) -> None:
    """Knobs flipped away from defaults can be restored via configure()."""
    optimizer.configure(
        kubectl_cache_enabled=False,
        l1_cache_enabled=False,
        llm_batching_enabled=False,
        llm_parallel_enabled=True,
        kubectl_cache_ttl=5.0,
        llm_max_concurrent=1,
    )
    assert optimizer.config.kubectl_cache_enabled is False
    assert optimizer.config.kubectl_cache_ttl == 5.0

    # "Reset" == reconfigure every knob back to the canonical defaults.
    defaults = OptimizationConfig()
    optimizer.configure(
        kubectl_cache_enabled=defaults.kubectl_cache_enabled,
        l1_cache_enabled=defaults.l1_cache_enabled,
        llm_batching_enabled=defaults.llm_batching_enabled,
        llm_parallel_enabled=defaults.llm_parallel_enabled,
        kubectl_cache_ttl=defaults.kubectl_cache_ttl,
        llm_max_concurrent=defaults.llm_max_concurrent,
    )
    assert optimizer.config.kubectl_cache_enabled is True
    assert optimizer.config.kubectl_cache_ttl == 30.0
    assert optimizer.config.llm_batching_enabled is True
    assert optimizer.config.llm_parallel_enabled is True
    assert optimizer.config.llm_max_concurrent == 3


def test_collector_calls_stat_tracking(
    optimizer: PerformanceOptimizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """collect_with_cache() increments the collector_calls counter."""
    def fake_run_cmd(cmd: Any, timeout: int = 30, use_cache: bool = True,
                     cache_ttl: Optional[float] = None) -> Tuple[int, str, str]:
        return 0, "out", ""

    monkeypatch.setattr(collector_module, "run_cmd", fake_run_cmd)
    before = optimizer.get_stats().collector_calls
    optimizer.collect_with_cache(["kubectl", "get", "pods"])
    assert optimizer.get_stats().collector_calls == before + 1


def test_get_summary_structure(optimizer: PerformanceOptimizer) -> None:
    """get_summary() reports every optimization flag and savings/stat blocks."""
    summary = optimizer.get_summary()
    assert "optimizations" in summary
    for flag in ("kubectl_caching", "l1_cache", "llm_batching", "llm_parallel"):
        assert flag in summary["optimizations"]
        assert summary["optimizations"][flag] is True
    assert "kubectl_cache_hit_rate" in summary["savings"]
    assert "llm_cache_hit_rate" in summary["savings"]
    assert "estimated_llm_cost" in summary["savings"]
    assert "stats" in summary


def test_get_performance_stats_global(optimizer: PerformanceOptimizer) -> None:
    """Global get_performance_stats() mirrors the singleton's summary."""
    stats = get_performance_stats()
    assert "optimizations" in stats
    assert "savings" in stats
    assert "stats" in stats


# --------------------------------------------------------------------------- #
# 6. demo_optimizations importability (no heavy work on import)
# --------------------------------------------------------------------------- #

def test_demo_optimizations_importable_without_side_effects() -> None:
    """The demo module imports cleanly; demos only run under __main__."""
    assert hasattr(demo_optimizations, "PerformanceOptimizer")
    assert callable(demo_optimizations.main)
    for name in ("demo_l1_cache", "demo_kubectl_caching",
                 "demo_llm_batching", "demo_llm_parallel"):
        assert callable(getattr(demo_optimizations, name))


def test_get_summary_and_print_report_savings_math(monkeypatch, capsys):
    """get_summary/print_summary compute hit-rate savings from cache stats.

    Kubectl 3/4 and LLM 1/4 → 75.0% / 25.0% in the savings block; the
    printed report must contain the same numbers (the print path was
    previously uncovered).
    """
    from types import SimpleNamespace

    import predictive_agent.collector as collector_module
    import predictive_agent.optimize as opt_mod
    from predictive_agent.optimize import OptimizationConfig, PerformanceOptimizer

    monkeypatch.setattr(collector_module, "get_kubectl_cache_stats",
                        lambda: {"hits": 3, "misses": 1, "hit_rate": "75.00%", "size": 0})
    fake_cache = SimpleNamespace(get_stats=lambda: {
        "llm_prompt": {"hits": 1, "misses": 3},
        "sizes": {"llm_prompt": 1},
    })
    monkeypatch.setattr(opt_mod, "get_cache", lambda: fake_cache)

    optimizer = PerformanceOptimizer(config=OptimizationConfig())
    summary = optimizer.get_summary()

    assert summary["savings"]["kubectl_cache_hit_rate"] == "75.0%"
    assert summary["savings"]["llm_cache_hit_rate"] == "25.0%"
    assert summary["optimizations"]["kubectl_caching"] is True
    assert summary["optimizations"]["l1_cache"] is True

    optimizer.print_summary()
    out = capsys.readouterr().out
    assert "PERFORMANCE OPTIMIZATION SUMMARY" in out
    assert "75.0%" in out
    assert "✓ kubectl_caching" in out
