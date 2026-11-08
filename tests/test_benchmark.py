"""Tests for the POMDP planner vs reactive-baseline benchmark."""

from __future__ import annotations

from predictive_agent.benchmark import (
    benchmark_scenario,
    reactive_actions,
    run_benchmark,
    scenarios_from_backtester,
)
from predictive_agent.markov import MarkovChain
from predictive_agent.pomdp import ActionAwareTransitionModel
from predictive_agent.remediator import (
    RemediationAction,
    RemediationManager,
    RemediationResult,
)


class _Pod:
    def __init__(self, state):
        self.state = state
        self.name = "ns/pod"
        self.namespace = "ns"
        self.node_name = "n"
        self.pod_phase = "Running"
        self.cpu_trend = 0.0
        self.memory_trend = 0.0
        self.memory_pct = 0
        self.cpu_pct = 0
        self.memory_anomaly_score = 0
        self.cpu_anomaly_score = 0


class _FakeAction(RemediationAction):
    name = "pod_restart"

    def should_execute(self, pod_state, prediction, risk_score):
        return True

    def execute(self, target, context):
        return RemediationResult(
            action=self.name, target=target, success=True,
            dry_run=context.dry_run, message="ok",
        )


def _manager():
    rem = RemediationManager(dry_run=False)
    rem.register_action(_FakeAction())
    rem.decision_model = ActionAwareTransitionModel(base_chain=MarkovChain())
    return rem


def test_reactive_actions_empty_below_threshold():
    rem = _manager()
    # Below the risk threshold, the reactive path fires nothing.
    assert reactive_actions(rem, "CRITICAL", 0.1) == []


def test_reactive_actions_fires_above_threshold():
    rem = _manager()
    actions = reactive_actions(rem, "CRITICAL", 0.95)
    assert "pod_restart" in actions


def test_benchmark_scenario_structure():
    rem = _manager()
    res = benchmark_scenario(rem, "CRITICAL", 0.95, horizon=3)
    # The planner optimises expected P(FAILED), so it is never worse than doing
    # nothing. (It is not guaranteed to beat the multi-action reactive set — that
    # gap is exactly what the benchmark measures.)
    assert res.planner_p_failed <= res.none_p_failed + 1e-9
    # From a bad state both strategies intervene.
    assert res.planner_actions
    assert res.reactive_actions


def test_run_benchmark_summary():
    rem = _manager()
    scenarios = [("HEALTHY", 0.2), ("STRESSED", 0.8), ("CRITICAL", 0.95), ("FAILED", 0.99)]
    report = run_benchmark(rem, scenarios, horizon=3)
    summary = report.summary()
    assert summary["scenarios"] == 4
    assert "agreement_rate" in summary
    assert "mean_planner_reduction_vs_reactive" in summary
    # Sanity: every scenario row is present.
    assert len(report.scenarios) == 4


def test_scenarios_from_backtester():
    class _Rec:
        def __init__(self, ms, rs):
            self.markov_state = ms
            self.risk_score = rs
            self.timestamp = 1.0

    class _BT:
        _predictions = [
            _Rec("STRESSED", 0.8),
            _Rec("STRESSED", 0.8),  # duplicate -> deduped
            _Rec("CRITICAL", 0.95),
            _Rec("BOGUS", 0.5),     # invalid state -> skipped
        ]

    scenarios = scenarios_from_backtester(_BT())
    states = {s[0] for s in scenarios}
    assert "STRESSED" in states and "CRITICAL" in states
    assert "BOGUS" not in states
    assert len(scenarios) == 2
