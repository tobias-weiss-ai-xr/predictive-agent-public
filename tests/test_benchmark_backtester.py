"""Tests for the decision-layer benchmark over real backtester data (#4).

The 4-strategy benchmark (none / reactive / planner / value-iteration policy)
should run over the fleet's *actual* (markov_state, risk_score) distribution
derived from historical predictions, not just synthetic inputs.
"""

import json as _json
import urllib.request

from predictive_agent.backtester import Backtester
from predictive_agent.benchmark import (
    run_benchmark,
    scenarios_from_backtester,
)
from predictive_agent.pomdp import ActionAwareTransitionModel, MarkovChain
from predictive_agent.remediator import (
    RemediationAction,
    RemediationManager,
    RemediationResult,
)
from predictive_agent.server import _context, start_server

ACTION_NAMES = [
    "pod_restart",
    "rollout_restart",
    "node_cordon",
    "scale",
    "right_size",
    "tune_resources",
]


class _FakeAction(RemediationAction):
    def __init__(self, name):
        self.name = name

    def should_execute(self, pod_state, prediction, risk_score):
        return True

    def execute(self, target, context):
        return RemediationResult(
            action=self.name, target=target, success=True,
            dry_run=context.dry_run, message="ok (fake)",
        )


def _build_manager():
    rem = RemediationManager(dry_run=False)
    for name in ACTION_NAMES:
        rem.register_action(_FakeAction(name))
    model = ActionAwareTransitionModel(base_chain=MarkovChain())
    for _ in range(40):
        model.record_transition("CRITICAL", "RECOVERED", "pod_restart")
        model.record_transition("CRITICAL", "FAILED", "none")
        model.record_transition("STRESSED", "RECOVERED", "scale")
        model.record_transition("STRESSED", "FAILED", "none")
    rem.decision_model = model
    return rem


def _seeded_backtester(tmp_path):
    bt = Backtester(history_file=str(tmp_path / "bt.json"))
    # Historical predictions across the fleet's real state distribution.
    bt.record_prediction("ns/a", 0.92, 5, 0.8, "CRITICAL")
    bt.record_prediction("ns/b", 0.88, 7, 0.8, "STRESSED")
    bt.record_prediction("ns/c", 0.60, 20, 0.7, "DEGRADED")
    return bt


def test_scenarios_from_backtester_derives_real_scenarios(tmp_path):
    bt = _seeded_backtester(tmp_path)
    scenarios = scenarios_from_backtester(bt)
    states = {s for s, _ in scenarios}
    assert states == {"CRITICAL", "STRESSED", "DEGRADED"}
    # Each scenario's state is a real Markov state and risk is in [0, 1].
    for state, risk in scenarios:
        assert state in MarkovChain.STATE_IDX
        assert 0.0 <= risk <= 1.0


def test_run_benchmark_over_real_scenarios(tmp_path):
    rem = _build_manager()
    bt = _seeded_backtester(tmp_path)
    scenarios = scenarios_from_backtester(bt)
    report = run_benchmark(rem, scenarios, horizon=3)
    assert len(report.scenarios) == len(scenarios)
    for s in report.scenarios:
        # All four strategies must be evaluated per scenario.
        assert "none_p_failed" in s.to_dict()
        assert "reactive_p_failed" in s.to_dict()
        assert "planner_p_failed" in s.to_dict()
        assert "policy_p_failed" in s.to_dict()
    summary = report.summary()
    assert summary["scenarios"] == len(scenarios)
    assert "mean_planner_p_failed" in summary


def test_benchmark_endpoint_uses_backtester_data(tmp_path):
    rem = _build_manager()
    bt = _seeded_backtester(tmp_path)

    ports = (18180, 18181)
    srv = start_server(*ports)
    _context.remediation_manager = rem
    _context.backtester = bt
    try:
        with urllib.request.urlopen(f"http://localhost:{ports[0]}/benchmark") as resp:
            status = resp.status
            data = _json.loads(resp.read().decode())
    finally:
        _context.remediation_manager = None
        _context.backtester = None
        srv.shutdown()
        srv.server_close()

    assert status == 200
    assert data["source"] == "backtester"
    assert len(data["scenarios"]) == 3
    assert data["summary"]["scenarios"] == 3


def test_benchmark_endpoint_falls_back_to_synthetic(tmp_path):
    rem = _build_manager()  # no backtester attached -> synthetic scenarios

    ports = (18182, 18183)
    srv = start_server(*ports)
    _context.remediation_manager = rem
    try:
        with urllib.request.urlopen(f"http://localhost:{ports[0]}/benchmark") as resp:
            status = resp.status
            data = _json.loads(resp.read().decode())
    finally:
        _context.remediation_manager = None
        srv.shutdown()
        srv.server_close()

    assert status == 200
    assert data["source"] == "synthetic"
    assert len(data["scenarios"]) == len(MarkovChain.STATES)
