"""Benchmark the POMDP planner against the reactive remediation baseline.

This quantifies, per scenario, the expected ``P(FAILED)`` under three decision
strategies using the operator's learned ``decision_model``:

* **none** — do nothing (the natural decay of the Markov chain).
* **reactive** — the production ``evaluate()`` would-fire action set (dry-run
  analysis; no execution), i.e. the current per-action remediation behaviour.
* **planner** — the POMDP planner's chosen intervention sequence.

Scenarios can be supplied explicitly or derived from real backtester
observations (the ``markov_state`` / ``risk_score`` of historical predictions),
so the comparison reflects the fleet's actual state distribution rather than
synthetic inputs.

The key metric is ``planner_reduction_vs_reactive``: how much more (or less)
expected failure probability the planner removes compared with the reactive
baseline. Positive => the planner is at least as good as production today.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List, Tuple

from predictive_agent.markov import MarkovChain
from predictive_agent.pomdp import Planner, ValueIterationPolicy
from predictive_agent.rl import QLearningPolicy

logger = logging.getLogger(__name__)

Scenario = Tuple[str, float]  # (markov_state, risk_score)


def _onehot(state: str) -> dict:
    dist = {s: 0.0 for s in MarkovChain.STATES}
    dist[state] = 1.0
    return dist


def expected_failed(model, state: str, actions: List[str], horizon: int) -> float:
    """Expected ``P(FAILED)`` after applying ``actions`` (cycled) over ``horizon``."""
    dist = _onehot(state)
    if not actions:
        actions = ["none"]
    for step in range(horizon):
        action = actions[step % len(actions)]
        dist = model.step_distribution(dist, action)
    return model.failed_probability(dist)


class _Tracker:
    """Minimal pod-like stub for ``should_execute`` probes (no real pod needed)."""

    def __init__(self, state: str):
        self.state = state
        self.name = "bench"
        self.namespace = ""
        self.node_name = ""
        self.pod_phase = "Running"
        self.cpu_trend = 0.0
        self.memory_trend = 0.0


def reactive_actions(manager, state: str, risk: float) -> List[str]:
    """Replicate ``evaluate()``'s would-fire action set without executing.

    Mirrors the gating in ``RemediationManager.evaluate``: below the risk
    threshold nothing fires; otherwise each registered action is included when
    its ``should_execute`` and the ``SafetyPolicy`` allow it. Dry-run analysis
    only — no remediation is performed.
    """
    if risk < manager.risk_threshold:
        return []
    fired: List[str] = []
    tracker = _Tracker(state)
    for action in manager.actions:
        try:
            if not action.should_execute(tracker, None, risk):
                continue
        except Exception:  # noqa: BLE001 - a broken guard must not crash the bench
            continue
        can_exec, _ = manager.safety_policy.can_execute(
            "bench", action.name, manager.context
        )
        if can_exec:
            fired.append(action.name)
    return fired


@dataclass
class ScenarioResult:
    state: str
    risk: float
    none_p_failed: float
    reactive_actions: List[str]
    reactive_p_failed: float
    planner_actions: List[str]
    planner_p_failed: float
    planner_reduction_vs_reactive: float  # reactive - planner (>0 => planner better)
    policy_actions: List[str] = field(default_factory=list)
    policy_p_failed: float = 0.0
    q_actions: List[str] = field(default_factory=list)
    q_p_failed: float = 0.0

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "risk": self.risk,
            "none_p_failed": round(self.none_p_failed, 4),
            "reactive_actions": self.reactive_actions,
            "reactive_p_failed": round(self.reactive_p_failed, 4),
            "planner_actions": self.planner_actions,
            "planner_p_failed": round(self.planner_p_failed, 4),
            "planner_reduction_vs_reactive": round(self.planner_reduction_vs_reactive, 4),
            "policy_actions": self.policy_actions,
            "policy_p_failed": round(self.policy_p_failed, 4),
            "q_actions": self.q_actions,
            "q_p_failed": round(self.q_p_failed, 4),
        }


def benchmark_scenario(manager, state: str, risk: float, horizon: int = 3) -> ScenarioResult:
    """Evaluate one (state, risk) scenario across the three strategies."""
    model = manager.decision_model
    none_pf = expected_failed(model, state, [], horizon)
    reactive = reactive_actions(manager, state, risk)
    # Reactive fires its would-fire set once (subsequent repeats are blocked by
    # the SafetyPolicy cooldown in production), then does nothing for the rest
    # of the horizon — matching real per-cycle remediation behaviour.
    reactive_dist = _onehot(state)
    for i in range(horizon):
        action = reactive[i] if i < len(reactive) else "none"
        reactive_dist = model.step_distribution(reactive_dist, action)
    reactive_pf = model.failed_probability(reactive_dist)

    plan = Planner(model=model).plan(state, horizon)
    planner_actions = [s.action for s in plan.steps if s.action != "none"]
    planner_pf = plan.steps[-1].expected_failed_prob_after

    # 4th strategy: the stationary optimal policy from value iteration, rolled
    # out over the horizon (the model-based RL / dynamic-programming baseline).
    policy = ValueIterationPolicy(model)
    pol_dist = _onehot(state)
    policy_actions = []
    for _ in range(horizon):
        current = max(pol_dist, key=lambda s: pol_dist[s])
        a = policy.action(current)
        policy_actions.append(a)
        pol_dist = model.step_distribution(pol_dist, a)
    policy_pf = model.failed_probability(pol_dist)

    # 5th strategy: model-free Q-learning policy (sampled RL, #8). Reuses the
    # manager's calibrated costs so it is directly comparable to the others.
    q_policy = QLearningPolicy(
        model=model,
        action_costs=manager.action_costs,
        failure_cost=manager.failure_cost,
    )
    q_actions, q_pf = q_policy.plan(state, horizon, model)

    return ScenarioResult(
        state=state,
        risk=risk,
        none_p_failed=none_pf,
        reactive_actions=reactive,
        reactive_p_failed=reactive_pf,
        planner_actions=planner_actions,
        planner_p_failed=planner_pf,
        planner_reduction_vs_reactive=reactive_pf - planner_pf,
        policy_actions=policy_actions,
        policy_p_failed=policy_pf,
        q_actions=q_actions,
        q_p_failed=q_pf,
    )


@dataclass
class BenchmarkReport:
    scenarios: List[ScenarioResult] = field(default_factory=list)
    horizon: int = 3

    def summary(self) -> dict:
        n = len(self.scenarios)
        if n == 0:
            return {"scenarios": 0}
        planner_better = sum(
            1 for s in self.scenarios if s.planner_reduction_vs_reactive > 1e-6
        )
        agree = sum(
            1
            for s in self.scenarios
            if (len(s.reactive_actions) > 0) == (len(s.planner_actions) > 0)
        )
        mean_red = sum(s.planner_reduction_vs_reactive for s in self.scenarios) / n
        mean_reactive = sum(s.reactive_p_failed for s in self.scenarios) / n
        mean_planner = sum(s.planner_p_failed for s in self.scenarios) / n
        mean_policy = sum(s.policy_p_failed for s in self.scenarios) / n
        mean_q = sum(s.q_p_failed for s in self.scenarios) / n
        return {
            "scenarios": n,
            "planner_intervenes": sum(1 for s in self.scenarios if s.planner_actions),
            "reactive_intervenes": sum(1 for s in self.scenarios if s.reactive_actions),
            "policy_intervenes": sum(1 for s in self.scenarios if s.policy_actions),
            "q_intervenes": sum(1 for s in self.scenarios if s.q_actions),
            "agree_on_intervention": agree,
            "agreement_rate": round(agree / n, 3),
            "planner_better_or_equal_count": planner_better,
            "mean_reactive_p_failed": round(mean_reactive, 4),
            "mean_planner_p_failed": round(mean_planner, 4),
            "mean_policy_p_failed": round(mean_policy, 4),
            "mean_q_p_failed": round(mean_q, 4),
            "mean_planner_reduction_vs_reactive": round(mean_red, 4),
        }

    def to_text(self) -> str:
        lines = ["# POMDP planner vs reactive baseline — benchmark", ""]
        lines.append(f"horizon={self.horizon}  scenarios={len(self.scenarios)}")
        for s in self.scenarios:
            lines.append(
                f"  {s.state:9s} risk={s.risk:.2f} | none={s.none_p_failed:.3f} "
                f"reactive={s.reactive_p_failed:.3f} {s.reactive_actions} "
                f"planner={s.planner_p_failed:.3f} {s.planner_actions} "
                f"policy={s.policy_p_failed:.3f} {s.policy_actions}"
            )
        lines.append("")
        lines.append("summary: " + json.dumps(self.summary()))
        return "\n".join(lines)


def run_benchmark(manager, scenarios: List[Scenario], horizon: int = 3) -> BenchmarkReport:
    """Run the benchmark over a list of (markov_state, risk_score) scenarios."""
    report = BenchmarkReport(horizon=horizon)
    for state, risk in scenarios:
        report.scenarios.append(benchmark_scenario(manager, state, risk, horizon))
    return report


def scenarios_from_backtester(backtester, limit: int = 200) -> List[Scenario]:
    """Derive benchmark scenarios from real backtester predictions.

    Returns de-duplicated ``(markov_state, risk_score)`` pairs from the most
    recent predictions, so the benchmark reflects the fleet's actual state/risk
    distribution rather than synthetic inputs.
    """
    preds = getattr(backtester, "_predictions", []) or []
    seen = set()
    scenarios: List[Scenario] = []
    for p in sorted(preds, key=lambda r: getattr(r, "timestamp", 0), reverse=True):
        if p.markov_state not in MarkovChain.STATE_IDX:
            continue
        key = (p.markov_state, round(float(p.risk_score), 2))
        if key in seen:
            continue
        seen.add(key)
        scenarios.append((p.markov_state, float(p.risk_score)))
        if len(scenarios) >= limit:
            break
    return scenarios


def main() -> None:  # pragma: no cover - CLI entry point
    """CLI: benchmark the live decision model (and backtester if available)."""
    import os

    from predictive_agent.persistence import StateStore
    from predictive_agent.pomdp import ActionAwareTransitionModel
    from predictive_agent.remediator import create_remediation_manager_from_config

    state_file = os.environ.get("STATE_MODEL_FILE", "state_model.json")
    store = StateStore(state_model_file=state_file)

    manager = create_remediation_manager_from_config()
    from predictive_agent.actions import (  # noqa: F401 - register side effects
        DeploymentScaleAction,
        NodeCordonAction,
        PodRestartAction,
        ResourceTunerAction,
        RightSizeAction,
        RolloutRestartAction,
    )
    manager.register_action(PodRestartAction())
    manager.register_action(NodeCordonAction())
    manager.register_action(RightSizeAction())
    manager.register_action(RolloutRestartAction())
    manager.register_action(DeploymentScaleAction())
    manager.register_action(ResourceTunerAction())

    model = ActionAwareTransitionModel()
    loaded = store.load_decision_model()
    if loaded:
        model.load_action_counts(loaded)
    manager.decision_model = model

    horizon = int(os.environ.get("PLANNING_HORIZON", "3"))

    scenarios: List[Scenario] = []
    try:
        from predictive_agent.backtester import Backtester

        bt = Backtester()
        scenarios = scenarios_from_backtester(bt)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not derive scenarios from backtester: %s", e)

    if not scenarios:
        scenarios = [(s, 0.85) for s in MarkovChain.STATES]

    report = run_benchmark(manager, scenarios, horizon=horizon)
    print(report.to_text())


if __name__ == "__main__":  # pragma: no cover
    main()
