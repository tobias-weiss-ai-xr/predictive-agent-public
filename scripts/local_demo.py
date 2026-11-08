#!/usr/bin/env python3
"""Local, cluster-free demonstration of the POMDP decision layer.

Spins up the real openDesk Predictive Agent HTTP server and drives the full
decision loop against a synthetic, degrading pod -- no Kubernetes cluster
required (actions are faked so nothing is actually executed).

Run it:

    python scripts/local_demo.py

It will:
  1. Boot the metrics/API server (ports 18160/18161).
  2. Simulate a pod degrading HEALTHY -> ... -> CRITICAL, planning + (dry-run)
     executing + learning at each step.
  3. Show the calibrated uncertainty interval (#4), the decision-theoretic cost
     model (#6), the world-model what-if query (#7) and the value-iteration
     policy (#8).
  4. Print the four-strategy benchmark (none / reactive / planner / policy).
  5. Verify model persistence round-trips to disk.
  6. Hit the live /plan, /world-model and /backtest endpoints.
"""

import json
import os
import sys
import tempfile
import time
import urllib.request

# Make the package importable when run as a standalone script from scripts/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictive_agent.backtester import Backtester
from predictive_agent.benchmark import run_benchmark
from predictive_agent.markov import MarkovChain
from predictive_agent.persistence import StateStore
from predictive_agent.pomdp import ActionAwareTransitionModel, ValueIterationPolicy
from predictive_agent.remediator import (
    RemediationAction,
    RemediationManager,
    RemediationResult,
)
from predictive_agent.server import start_server

ACTION_NAMES = [
    "pod_restart",
    "rollout_restart",
    "node_cordon",
    "scale",
    "right_size",
    "tune_resources",
]
METRICS_PORT = 18160
HEALTH_PORT = 18161


class _FakeAction(RemediationAction):
    def __init__(self, name):
        self.name = name

    def should_execute(self, pod_state, prediction, risk_score):
        return True

    def execute(self, target, context):
        return RemediationResult(
            action=self.name,
            target=target,
            success=True,
            dry_run=context.dry_run,
            message="ok (fake, no kubectl)",
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


def _banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _get(path):
    with urllib.request.urlopen(f"http://localhost:{METRICS_PORT}{path}") as r:
        return json.loads(r.read().decode())


def main():
    tmp = tempfile.mkdtemp(prefix="predagent-demo-")
    rem = RemediationManager(dry_run=False)
    for name in ACTION_NAMES:
        rem.register_action(_FakeAction(name))
    model = ActionAwareTransitionModel(base_chain=MarkovChain())
    # Seed learning: pod_restart recovers CRITICAL; scale recovers STRESSED.
    for _ in range(40):
        model.record_transition("CRITICAL", "RECOVERED", "pod_restart")
        model.record_transition("CRITICAL", "FAILED", "none")
        model.record_transition("STRESSED", "RECOVERED", "scale")
        model.record_transition("STRESSED", "FAILED", "none")
    rem.decision_model = model
    bt = Backtester(history_file=f"{tmp}/backtester.json")

    server = start_server(METRICS_PORT, HEALTH_PORT, remediation_manager=rem, backtester=bt)
    time.sleep(0.8)
    _banner("1. Live decision loop on a degrading pod (dry-run execution)")
    trajectory = ["HEALTHY", "DEGRADED", "STRESSED", "CRITICAL"]
    for state in trajectory:
        pod = _Pod(state)
        plan = rem.plan_remediation(state, horizon=3)
        actions = [s["action"] for s in plan["steps"] if s["action"] != "none"]
        print(f"  state={state:9s} planned={actions}")
        if actions:
            rem.plan_and_execute(state, "ns/pod", pod, None, 0.95, horizon=3)
            # Learn from the simulated observed next state.
            rem.feed_transition(state, actions[0], "RECOVERED")
        time.sleep(0.05)

    _banner("2. Calibrated uncertainty on the plan (#4)")
    plan = rem.plan_remediation("CRITICAL", horizon=3)
    step0 = plan["steps"][0]
    print(f"  CRITICAL -> first action {step0['action']}")
    print(f"  P(FAILED) point     = {step0['expected_failed_prob_after']}")
    print(f"  P(FAILED) 95% CI    = {step0['expected_failed_interval_after']}")
    # Contrast: a freshly-seeded low-data model shows a WIDE interval.
    lowdata = ActionAwareTransitionModel(base_chain=MarkovChain())
    for _ in range(3):
        lowdata.record_transition("CRITICAL", "RECOVERED", "pod_restart")
    lo_mean, lo_lo, lo_hi = lowdata.predictive_interval({"CRITICAL": 1.0}, "pod_restart")
    full = rem.decision_model.predictive_interval({"CRITICAL": 1.0}, "pod_restart")
    print(f"  low-data  CI width  = {round(lo_hi - lo_lo, 3)}  (high uncertainty)")
    print(f"  learned   CI width  = {round(full[2] - full[0], 3)}  (uncertainty shrinks with data)")

    _banner("3. Decision-theoretic cost model (#6)")
    from predictive_agent.pomdp import Planner

    cheap = Planner(model=model).plan("CRITICAL", 3)
    costly = Planner(model=model, action_costs={a: 10.0 for a in ACTION_NAMES}).plan("CRITICAL", 3)
    print(f"  default costs -> acts with: {[s.action for s in cheap.steps if s.action != 'none']}")
    print(f"  inflated costs -> acts with: {[s.action for s in costly.steps if s.action != 'none']}")

    _banner("4. World-model what-if query (#7)")
    wm = _get("/world-model?state=CRITICAL&action=pod_restart&horizon=2")
    for s in wm["steps"]:
        print(f"  step {s['step']}: P(FAILED)={s['expected_failed_prob']} "
              f"CI={s['expected_failed_interval']}")

    _banner("5. Value-iteration (RL) stationary policy (#8)")
    pol = ValueIterationPolicy(model=model)
    print(f"  policy[CRITICAL] = {pol.action('CRITICAL')}")
    print(f"  policy[STRESSED] = {pol.action('STRESSED')}")
    pplan = _get("/plan?state=CRITICAL&strategy=policy")
    print(f"  /plan?strategy=policy -> {[s['action'] for s in pplan['plan']['steps']]}")

    _banner("6. Four-strategy benchmark (none / reactive / planner / policy)")
    report = run_benchmark(rem, [("CRITICAL", 0.9), ("STRESSED", 0.9)], horizon=3)
    for s in report.scenarios:
        print(f"  {s.state:9s} none={s.none_p_failed:.3f} reactive={s.reactive_p_failed:.3f} "
              f"planner={s.planner_p_failed:.3f} policy={s.policy_p_failed:.3f}")
    print(f"  summary: {json.dumps(report.summary())}")

    _banner("7. Persistence round-trip")
    store = StateStore(state_model_file=f"{tmp}/state.json")
    store.save_decision_model(rem.decision_model.to_action_dict())
    loaded = store.load_decision_model()
    restored = ActionAwareTransitionModel(base_chain=MarkovChain())
    restored.load_action_counts(loaded)
    ok = restored.action_counts == rem.decision_model.action_counts
    print(f"  saved -> loaded -> restored matches in-memory model: {ok}")

    _banner("8. Live endpoints")
    print("  GET /plan            ->", _get("/plan?state=CRITICAL&horizon=3")["status"])
    print("  GET /world-model     ->", _get("/world-model?state=CRITICAL&action=scale&horizon=2")["status"])
    print("  GET /backtest        ->", "ok" if "brier_score" in _get("/backtest") else "missing")

    server.shutdown()
    server.server_close()
    print("\nDemo complete. Server stopped.")


if __name__ == "__main__":
    main()
