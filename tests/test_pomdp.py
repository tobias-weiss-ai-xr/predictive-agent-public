"""Tests for the POMDP decision layer (predictive_agent.pomdp)."""

from __future__ import annotations

import math
import time
from types import SimpleNamespace

import pytest

from predictive_agent.markov import MarkovChain
from predictive_agent.pomdp import (
    ACTION_TYPES,
    DEFAULT_ACTION_COSTS,
    ActionAwareTransitionModel,
    Planner,
    ValueIterationPolicy,
    _step_rationale,
    advisory_plan,
)


def _fresh() -> ActionAwareTransitionModel:
    return ActionAwareTransitionModel()


def _ref_value_iteration(m, failure_cost=1.0, gamma=0.95, threshold=1e-4, max_iter=500):
    """Mirror of ValueIterationPolicy.solve/_q/_reward/_action_cost algebra."""

    def cost(a):
        return 0.0 if a == "none" else DEFAULT_ACTION_COSTS.get(a, 0.0)

    def reward(s, a):
        dist = {x: 0.0 for x in MarkovChain.STATES}
        dist[s] = 1.0
        after = m.step_distribution(dist, a)
        return -failure_cost * after.get("FAILED", 0.0) - cost(a)

    def q(s, a, V):
        row = m.transition_matrix(a)[MarkovChain.STATE_IDX[s]]
        return reward(s, a) + gamma * sum(
            row[j] * V[MarkovChain.STATES[j]] for j in range(6)
        )

    V = {s: 0.0 for s in MarkovChain.STATES}
    for _ in range(max_iter):
        new_V = {}
        for s in MarkovChain.STATES:
            best = max((q(s, a, V) for a in ACTION_TYPES), default=float("-inf"))
            new_V[s] = best if best > float("-inf") else 0.0
        delta = max(abs(new_V[s] - V[s]) for s in MarkovChain.STATES)
        V = new_V
        if delta < threshold:
            break
    policy = {
        s: max(ACTION_TYPES, key=lambda a: q(s, a, V)) for s in MarkovChain.STATES
    }
    return V, policy


# ── Planner.plan exact contract ──────────────────────────────────────────


def test_planner_single_step_exact_fields():
    p = Planner(model=_fresh())
    plan = p.plan("CRITICAL", horizon=1)
    assert plan.current_state == "CRITICAL"
    assert plan.horizon == 1
    s = plan.steps[0]
    assert s.step == 0
    assert s.action == "scale"
    assert s.expected_failed_prob_before == 0.0
    assert s.expected_failed_prob_after == 0.125
    assert s.expected_failed_interval_after == (
        pytest.approx(0.1650641025641026),
        0.0,
        pytest.approx(0.6590981004414721),
    )
    assert s.expected_state_distribution == {
        "HEALTHY": 0.125,
        "DEGRADED": 0.125,
        "STRESSED": pytest.approx(0.08333333333333333),
        "CRITICAL": 0.5,
        "FAILED": 0.125,
        "RECOVERED": pytest.approx(0.041666666666666664),
    }
    assert s.rationale == (
        "Step 0: apply scale from CRITICAL — expected P(FAILED) stays ~12.5% "
        "(no further reduction expected); action cost 0.02."
    )
    assert plan.rationale == (
        "Greedy Scenario-MPC lookahead over 1 steps minimising expected "
        "P(FAILED); intervention chosen only when it strictly reduces risk "
        "beyond the action-cost penalty. Output is passed to RemediationManager "
        "for execution under SafetyPolicy."
    )


def test_planner_cooldown_writeback_and_second_plan():
    p = Planner(model=_fresh())
    before = time.time()
    p.plan("FAILED", horizon=1)
    after = time.time()
    assert set(p.cooldowns) == {"pod_restart"}
    assert before <= p.cooldowns["pod_restart"] <= after
    # pod_restart now cooling down: rollout_restart must win over none
    follow = p.plan("FAILED", horizon=1)
    assert follow.steps[0].action == "rollout_restart"


def test_planner_does_not_cooldown_none_action():
    p = Planner(
        model=_fresh(),
        cooldowns={"scale": time.time(), "pod_restart": time.time()},
    )
    plan = p.plan("CRITICAL", horizon=1)
    assert plan.steps[0].action == "none"
    # "none" must never be recorded as cooling down (prefill stays untouched)
    assert set(p.cooldowns) == {"scale", "pod_restart"}


def test_planner_failure_cost_drives_decision_not_just_penalty():
    # with failure_cost=1, none (0.8) vs pod_restart (0.5 + 0.05): pod wins;
    # with failure_cost=100 none would score 80 — pod_restart must still win.
    p = Planner(model=_fresh(), failure_cost=100.0)
    assert p.plan("FAILED", horizon=1).steps[0].action == "pod_restart"


def test_plan_to_dict_exact_rounded_payload():
    plan = Planner(model=_fresh()).plan("CRITICAL", horizon=3)
    d = plan.to_dict()
    assert d == {
        "current_state": "CRITICAL",
        "horizon": 3,
        "execution_ready": True,
        "steps": [
            {
                "step": 0,
                "action": "scale",
                "expected_failed_prob_before": 0.0,
                "expected_failed_prob_after": 0.125,
                "expected_failed_interval_after": [0.1651, 0.0, 0.6591],
                "expected_state_distribution": {
                    "HEALTHY": 0.125,
                    "DEGRADED": 0.125,
                    "STRESSED": 0.0833,
                    "CRITICAL": 0.5,
                    "FAILED": 0.125,
                    "RECOVERED": 0.0417,
                },
                "rationale": (
                    "Step 0: apply scale from CRITICAL — expected P(FAILED) "
                    "stays ~12.5% (no further reduction expected); action cost 0.02."
                ),
            },
            {
                "step": 1,
                "action": "pod_restart",
                "expected_failed_prob_before": 0.125,
                "expected_failed_prob_after": 0.1151,
                "expected_failed_interval_after": [0.0819, 0.0, 0.3958],
                "expected_state_distribution": {
                    "HEALTHY": 0.2064,
                    "DEGRADED": 0.0928,
                    "STRESSED": 0.0779,
                    "CRITICAL": 0.2202,
                    "FAILED": 0.1151,
                    "RECOVERED": 0.2876,
                },
                "rationale": (
                    "Step 1: apply pod_restart from CRITICAL — expected P(FAILED) "
                    "12.5% -> 11.5% (reduces failure risk by 1.0%); action cost "
                    "0.05 vs failure cost 1.00."
                ),
            },
            {
                "step": 2,
                "action": "none",
                "expected_failed_prob_before": 0.1151,
                "expected_failed_prob_after": 0.1282,
                "expected_failed_interval_after": [0.1292, 0.0, 0.6514],
                "expected_state_distribution": {
                    "HEALTHY": 0.4728,
                    "DEGRADED": 0.1244,
                    "STRESSED": 0.0836,
                    "CRITICAL": 0.1553,
                    "FAILED": 0.1282,
                    "RECOVERED": 0.0357,
                },
                "rationale": (
                    "Step 2: stay put from RECOVERED (expected P(FAILED)=11.5%); "
                    "intervening is not worthwhile — the cost of acting (0.00) "
                    "outweighs the failure-risk reduction."
                ),
            },
        ],
        "rationale": plan.rationale,
    }


# ── predictive_interval closed-form contract ─────────────────────────────


def test_predictive_interval_matches_closed_form_learned():
    m = _fresh()
    # learned overrides: STRESSED p=0.45, CRITICAL p=0.55 (totals 100)
    m.record_transition("STRESSED", "FAILED", "pod_restart")
    m.record_transition("STRESSED", "FAILED", "pod_restart")
    for _ in range(42):
        m.record_transition("STRESSED", "FAILED", "pod_restart")
    for _ in range(55):
        m.record_transition("STRESSED", "HEALTHY", "pod_restart")
    for _ in range(54):
        m.record_transition("CRITICAL", "FAILED", "pod_restart")
    for _ in range(45):
        m.record_transition("CRITICAL", "HEALTHY", "pod_restart")
    dist = {"STRESSED": 0.5, "CRITICAL": 0.5}
    mean, lo, hi = m.predictive_interval(dist, "pod_restart")
    [m.base_chain.counts[2], m.base_chain.counts[3]]
    learned = [m.action_counts["pod_restart"][2], m.action_counts["pod_restart"][3]]
    ps, vs = [], []
    for row in learned:
        tot = sum(row)
        fc = row[4]
        ps.append(fc / tot)
        vs.append((fc + 1) * (tot - fc + 1) / (((tot + 2) ** 2) * (tot + 3)))
    exp_mean = 0.5 * ps[0] + 0.5 * ps[1]
    exp_var = (
        0.25 * vs[0]
        + 0.25 * vs[1]
        + 0.5 * (ps[0] - exp_mean) ** 2
        + 0.5 * (ps[1] - exp_mean) ** 2
    )
    exp_sd = math.sqrt(exp_var)
    assert mean == pytest.approx(exp_mean)
    assert hi == pytest.approx(exp_mean + 1.96 * exp_sd)  # z=1.96 for default alpha
    assert lo == pytest.approx(exp_mean - 1.96 * exp_sd)  # both sides unclamped
    # alpha > 0.05 selects the 1.645 quantile
    _, lo2, hi2 = m.predictive_interval(dist, "pod_restart", alpha=0.2)
    assert hi2 == pytest.approx(exp_mean + 1.645 * exp_sd)
    assert lo2 == pytest.approx(exp_mean - 1.645 * exp_sd)
    # zero-weight entry must be skipped, not terminate the scan
    mean_z, _, hi_z = m.predictive_interval({"HEALTHY": 0.0, **dist}, "pod_restart")
    assert mean_z == pytest.approx(exp_mean)
    assert hi_z == pytest.approx(exp_mean + 1.96 * exp_sd)
    # unknown state maps to the HEALTHY row of the merged counts
    mean_u, lo_u, hi_u = m.predictive_interval({"BOGUS": 1.0}, "none")
    row = m.base_chain.counts[0]
    tot = sum(row)
    v0 = (row[4] + 1) * (tot - row[4] + 1) / (((tot + 2) ** 2) * (tot + 3))
    assert mean_u == 0.0
    assert lo_u == 0.0
    assert hi_u == pytest.approx(1.96 * math.sqrt(v0))
    # interval for a single-count learned row still uses the Beta posterior
    m.record_transition("HEALTHY", "RECOVERED", "scale")
    mean_s, _, hi_s = m.predictive_interval({"HEALTHY": 1.0}, "scale")
    assert mean_s == 0.0
    assert hi_s == pytest.approx(1.96 * math.sqrt(2 / 36))
    # empty-row states are skipped so later states still contribute
    m2 = _fresh()
    m2.record_transition("CRITICAL", "FAILED", "scale")
    mean_e = m2.predictive_interval(
        {"HEALTHY": 0.3, "STRESSED": 0.3, "CRITICAL": 0.4}, "scale"
    )[0]
    assert mean_e == pytest.approx(0.4)
    # clamped upper bound never exceeds 1
    assert m2.predictive_interval({"FAILED": 1.0}, "none")[2] == 1.0


# ── backtester refinement pairing ─────────────────────────────────────────


def _p(ts, pod, state):
    return SimpleNamespace(timestamp=ts, pod_key=pod, markov_state=state, state=None)


def _o(ts, pod, state):
    return SimpleNamespace(timestamp=ts, pod_key=pod, state=state, markov_state=None)


def test_refine_from_backtester_pairing_and_counts_exact():
    m = _fresh()
    bt = SimpleNamespace(
        _predictions=[
            _p(3, "B", "DEGRADED"),
            _p(1, "A", "HEALTHY"),
            _p(2, "A", "STRESSED"),
            _p(4, "A", "BOGUS"),
        ],
        # out of chronological order on purpose: sorting matters
        _outcomes=[
            _o(9, "B", "FAILED"),
            _o(2, "A", "CRITICAL"),
            _o(1, "A", "DEGRADED"),
        ],
    )
    n = m.refine_from_backtester(bt)
    # consecutive pairs: (H,S) on A; outcome pairs: (H,D) and (S,C) on A, (D,F) on B;
    # (S,BOGUS) pred pair and (BOGUS,D) outcome pair are rejected by validation
    assert n == 4
    assert m.base_chain.counts[0][2] == pytest.approx(0.1 + 1.0)  # HEALTHY->STRESSED
    assert m.base_chain.counts[0][1] == pytest.approx(0.4 + 1.0)  # HEALTHY->DEGRADED
    assert m.base_chain.counts[2][3] == pytest.approx(2.5 + 1.0)  # STRESSED->CRITICAL
    assert m.base_chain.counts[1][4] == pytest.approx(0.0 + 1.0)  # DEGRADED->FAILED
    assert m.base_chain.counts[3][4] == pytest.approx(1.5)  # untouched
    assert m._backtester_refined is True
    # outcome exactly at the prediction timestamp counts as "later"
    m2 = _fresh()
    bt2 = SimpleNamespace(
        _predictions=[_p(5, "A", "HEALTHY")],
        _outcomes=[_o(5, "A", "DEGRADED")],
    )
    assert m2.refine_from_backtester(bt2) == 1
    assert m2.base_chain.counts[0][1] == pytest.approx(1.4)
    # invalid outcome state is rejected by the guard, not recorded
    m3 = _fresh()
    bt3 = SimpleNamespace(
        _predictions=[_p(5, "A", "HEALTHY")],
        _outcomes=[_o(6, "A", "BOGUS")],
    )
    assert m3.refine_from_backtester(bt3) == 0
    assert m3.base_chain.counts[0] == [pytest.approx(v) for v in [9.5, 0.4, 0.1, 0, 0, 0]]


def test_refine_breaks_neither_on_outcomeless_first_pod():
    # first pod has no outcomes; second pod's outcome pair must still be recorded
    m = _fresh()
    bt = SimpleNamespace(
        _predictions=[_p(1, "A", "HEALTHY"), _p(2, "A", "DEGRADED"), _p(3, "B", "STRESSED")],
        _outcomes=[_o(9, "B", "CRITICAL")],
    )
    n = m.refine_from_backtester(bt)
    assert n == 2  # (HEALTHY, DEGRADED) pair + (STRESSED, CRITICAL) outcome pair
    assert m.base_chain.counts[0][1] == pytest.approx(1.4)
    assert m.base_chain.counts[2][3] == pytest.approx(3.5)


# ── value iteration reference ────────────────────────────────────────────


def test_value_iteration_matches_reference_solution():
    m = _fresh()
    v = ValueIterationPolicy(model=m)
    V, policy = _ref_value_iteration(m)
    for s in MarkovChain.STATES:
        assert v.V[s] == pytest.approx(V[s], abs=1e-9)
        assert v.policy[s] == policy[s]
        assert v.action(s) == policy[s]
        assert v.value(s) == pytest.approx(V[s], abs=1e-9)
    assert v.action("BOGUS") == "none"
    assert v.value("BOGUS") == 0.0
    # explicit model is used, not silently replaced by a fresh default
    m2 = _fresh()
    assert ValueIterationPolicy(model=m2).model is m2


def test_value_iteration_plan_rollout_exact():
    m = _fresh()
    v = ValueIterationPolicy(model=m)
    plan = v.plan("FAILED", horizon=1)
    assert plan.current_state == "FAILED"
    assert plan.horizon == 1
    s = plan.steps[0]
    assert s.step == 0
    assert s.action == v.action("FAILED") == "pod_restart"
    assert s.expected_failed_prob_before == pytest.approx(1.0)
    assert s.expected_failed_prob_after == 0.5
    assert s.expected_failed_interval_after == (
        0.5,
        pytest.approx(0.2751725808068495),
        pytest.approx(0.7248274191931505),
    )
    assert s.expected_state_distribution == {
        "HEALTHY": 0.0,
        "DEGRADED": 0.0,
        "STRESSED": 0.0,
        "CRITICAL": 0.0,
        "FAILED": 0.5,
        "RECOVERED": 0.5,
    }
    # rollout from STRESSED: distribution is rounded to 4 decimals (contract)
    plan2 = v.plan("STRESSED", horizon=1)
    assert plan2.steps[0].expected_state_distribution == {
        "HEALTHY": 0.2308,
        "DEGRADED": 0.2308,
        "STRESSED": 0.3077,
        "CRITICAL": 0.1923,
        "FAILED": 0.0308,
        "RECOVERED": 0.0077,
    }


# ── transition model units ────────────────────────────────────────────────


def test_record_transition_exact_counts_and_unknown_state_routing():
    m = _fresh()
    m.record_transition("STRESSED", "RECOVERED", "pod_restart")
    m.record_transition("BOGUS", "RECOVERED", "pod_restart")  # unknown from -> 0
    m.record_transition("STRESSED", "BOGUS", "pod_restart")  # unknown to -> 0
    mat = m.action_counts["pod_restart"]
    assert len(mat) == 6 and all(len(r) == 6 for r in mat)
    assert mat[2][5] == 1.0
    assert mat[0][5] == 1.0
    assert mat[2][0] == 1.0
    assert mat[3][4] == 0.0


def test_action_persistence_roundtrip_and_defaults():
    m = _fresh()
    m.record_transition("CRITICAL", "RECOVERED", "scale")
    d = m.to_action_dict()
    assert d["backtester_refined"] is False  # default when attr never set
    assert set(d["action_counts"]) == {"scale"}
    m._backtester_refined = True
    assert m.to_action_dict()["backtester_refined"] is True
    m2 = _fresh()
    m2.load_action_counts(m.to_action_dict())
    assert m2.action_counts == m.action_counts
    assert m2._backtester_refined is True
    # sparse payload keeps defaults for the missing key
    m3 = _fresh()
    m3.load_action_counts({"backtester_refined": True})
    assert m3.action_counts == {}
    assert m3._backtester_refined is True
    # payload without the refined flag defaults it to False
    m4 = _fresh()
    m4.load_action_counts({"action_counts": {}})
    assert m4._backtester_refined is False
    # empty payload is a no-op
    m5 = _fresh()
    m5.load_action_counts({})
    assert m5.action_counts == {}


def test_raw_counts_merge_priors_and_learned_override():
    m = _fresh()
    base = [row[:] for row in m.base_chain.counts]
    assert m._raw_counts("none") == base
    assert m._raw_counts("") == base
    assert m._raw_counts("bogus") == base  # unknown action falls back to base chain
    merged = m._raw_counts("pod_restart")
    assert merged[3] == [
        pytest.approx(b + p) for b, p in zip(base[3], [0, 0, 0, 0, 0, 5])
    ]
    m.record_transition("CRITICAL", "RECOVERED", "pod_restart")
    assert m._raw_counts("pod_restart")[3] == [0.0] * 5 + [1.0]
    # zero-matrix learned counts are ignored: priors still merge in
    m2 = _fresh()
    m2.action_counts["scale"] = [[0.0] * 6 for _ in range(6)]
    zero_merged = m2._raw_counts("scale")
    assert zero_merged[3] == [
        pytest.approx(b + p) for b, p in zip(base[3], [1, 1, 0, 0, 0, 0])
    ]


def test_refine_base_from_pairs_validates_and_routes():
    m = _fresh()
    m.refine_base_from_pairs(
        [("HEALTHY", "DEGRADED"), ("BOGUS", "FAILED"), ("HEALTHY", "BOGUS")]
    )
    assert m.base_chain.counts[0][1] == pytest.approx(0.4 + 1.0)
    assert m.base_chain.counts[0][0] == pytest.approx(9.5)  # unknown-to rejected
    assert m.base_chain.counts[0][4] == pytest.approx(0.0)  # unknown-from rejected


def test_transition_matrix_normalises_and_uniform_zero_row():
    m = _fresh()
    m.base_chain.counts[5] = [0.0] * 6
    tm = m.transition_matrix("none")
    assert len(tm) == 6 and all(len(r) == 6 for r in tm)
    assert all(abs(sum(r) - 1.0) < 1e-9 for r in tm[:5])
    assert tm[5] == [pytest.approx(1 / 6)] * 6


def test_step_distribution_partial_dict_defaults_to_zero():
    d = _fresh().step_distribution({"FAILED": 1.0}, "none")
    assert d == {
        "HEALTHY": 0.0,
        "DEGRADED": 0.0,
        "STRESSED": 0.0,
        "CRITICAL": 0.0,
        "FAILED": 0.8,
        "RECOVERED": 0.2,
    }


def test_failed_probability_missing_key_is_zero():
    assert _fresh().failed_probability({"HEALTHY": 1.0}) == 0.0
    assert _fresh().failed_probability({"FAILED": 0.5}) == 0.5


# ── planner cost / cooldown units ────────────────────────────────────────


def test_planner_cost_fallback_and_cooldown_boundary():
    p = Planner(model=_fresh())
    assert p.cost_weight == 0.02
    assert p._action_cost("none") == 0.0
    assert p._action_cost("bogus") == 0.02  # unknown action -> cost_weight fallback
    assert p._action_allowed("none", 12345.0) is True
    p.cooldowns["pod_restart"] = 100.0
    assert p._action_allowed("pod_restart", 400.0) is True  # exactly 300s boundary
    assert p._action_allowed("pod_restart", 399.999) is False


def test_step_rationale_exact_strings():
    assert _step_rationale(0, "none", "HEALTHY", 0.25, 0.2, 0.05, 1.0) == (
        "Step 0: stay put from HEALTHY (expected P(FAILED)=25.0%); intervening "
        "is not worthwhile — the cost of acting (0.05) outweighs the failure-risk reduction."
    )
    assert _step_rationale(1, "scale", "CRITICAL", 0.5, 0.2, 0.02, 1.0) == (
        "Step 1: apply scale from CRITICAL — expected P(FAILED) 50.0% -> 20.0% "
        "(reduces failure risk by 30.0%); action cost 0.02 vs failure cost 1.00."
    )
    assert _step_rationale(2, "scale", "CRITICAL", 1e-6, 0.0, 0.02, 1.0) == (
        "Step 2: apply scale from CRITICAL — expected P(FAILED) stays ~0.0% "
        "(no further reduction expected); action cost 0.02."
    )
    assert _step_rationale(3, "scale", "CRITICAL", 1.5, 0.5, 0.02, 1.0) == (
        "Step 3: apply scale from CRITICAL — expected P(FAILED) 150.0% -> 50.0% "
        "(reduces failure risk by 100.0%); action cost 0.02 vs failure cost 1.00."
    )


# ── advisory_plan wiring ─────────────────────────────────────────────────


def test_advisory_plan_uses_given_chain_and_cooldowns():
    chain = MarkovChain()
    chain.counts[3] = [0.0] * 6  # CRITICAL self-absorbing with zero failure mass
    d = advisory_plan("CRITICAL", horizon=1, base_chain=chain)
    assert d["current_state"] == "CRITICAL"
    assert d["horizon"] == 1
    assert d["steps"][0]["action"] == "scale"
    # custom chain: scale only adds prior counts -> zero failure mass after
    assert d["steps"][0]["expected_failed_prob_after"] == 0.0
    assert d["steps"][0]["expected_failed_prob_before"] == 0.0
    # cooldowns block every real action from FAILED -> none is chosen
    d2 = advisory_plan(
        "FAILED",
        horizon=1,
        cooldowns={a: time.time() for a in ACTION_TYPES if a != "none"},
    )
    assert d2["steps"][0]["action"] == "none"


def test_planner_blocked_action_does_not_abort_action_scan():
    # pod_restart blocked: rollout_restart (equal prior, equal fail prob) must
    # still be evaluated after the blocked entry
    p = Planner(model=_fresh(), cooldowns={"pod_restart": time.time()})
    assert p.plan("FAILED", horizon=1).steps[0].action == "rollout_restart"


def test_action_aware_model_falls_back_to_base_chain():
    chain = MarkovChain()
    model = ActionAwareTransitionModel(base_chain=chain)
    # With no learned action data, transition matrix equals the base chain.
    base = chain.transition_matrix()
    none_mat = model.transition_matrix("none")
    for i in range(6):
        for j in range(6):
            assert abs(base[i][j] - none_mat[i][j]) < 1e-9


def test_action_prior_changes_transition():
    chain = MarkovChain()
    model = ActionAwareTransitionModel(base_chain=chain)
    # From CRITICAL, pod_restart should increase probability of RECOVERED.
    before = model.transition_matrix("none")[MarkovChain.STATE_IDX["CRITICAL"]]
    after = model.transition_matrix("pod_restart")[MarkovChain.STATE_IDX["CRITICAL"]]
    recovered_idx = MarkovChain.STATE_IDX["RECOVERED"]
    assert after[recovered_idx] > before[recovered_idx]


def test_record_transition_learns_action_model():
    model = ActionAwareTransitionModel()
    # Observe that pod_restart from FAILED reliably leads to RECOVERED.
    for _ in range(20):
        model.record_transition("FAILED", "RECOVERED", "pod_restart")
    mat = model.transition_matrix("pod_restart")
    row = mat[MarkovChain.STATE_IDX["FAILED"]]
    recovered_idx = MarkovChain.STATE_IDX["RECOVERED"]
    # Learned count dominates the prior -> near-deterministic RECOVERED.
    assert row[recovered_idx] > 0.9


def test_planner_reduces_failure_probability():
    model = ActionAwareTransitionModel()
    planner = Planner(model=model)
    # Do-nothing baseline: apply "none" for the full horizon.
    def _none_baseline(state, horizon):
        dist = {s: 0.0 for s in MarkovChain.STATES}
        dist[state] = 1.0
        for _ in range(horizon):
            dist = model.step_distribution(dist, "none")
        return model.failed_probability(dist)

    for state in ("DEGRADED", "STRESSED", "CRITICAL", "FAILED"):
        # Fresh planner per state: a shared planner would carry cooldowns from
        # earlier states into later ones (production uses one planner per
        # advisory_plan() call, which is state-independent).
        plan = Planner(model=model).plan(state, horizon=3)
        actions = [s.action for s in plan.steps]
        # Cost model: act only when failure risk outweighs the action cost.
        # Watchful waiting is correct from low-risk states ...
        if state in ("DEGRADED", "STRESSED"):
            assert all(a == "none" for a in actions), f"costly churn from {state}"
        else:
            # ... but CRITICAL/FAILED carry near-certain failure cost, so the
            # planner must intervene.
            assert any(a != "none" for a in actions), f"no intervention from {state}"
        # Expected P(FAILED) at the end must be <= the do-nothing baseline.
        last = plan.steps[-1].expected_failed_prob_after
        baseline = _none_baseline(state, 3)
        # The cost-aware planner never does worse than doing nothing (it can
        # always pick "none", which has zero cost).
        assert last <= baseline + 1e-9, (state, last, baseline)
    # From a clearly-at-risk state the planner intervenes (reduces failure).
    critical = planner.plan("CRITICAL", horizon=3)
    assert any(a != "none" for a in critical.steps)


def test_planner_respects_cooldowns():
    planner = Planner(cooldowns={"pod_restart": __import__("time").time()})
    plan = planner.plan("CRITICAL", horizon=3)
    # pod_restart was just used (cooldown) -> should not be the chosen action.
    assert all(s.action != "pod_restart" for s in plan.steps)


def test_advisory_plan_dict_shape():
    result = advisory_plan("STRESSED", horizon=2)
    assert result["execution_ready"] is True
    assert result["current_state"] == "STRESSED"
    assert len(result["steps"]) == 2
    for step in result["steps"]:
        assert "action" in step
        assert "expected_failed_prob_after" in step


# ---------------------------------------------------------------------------
# RemediationManager integration (decision model persistence + learning)
# ---------------------------------------------------------------------------
from predictive_agent.remediator import RemediationManager


def _fake_backtester():
    """Minimal backtester stub exposing the attributes refine_from_backtester reads."""
    class _Rec:
        def __init__(self, pk, ts, ms):
            self.pod_key, self.timestamp, self.markov_state = pk, ts, ms
    class _Out:
        def __init__(self, pk, ts, st):
            self.pod_key, self.timestamp, self.state = pk, ts, st
    class _BT:
        _predictions = [
            _Rec("a", 1.0, "STRESSED"),
            _Rec("a", 2.0, "CRITICAL"),
            _Rec("a", 3.0, "FAILED"),
        ]
        _outcomes = [_Out("a", 3.5, "FAILED")]
    return _BT()


def test_manager_plan_remediation_preview():
    rem = RemediationManager(dry_run=True)
    plan = rem.plan_remediation("CRITICAL", horizon=3)
    assert plan["execution_ready"] is True
    assert any(s["action"] != "none" for s in plan["steps"])


def test_feed_transition_learns_action_effect():
    rem = RemediationManager(dry_run=True)
    rem.feed_transition("FAILED", "pod_restart", "RECOVERED")
    mat = rem.decision_model.transition_matrix("pod_restart")
    fi = MarkovChain.STATE_IDX["FAILED"]
    ri = MarkovChain.STATE_IDX["RECOVERED"]
    # One observed FAILED->RECOVERED makes the learned matrix deterministic.
    assert mat[fi][ri] > 0.9


def test_refine_from_backtester_is_idempotent():
    rem = RemediationManager(dry_run=True)
    bt = _fake_backtester()
    n = rem.refine_from_backtester(bt)
    assert n > 0
    assert rem.decision_model._backtester_refined is True
    # Second call must not double-count.
    assert rem.refine_from_backtester(bt) == 0


# ---------------------------------------------------------------------------
# Closed learning loop (reconcile integration)
# ---------------------------------------------------------------------------
from predictive_agent.persistence import StateStore
from predictive_agent.remediator import (
    RemediationAction,
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


def _fake_manager():
    rem = RemediationManager(dry_run=False)
    rem.register_action(_FakeAction())
    rem.decision_model = ActionAwareTransitionModel(base_chain=MarkovChain())
    return rem


def test_learning_loop_records_and_learns():
    rem = _fake_manager()
    # Cycle 1: high-risk pod at CRITICAL -> remediation executes pod_restart.
    rem.evaluate(_Pod("CRITICAL"), None, 0.95)
    assert rem._pending_feedback, "executed action should be pending feedback"
    # Cycle 2: pod is now RECOVERED -> outcome is learned.
    rem.record_observations({"ns/pod": "RECOVERED"})
    assert rem._pending_feedback == {}
    mat = rem.decision_model.transition_matrix("pod_restart")
    ci = MarkovChain.STATE_IDX["CRITICAL"]
    ri = MarkovChain.STATE_IDX["RECOVERED"]
    assert mat[ci][ri] > 0.9


def test_decision_model_persistence_roundtrip():
    import os
    import tempfile
    rem = _fake_manager()
    rem.evaluate(_Pod("CRITICAL"), None, 0.95)
    rem.record_observations({"ns/pod": "RECOVERED"})
    tmp = tempfile.mkdtemp()
    store = StateStore(
        state_model_file=os.path.join(tmp, "state.json"),
        predictions_file=os.path.join(tmp, "pred.json"),
    )
    store.save_decision_model(rem.decision_model.to_action_dict())
    loaded = store.load_decision_model()
    assert loaded and "action_counts" in loaded
    dm2 = ActionAwareTransitionModel(base_chain=MarkovChain())
    dm2.load_action_counts(loaded)
    ci = MarkovChain.STATE_IDX["CRITICAL"]
    ri = MarkovChain.STATE_IDX["RECOVERED"]
    assert dm2.transition_matrix("pod_restart")[ci][ri] > 0.9


def _make_action(execute_flag, should_exec):
    class _A(RemediationAction):
        name = "pod_restart"

        def __init__(self):
            self.executed = []
            self._should = should_exec

        def should_execute(self, pod_state, prediction, risk_score):
            return self._should

        def execute(self, target, context):
            self.executed.append(target)
            return RemediationResult(
                action=self.name, target=target, success=True,
                dry_run=context.dry_run, message="ok",
            )
    return _A()


def test_plan_and_execute_respects_should_execute():
    # Eligible action is executed by the planner from a degraded state.
    a = _make_action(True, True)
    rem = RemediationManager(dry_run=False)
    rem.register_action(a)
    rem.decision_model = ActionAwareTransitionModel(base_chain=MarkovChain())
    rem.plan_and_execute("CRITICAL", "ns/pod", _Pod("CRITICAL"), None, 0.95, horizon=3)
    assert a.executed, "planner should execute an eligible action"

    # Ineligible action must NOT be executed (safety gate).
    b = _make_action(True, False)
    rem2 = RemediationManager(dry_run=False)
    rem2.register_action(b)
    rem2.decision_model = ActionAwareTransitionModel(base_chain=MarkovChain())
    results = rem2.plan_and_execute(
        "CRITICAL", "ns/pod", _Pod("CRITICAL"), None, 0.95, horizon=3
    )
    assert not b.executed, "planner must not execute an ineligible action"
    assert any("ineligible" in r.message for r in results)


def test_predictive_interval_shape_and_narrows_with_data():
    model = ActionAwareTransitionModel(base_chain=MarkovChain())
    dist = {"CRITICAL": 1.0}
    mean, lo, hi = model.predictive_interval(dist, "none")
    assert 0.0 <= lo <= mean <= hi <= 1.0
    wide = hi - lo
    # More observed transitions -> tighter (calibrated) uncertainty.
    for _ in range(60):
        model.record_transition("CRITICAL", "RECOVERED", "pod_restart")
    mean2, lo2, hi2 = model.predictive_interval(dist, "pod_restart")
    assert 0.0 <= lo2 <= mean2 <= hi2 <= 1.0
    assert (hi2 - lo2) < wide


def test_decision_theoretic_cost_tradeoff():
    """Planner intervenes only when failure-reduction outweighs action cost (#6)."""
    ACTIONS = [
        "pod_restart", "rollout_restart", "node_cordon", "scale",
        "right_size", "tune_resources",
    ]
    # Model where pod_restart reliably recovers CRITICAL and doing nothing fails.
    m = ActionAwareTransitionModel(base_chain=MarkovChain())
    for _ in range(80):
        m.record_transition("CRITICAL", "RECOVERED", "pod_restart")
    for _ in range(80):
        m.record_transition("CRITICAL", "FAILED", "none")

    plan_default = Planner(model=m).plan("CRITICAL", 3)
    assert plan_default.steps[0].action == "pod_restart"

    # Disproportionate action costs -> accept the failure risk, choose "none".
    expensive = Planner(
        model=m, action_costs={a: 10.0 for a in ACTIONS}
    )
    plan_expensive = expensive.plan("CRITICAL", 3)
    assert plan_expensive.steps[0].action == "none"
    assert (
        plan_expensive.steps[0].expected_failed_prob_after
        >= plan_default.steps[0].expected_failed_prob_after
    )


def test_value_iteration_policy_reduces_failure():
    """RL/value-iteration stationary policy minimises long-run expected cost (#8)."""
    m = ActionAwareTransitionModel(base_chain=MarkovChain())
    for _ in range(80):
        m.record_transition("CRITICAL", "RECOVERED", "pod_restart")
    for _ in range(80):
        m.record_transition("CRITICAL", "FAILED", "none")
    pol = ValueIterationPolicy(model=m)
    assert pol.action("CRITICAL") == "pod_restart"

    plan = pol.plan("CRITICAL", 3)
    assert plan.steps[0].action == "pod_restart"

    # Rolling out the stationary policy beats doing nothing over the horizon.
    d = {s: 0.0 for s in MarkovChain.STATES}
    d["CRITICAL"] = 1.0
    none_d = d
    for _ in range(3):
        none_d = m.step_distribution(none_d, "none")
    pol_d = d
    for _ in range(3):
        cur = max(pol_d, key=lambda s: pol_d[s])
        pol_d = m.step_distribution(pol_d, pol.action(cur))
    assert m.failed_probability(pol_d) <= m.failed_probability(none_d)



def test_what_if_exact_payload_and_error_path():
    m = ActionAwareTransitionModel()
    w = m.what_if("CRITICAL", "none", 2)
    assert w["start_state"] == "CRITICAL"
    assert w["action"] == "none"
    assert w["horizon"] == 2
    assert w["steps"] == [
        {
            "step": 1,
            "action": "none",
            "expected_state_distribution": {
                "HEALTHY": 0.05,
                "DEGRADED": 0.05,
                "STRESSED": 0.1,
                "CRITICAL": 0.6,
                "FAILED": 0.15,
                "RECOVERED": 0.05,
            },
            "expected_failed_prob": 0.15,
            "expected_failed_interval": [0.15, 0.0, 0.3708],
        },
        {
            "step": 2,
            "action": "none",
            "expected_state_distribution": {
                "HEALTHY": 0.1425,
                "DEGRADED": 0.0845,
                "STRESSED": 0.1105,
                "CRITICAL": 0.387,
                "FAILED": 0.214,
                "RECOVERED": 0.0615,
            },
            "expected_failed_prob": 0.214,
            "expected_failed_interval": [0.214, 0.0, 0.7284],
        },
    ]
    # default horizon is 3; 1-based step numbering
    w2 = ActionAwareTransitionModel().what_if("FAILED", "pod_restart")
    assert w2["horizon"] == 3 and len(w2["steps"]) == 3
    assert w2["steps"][-1]["step"] == 3
    assert w2["steps"][-1]["action"] == "pod_restart"
    assert w2["steps"][-1]["expected_failed_interval"] == [0.1258, 0.0, 0.559]
    # default action is "none" (echoed in the payload)
    w0 = ActionAwareTransitionModel().what_if("CRITICAL")
    assert w0["action"] == "none"
    assert w0["steps"][0]["action"] == "none"
    # step-2 values carry >4 decimals: rounding to 4 is part of the contract
    w3 = ActionAwareTransitionModel().what_if("CRITICAL", "pod_restart", 2)
    assert w3["steps"][1]["expected_state_distribution"] == {
        "HEALTHY": 0.3518,
        "DEGRADED": 0.0938,
        "STRESSED": 0.07,
        "CRITICAL": 0.1739,
        "FAILED": 0.0921,
        "RECOVERED": 0.2184,
    }
    assert w3["steps"][1]["expected_failed_prob"] == 0.0921
    # unknown start state -> error payload, no steps
    assert m.what_if("BOGUS") == {"error": "invalid state 'BOGUS'"}


def test_refine_action_from_triples_routes_and_validates():
    m = ActionAwareTransitionModel()
    m.refine_action_from_triples(
        [
            ("STRESSED", "pod_restart", "RECOVERED"),
            ("BOGUS", "scale", "RECOVERED"),
            ("STRESSED", "scale", "BOGUS"),
        ]
    )
    assert m.action_counts["pod_restart"][2][5] == 1.0
    assert "scale" not in m.action_counts


def test_support_metrics_and_confidence_gate():
    m = ActionAwareTransitionModel()
    assert m.total_support() == 0.0
    assert not m.is_confident("CRITICAL", 0.1)
    assert m.is_confident("CRITICAL", 0)
    assert m.is_confident("CRITICAL", -1.0)
    m.record_transition("STRESSED", "RECOVERED", "pod_restart")
    m.record_transition("STRESSED", "FAILED", "scale")
    m.record_transition("HEALTHY", "DEGRADED", "scale")
    assert m.total_support() == 3.0
    assert m.state_support("STRESSED") == 2.0
    # unknown state falls back to the HEALTHY row
    assert m.state_support("BOGUS") == 1.0
    assert not m.is_confident("CRITICAL", 2.0)
    assert m.is_confident("STRESSED", 2.0)
    assert not m.is_confident("STRESSED", 2.0001)


def test_decay_scales_counts_and_reports_removed_weight():
    m = ActionAwareTransitionModel()
    m.record_transition("CRITICAL", "RECOVERED", "scale")
    m.record_transition("STRESSED", "FAILED", "pod_restart")
    assert m.decay() == pytest.approx(0.02)
    assert m.action_counts["scale"][3][5] == pytest.approx(0.99)
    assert m.action_counts["pod_restart"][2][4] == pytest.approx(0.99)
    assert m.decay(1.0) == 0.0
    assert m.action_counts["scale"][3][5] == pytest.approx(0.99)
    # alpha > 1 is a no-op (no growth)
    assert m.decay(1.5) == 0.0
    assert m.action_counts["scale"][3][5] == pytest.approx(0.99)
    assert m.decay(0.5) == pytest.approx(0.99)
    assert m.action_counts["scale"][3][5] == pytest.approx(0.495)
    # sub-floor residue is floored to 0.0 in the stored counts
    m.action_counts["scale"][3][5] = 1.5e-06
    m.decay(0.5)
    assert m.action_counts["scale"][3][5] == 0.0
    # exact floor boundary: 2e-6 * 0.5 == 1e-6 stays (>= floor, not >)
    m.action_counts["scale"][3][5] = 2e-06
    m.decay(0.5)
    assert m.action_counts["scale"][3][5] == 1e-06
    # remaining pod_restart + floor-cell weight still decays
    assert m.decay() == pytest.approx(0.12375 * 0.01 + 1e-06 * 0.01)
