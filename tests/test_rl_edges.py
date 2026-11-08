"""Edge-case tests for predictive_agent.rl QLearningPolicy.

Covers the persistence and bound/decay paths that a happy-path Q-learning
suite does not hit:

1. to_dict() -> load() round-trip preserves Q-values for every (state, action);
2. load(None) / load({}) / malformed payloads leave the policy usable;
3. update() with unknown states/actions/next-states is a safe no-op;
4. epsilon action-selection stays within the action set across many seeded
   draws;
5. the discount path drives Q-values down monotonically under repeated
   failures, and plan() rolls out without a model.

Written against the REAL QLearningPolicy API (update/to_dict/load).
"""
import json
import random

import pytest

from predictive_agent.pomdp import ACTION_TYPES, MarkovChain
from predictive_agent.rl import FAILURE_COST, QLearningPolicy

ACTIONS = list(ACTION_TYPES)
STATES = list(MarkovChain.STATES)


def _policy(**kwargs) -> QLearningPolicy:
    """Fresh policy with the global RNG seeded for reproducible draws."""
    random.seed(kwargs.pop("seed", 42))
    return QLearningPolicy(**kwargs)


class TestSaveLoadPersistence:
    def test_save_load_roundtrip_preserves_all_q_values(self):
        """to_dict() then load() restores every (state, action) Q-value exactly."""
        q = _policy()
        for state in ("CRITICAL", "STRESSED", "HEALTHY"):
            for action in ("pod_restart", "rollout_restart", "scale", "right_size"):
                q.update(state, action, "RECOVERED")
                q.update(state, action, "FAILED")
        # a net-positive outcome so some values drift above the pessimist prior
        q.update("CRITICAL", "pod_restart", "RECOVERED")

        restored = QLearningPolicy()
        restored.load(q.to_dict())

        for state in STATES:
            for action in ACTIONS:
                assert restored.Q[state][action] == pytest.approx(q.Q[state][action])

    def test_snapshot_is_json_serialisable(self):
        """to_dict() output survives a JSON round-trip (as a PVC blob would)."""
        q = _policy()
        q.update("CRITICAL", "pod_restart", "FAILED")
        q.update("STRESSED", "scale", "RECOVERED")
        q.update("CRITICAL", "node_cordon", "RECOVERED")

        blob = json.dumps(q.to_dict())
        restored = QLearningPolicy()
        restored.load(json.loads(blob))

        assert restored.Q["CRITICAL"]["pod_restart"] == pytest.approx(q.Q["CRITICAL"]["pod_restart"])
        assert restored.Q["STRESSED"]["scale"] == pytest.approx(q.Q["STRESSED"]["scale"])
        assert restored.Q["CRITICAL"]["node_cordon"] == pytest.approx(q.Q["CRITICAL"]["node_cordon"])

    def test_load_none_or_empty_dict_leaves_defaults(self):
        """load(None) / load({}) keep default priors and a usable policy."""
        q = _policy()
        for payload in (None, {}):
            q.load(payload)  # must not raise
            for state in STATES:
                for action in ACTIONS:
                    expected = 0.0 if action == "none" else -FAILURE_COST
                    assert q.Q[state][action] == pytest.approx(expected)

        # still fully usable afterwards
        q.update("CRITICAL", "pod_restart", "RECOVERED")
        assert q.action("CRITICAL") in ACTIONS

    def test_load_malformed_payloads_do_not_corrupt_policy(self):
        """Corrupt load() data is tolerated; subsequent calls keep working."""
        q = _policy()
        q.update("CRITICAL", "pod_restart", "RECOVERED")
        good_value = q.Q["CRITICAL"]["pod_restart"]

        payloads = [
            "not-a-dict",
            123,
            ["CRITICAL", {"pod_restart": 1.0}],
            {"CRITICAL": "not-a-row"},
            {"CRITICAL": {"pod_restart": "not-a-number"}},
            {"CRITICAL": {"pod_restart": None}},
            {"CRITICAL": {"pod_restart": [1, 2]}},
            {"CRITICAL": {"bogus": object()}},
        ]
        for payload in payloads:
            q.load(payload)  # must not raise and must not hang

        # malformed entries were skipped — the good value is untouched
        assert q.Q["CRITICAL"]["pod_restart"] == pytest.approx(good_value)

        # …and the policy keeps working
        q.update("CRITICAL", "pod_restart", "FAILED")
        q.update("CRITICAL", "bogus", "RECOVERED")
        assert q.action("STRESSED") in ACTIONS


class TestUpdateEdges:
    def test_unknown_state_is_a_no_op(self):
        """update() with an unknown state neither raises nor corrupts the table."""
        q = _policy()
        before = json.dumps(q.to_dict(), sort_keys=True)
        q.update("EVICTED", "pod_restart", "RECOVERED")
        assert json.dumps(q.to_dict(), sort_keys=True) == before
        assert "EVICTED" not in q.Q

    def test_unknown_action_is_a_no_op(self):
        """update() with an unknown action leaves every Q-value untouched."""
        q = _policy()
        before = json.dumps(q.to_dict(), sort_keys=True)
        q.update("CRITICAL", "custom_cryogenic_freeze", "RECOVERED")
        assert json.dumps(q.to_dict(), sort_keys=True) == before

    def test_unknown_next_state_is_a_no_op(self):
        """update() with an unseen next-state does not raise (guarded lookup)."""
        q = _policy()
        q.update("CRITICAL", "pod_restart", "MYSTERY")  # must not raise
        # nothing was learned for the attempted pair
        assert q.Q["CRITICAL"]["pod_restart"] == pytest.approx(-FAILURE_COST)

    def test_pessimist_prior_holds_for_fresh_policy(self):
        """Untried actions start at -failure_cost; 'none' starts neutral."""
        q = _policy()
        for state in STATES:
            for action in ACTIONS:
                expected = 0.0 if action == "none" else -FAILURE_COST
                assert q.Q[state][action] == pytest.approx(expected)


class TestEpsilonBounds:
    def test_epsilon_action_selection_stays_within_action_set(self):
        """Seeded epsilon-greedy draws never escape the action set."""
        q = _policy(epsilon=0.5)
        for _ in range(3):
            q.update("CRITICAL", "pod_restart", "RECOVERED")
        chosen = {q.action(s) for _ in range(2000) for s in ("CRITICAL", "MYSTERY_STATE")}
        assert chosen  # non-empty
        assert chosen <= set(ACTIONS)  # never out of bounds

    def test_full_exploration_covers_every_action(self):
        """At epsilon=1.0, seeded draws eventually sample every action."""
        q = _policy(epsilon=1.0, seed=7)
        drawn = {q.action("CRITICAL") for _ in range(5000)}
        assert drawn == set(ACTIONS)
        assert "none" in drawn

    def test_greedy_epsilon_zero_is_deterministic(self):
        """At epsilon=0.0, selection is the fixed argmax over the Q-row."""
        q = _policy(epsilon=0.0)
        q.update("CRITICAL", "pod_restart", "RECOVERED")
        q.update("CRITICAL", "pod_restart", "RECOVERED")
        first = q.action("CRITICAL")
        for _ in range(50):
            assert q.action("CRITICAL") == first


class TestDiscountAndPlan:
    def test_discount_path_reduces_q_values_monotonically(self):
        """A fixed negative-outcome fixture drives Q down monotonically."""
        q = QLearningPolicy(actions=["none", "pod_restart"], alpha=0.5, gamma=0.9)
        values = [q.Q["CRITICAL"]["pod_restart"]]
        for _ in range(20):
            q.update("CRITICAL", "pod_restart", "FAILED")
            values.append(q.Q["CRITICAL"]["pod_restart"])

        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))
        assert values[-1] < values[0]

    def test_positive_outcome_raises_q_above_prior(self):
        """Repeated RECOVERED outcomes push the action above the pessimist prior."""
        q = _policy(alpha=0.5)
        for _ in range(15):
            q.update("CRITICAL", "pod_restart", "RECOVERED")
        assert q.Q["CRITICAL"]["pod_restart"] > -FAILURE_COST

    def test_plan_without_model_returns_empty(self):
        """plan() with no model cannot project — returns no actions safely."""
        q = _policy()
        actions, pf = q.plan("CRITICAL", horizon=3, model=None)
        assert actions == []
        assert pf == 0.0

    def test_plan_with_model_returns_actions_within_set(self):
        """plan() rollouts only ever propose known actions."""
        from predictive_agent.pomdp import ActionAwareTransitionModel

        model = ActionAwareTransitionModel(base_chain=MarkovChain())
        q = _policy(epsilon=0.0)
        q.update("CRITICAL", "pod_restart", "RECOVERED")
        actions, pf = q.plan("CRITICAL", horizon=3, model=model)
        assert all(a in ACTIONS for a in actions)
        assert 0.0 <= pf <= 1.0
