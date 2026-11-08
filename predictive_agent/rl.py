"""Model-free sampled RL: tabular Q-learning from real outcomes (#8).

Complements the model-based DP planner (:class:`~predictive_agent.pomdp.Planner`)
and exact value iteration (:class:`~predictive_agent.pomdp.ValueIterationPolicy`).
Instead of learning transition dynamics, it learns ``Q(state, action)`` directly
from observed ``(state, action, next_state)`` outcomes fed by the same reconcile
learning loop that trains the transition model. Reward is the decision-theoretic
objective already used elsewhere — ``-failure_cost * P(FAILED) - action_cost`` —
so the Q policy converges to the same optimum while requiring no model of the
environment. It is useful as a robust, model-free cross-check and as an
exploration-augmented action selector.
"""

import logging
import random
from typing import Dict, List, Optional

from predictive_agent.markov import MarkovChain as _MC  # noqa: F401 - re-export convenience
from predictive_agent.pomdp import (
    ACTION_TYPES,
    DEFAULT_ACTION_COSTS,
    FAILURE_COST,
    MarkovChain,
)

logger = logging.getLogger(__name__)

FAILURE_STATES = ("CRITICAL", "FAILED")


class QLearningPolicy:
    """Tabular Q-learning policy over the 6-state action-MDP.

    Learns from real outcomes via :meth:`update`; selects actions via
    :meth:`action` (greedy, with optional epsilon exploration). Rolls out over a
    horizon using a supplied transition ``model`` to project next states (a
    hybrid: model-free value, model-based projection).
    """

    def __init__(
        self,
        model=None,
        actions: Optional[List[str]] = None,
        action_costs: Optional[dict] = None,
        failure_cost: float = FAILURE_COST,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.0,
    ):
        self.model = model
        self.actions = actions or list(ACTION_TYPES)
        self.action_costs = action_costs or dict(DEFAULT_ACTION_COSTS)
        self.failure_cost = failure_cost
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Pessimistic prior: an untried remediation action is assumed to lead to
        # failure (Q = -failure_cost); "none" starts neutral (0). This keeps the
        # policy conservative on a live cluster — it only chooses an action once
        # observed outcomes show it beats doing nothing. Exploration is meant to
        # happen off-policy (shadow mode), not on real incidents.
        self.Q: Dict[str, Dict[str, float]] = {
            s: {a: 0.0 if a == "none" else -self.failure_cost for a in self.actions}
            for s in MarkovChain.STATES
        }

    # ─── learning ───────────────────────────────────────────────────────────
    def _action_cost(self, action: str) -> float:
        if action == "none":
            return 0.0
        return self.action_costs.get(action, 0.0)

    def _reward(self, next_state: str, action: str) -> float:
        fail = 1.0 if next_state in FAILURE_STATES else 0.0
        return -self.failure_cost * fail - self._action_cost(action)

    def update(self, state: str, action: str, next_state: str) -> None:
        """One Q-learning step on an observed (s, a, s') outcome."""
        if (
            state not in self.Q
            or action not in self.Q[state]
            or next_state not in self.Q
        ):
            return
        best_next = max(self.actions, key=lambda a: self.Q[next_state][a])
        target = self._reward(next_state, action) + self.gamma * self.Q[next_state][best_next]
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])

    # ─── acting ──────────────────────────────────────────────────────────────
    def action(self, state: str) -> str:
        """Greedy (epsilon-exploration) action for ``state``."""
        if state not in self.Q:
            return "none"
        if self.epsilon > 0 and random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.actions, key=lambda a: self.Q[state][a])

    def plan(self, state: str, horizon: int = 3, model=None):
        """Roll out the greedy Q policy over ``horizon`` steps.

        Returns ``(actions, expected_p_failed)``; next states are projected with
        ``model`` (or ``self.model``). Useful as a 5th benchmark strategy and a
        ``/plan?strategy=q`` preview.
        """
        mdl = model or self.model
        dist = {s: 0.0 for s in MarkovChain.STATES}
        dist[state] = 1.0
        actions: List[str] = []
        for _ in range(horizon):
            cur = max(dist, key=lambda s: dist[s])
            a = self.action(cur)
            if a != "none":
                actions.append(a)
            if mdl is not None:
                dist = mdl.step_distribution(dist, a)
            else:
                break
        pf = mdl.failed_probability(dist) if mdl is not None else 0.0
        return actions, pf

    # ─── persistence ──────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {s: dict(self.Q[s]) for s in self.Q}

    def load(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        for s in self.Q:
            row = data.get(s, {})
            if not isinstance(row, dict):
                continue
            for a in self.Q[s]:
                if a in row:
                    try:
                        self.Q[s][a] = float(row[a])
                    except (TypeError, ValueError):
                        continue
