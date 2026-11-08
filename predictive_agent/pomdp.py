"""POMDP-style decision layer for the predictive agent (advisory).

The operator already maintains a *predictive* 6-state Markov chain
(``predictive_agent.markov.MarkovChain``) and a library of remediation
actions executed by ``predictive_agent.remediator.RemediationManager``. This
module closes the gap the decision-making research corpus highlights: a Markov
chain only *forecasts* state; it does not model how an **action** changes the
transition, nor does it *plan* a sequence of interventions before the
RemediationManager executes them.

Two pieces:

* ``ActionAwareTransitionModel`` — extends the 6-state chain to be
  action-conditioned: ``P(next | state, action)``. Action priors are
  principled pseudo-counts (clearly marked) and are meant to be refined from
  ``predictive_agent.backtester`` data over time.
* ``Planner`` — short-horizon (Scenario-MPC-style) lookahead that picks the
  action sequence minimising the expected probability of ``FAILED`` while
  respecting safety cooldowns. Its output is an **execution-ready** action
  sequence handed to ``RemediationManager`` (which still enforces
  ``SafetyPolicy`` on every step).

Grounded in the dm-research corpus (12,860 papers): POMDP maintenance
planning and POMDP + Scenario-MPC planning under uncertainty.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from predictive_agent.markov import MarkovChain

logger = logging.getLogger(__name__)

# Canonical intervention set, mirroring predictive_agent.actions.__all__.
# "none" is the do-nothing baseline used by the planner to compare against acting.
ACTION_TYPES = [
    "none",
    "pod_restart",
    "rollout_restart",
    "node_cordon",
    "scale",
    "right_size",
    "tune_resources",
]

# Index of the absorbing failure state used as the risk objective.
FAILED_IDX = MarkovChain.STATE_IDX["FAILED"]

# Decision-theoretic cost model: each remediation action carries an estimated
# cost (downtime / churn / blast-radius proxy) and failure carries a unit cost.
# The planner minimises *expected cost* = P(FAILED) * FAILURE_COST + action_cost,
# so it only intervenes when the expected failure reduction outweighs the cost
# of acting. Grounded in the corpus Decision Theory cluster (df 909).
FAILURE_COST = 1.0
DEFAULT_ACTION_COSTS = {
    "pod_restart": 0.05,
    "rollout_restart": 0.05,
    "node_cordon": 0.10,
    "scale": 0.02,
    "right_size": 0.03,
    "tune_resources": 0.03,
}


def _step_rationale(
    step: int,
    action: str,
    state_before: str,
    before: float,
    after: float,
    action_cost: float,
    failure_cost: float,
) -> str:
    """Human-readable explanation of one planner step (option #6)."""
    delta = before - after
    if action == "none":
        return (
            f"Step {step}: stay put from {state_before} "
            f"(expected P(FAILED)={before:.1%}); intervening is not worthwhile "
            f"— the cost of acting ({action_cost:.2f}) outweighs the "
            f"failure-risk reduction."
        )
    if delta > 1e-6:
        return (
            f"Step {step}: apply {action} from {state_before} — "
            f"expected P(FAILED) {before:.1%} -> {after:.1%} "
            f"(reduces failure risk by {delta:.1%}); action cost {action_cost:.2f} "
            f"vs failure cost {failure_cost:.2f}."
        )
    return (
        f"Step {step}: apply {action} from {state_before} — "
        f"expected P(FAILED) stays ~{after:.1%} (no further reduction expected); "
        f"action cost {action_cost:.2f}."
    )


@dataclass
class PlanStep:
    """One step of an advisory remediation plan."""

    step: int
    action: str
    expected_failed_prob_before: float
    expected_failed_prob_after: float
    expected_state_distribution: dict
    expected_failed_interval_after: tuple = (0.0, 0.0, 0.0)
    rationale: str = ""


@dataclass
class Plan:
    """Execution-ready remediation plan over a finite horizon."""

    current_state: str
    horizon: int
    steps: list = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "current_state": self.current_state,
            "horizon": self.horizon,
            "execution_ready": True,
            "steps": [
                {
                    "step": s.step,
                    "action": s.action,
                    "expected_failed_prob_before": round(s.expected_failed_prob_before, 4),
                    "expected_failed_prob_after": round(s.expected_failed_prob_after, 4),
                    "expected_failed_interval_after": [
                        round(x, 4) for x in s.expected_failed_interval_after
                    ],
                    "expected_state_distribution": {
                        k: round(v, 4) for k, v in s.expected_state_distribution.items()
                    },
                    "rationale": s.rationale,
                }
                for s in self.steps
            ],
            "rationale": self.rationale,
        }


class ActionAwareTransitionModel:
    """Action-conditioned 6-state transition model.

    For each action we keep a 6x6 count matrix. When no action-specific data
    has been learned, the model falls back to the base Markov chain matrix,
    so behaviour degrades gracefully to the existing predictor.
    """

    # Action priors: pseudo-counts added on top of the base chain when an
    # action is taken FROM each row-state. They encode the *intended* effect
    # of an intervention and are conservative — real transitions observed via
    # the backtester should dominate after a handful of occurrences.
    #
    # Layout matches MarkovChain.STATES order:
    #   HEALTHY, DEGRADED, STRESSED, CRITICAL, FAILED, RECOVERED
    ACTION_PRIORS = {
        # Restart family: pull CRITICAL/FAILED strongly back toward RECOVERED.
        "pod_restart": [
            [0, 0, 0, 0, 0, 0],       # HEALTHY  (rarely used from here)
            [0, 0, 0, 0, 0, 2],       # DEGRADED
            [0, 0, 0, 0, 0, 3],       # STRESSED
            [0, 0, 0, 0, 0, 5],       # CRITICAL
            [0, 0, 0, 0, 0, 6],       # FAILED
            [0, 0, 0, 0, 0, 0],       # RECOVERED
        ],
        "rollout_restart": [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 6],
            [0, 0, 0, 0, 0, 0],
        ],
        # node_cordon: relieves pressure on a node — nudges STRESSED/CRITICAL
        # back toward DEGRADED/HEALTHY rather than recovering outright.
        "node_cordon": [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        # scale / right_size / tune_resources: ease resource pressure so
        # STRESSED drifts back toward HEALTHY.
        "scale": [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [2, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        "right_size": [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [2, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        "tune_resources": [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [2, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
    }

    def __init__(self, base_chain: Optional[MarkovChain] = None):
        self.base_chain = base_chain or MarkovChain()
        self.action_counts: dict[str, list[list[float]]] = {}

    # -- learning ---------------------------------------------------------
    def record_transition(self, from_state: str, to_state: str, action: str):
        """Record an observed (state, action) -> next-state transition.

        Mirrors ``MarkovChain.record_transition`` but is action-conditioned,
        so the planner improves from real backtested interventions.
        """
        if action not in self.action_counts:
            self.action_counts[action] = [
                [0.0] * 6 for _ in range(6)
            ]
        fi = MarkovChain.STATE_IDX.get(from_state, 0)
        ti = MarkovChain.STATE_IDX.get(to_state, 0)
        self.action_counts[action][fi][ti] += 1

    # -- inference --------------------------------------------------------
    def _raw_counts(self, action: str) -> list[list[float]]:
        """Return the (unnormalised) 6x6 count matrix used for ``action``.

        Combines the base Markov-chain counts with the action prior, or uses the
        learned action counts directly once observations exist. These counts are
        the Dirichlet concentration parameters used for uncertainty intervals.
        """
        if action in ("none", "") or action not in self.ACTION_PRIORS:
            return [row[:] for row in self.base_chain.counts]
        prior = self.ACTION_PRIORS[action]
        base = self.base_chain.counts
        merged = [[base[i][j] + prior[i][j] for j in range(6)] for i in range(6)]
        if action in self.action_counts:
            learned = self.action_counts[action]
            if sum(sum(r) for r in learned) > 0:
                merged = [[learned[i][j] for j in range(6)] for i in range(6)]
        return merged

    def transition_matrix(self, action: str) -> list[list[float]]:
        """Return the 6x6 transition matrix for ``action`` (normalised)."""
        counts = self._raw_counts(action)
        out = []
        for row in counts:
            tot = sum(row)
            out.append([v / tot for v in row] if tot > 0 else [1 / 6] * 6)
        return out

    def step_distribution(self, dist: dict, action: str) -> dict:
        """Advance a state distribution by one step under ``action``."""
        matrix = self.transition_matrix(action)
        vec = [dist.get(s, 0.0) for s in MarkovChain.STATES]
        new = [0.0] * 6
        for i in range(6):
            for j in range(6):
                new[j] += vec[i] * matrix[i][j]
        return {MarkovChain.STATES[i]: new[i] for i in range(6)}

    def failed_probability(self, dist: dict) -> float:
        return dist.get("FAILED", 0.0)

    def predictive_interval(self, dist: dict, action: str, alpha: float = 0.05) -> tuple:
        """Analytic credible interval for expected ``P(FAILED)`` after ``action``.

        Treats each source state's FAILED-probability as a Beta posterior derived
        from its (base + prior, or learned) transition counts, then propagates the
        mixture over the current state distribution. Returns ``(mean, lo, hi)``.
        This quantifies the *uncertainty* of the planner's risk estimate, grounded
        in the decision-making corpus's Bayesian / uncertainty-quantification line
        (Bayesian Optimization df=1113, Optimization Under Uncertainty df=747).
        """
        z = 1.96 if alpha <= 0.05 else 1.645
        fi = MarkovChain.STATE_IDX
        failed = fi["FAILED"]
        counts = self._raw_counts(action)
        mean = 0.0
        comp = []
        for state, w in dist.items():
            if w <= 0:
                continue
            i = fi.get(state, 0)
            row = counts[i]
            total = sum(row)
            if total <= 0:
                continue
            fc = row[failed]
            # Centre the interval on the MLE estimate (so it matches
            # `failed_probability`) but use the Beta posterior variance for width.
            p_i = fc / total
            var_i = (fc + 1) * (total - fc + 1) / (((total + 2) ** 2) * (total + 3))
            comp.append((w, p_i, var_i))
            mean += w * p_i
        var = sum(w * w * v for w, p, v in comp)
        for w, p, v in comp:
            var += w * (p - mean) ** 2
        sd = var ** 0.5
        lo = max(0.0, mean - z * sd)
        hi = min(1.0, mean + z * sd)
        return (mean, lo, hi)

    def what_if(
        self, start_state: str, action: str = "none", horizon: int = 3
    ) -> dict:
        """World-model "what-if" simulator for an analyst/LLM.

        Applies ``action`` for ``horizon`` steps from ``start_state`` and returns
        the per-step expected state distribution, ``P(FAILED)`` and its calibrated
        credible interval. Lets operators reason about "what happens if we run
        action X now?" without executing anything. Grounded in the corpus's
        model-based / what-if reasoning line.
        """
        dist = {s: 0.0 for s in MarkovChain.STATES}
        if start_state in MarkovChain.STATE_IDX:
            dist[start_state] = 1.0
        else:
            return {"error": f"invalid state '{start_state}'"}
        steps = []
        for i in range(horizon):
            # Calibrated uncertainty of applying `action` from the current
            # distribution — its mean brackets the point estimate below.
            interval = self.predictive_interval(dist, action)
            after = self.step_distribution(dist, action)
            pf = self.failed_probability(after)
            steps.append(
                {
                    "step": i + 1,
                    "action": action,
                    "expected_state_distribution": {
                        k: round(v, 4) for k, v in after.items()
                    },
                    "expected_failed_prob": round(pf, 4),
                    "expected_failed_interval": [round(x, 4) for x in interval],
                }
            )
            dist = after
        return {
            "start_state": start_state,
            "action": action,
            "horizon": horizon,
            "steps": steps,
        }

    # -- refinement from observations ------------------------------------
    def refine_base_from_pairs(self, pairs) -> None:
        """Increment the base chain counts from observed (from, to) pairs."""
        for from_state, to_state in pairs:
            if from_state in MarkovChain.STATE_IDX and to_state in MarkovChain.STATE_IDX:
                self.base_chain.record_transition(from_state, to_state)

    def refine_action_from_triples(self, triples) -> None:
        """Learn action-conditioned transitions from (from, action, to) triples."""
        for from_state, action, to_state in triples:
            if from_state in MarkovChain.STATE_IDX and to_state in MarkovChain.STATE_IDX:
                self.record_transition(from_state, to_state, action)

    def refine_from_backtester(self, backtester) -> int:
        """Refine the *base* transition dynamics from observed backtester data.

        Pairs consecutive (pod, time)-ordered prediction states as observed
        state->state transitions, and prediction->subsequent-outcome states.
        This improves the underlying Markov dynamics from real fleet data. It
        does not yet carry action labels (the backtester records outcomes, not
        the remediation applied); action effects are learned via
        ``record_transition`` / ``feed_transition`` as post-action states are
        observed. Idempotent: only applied once per model instance.
        """
        if getattr(self, "_backtester_refined", False):
            return 0
        preds = getattr(backtester, "_predictions", []) or []
        outs = getattr(backtester, "_outcomes", []) or []
        pairs = []
        by_pod: dict = {}
        for p in sorted(preds, key=lambda r: r.timestamp):
            by_pod.setdefault(p.pod_key, []).append(p)
        for lst in by_pod.values():
            for a, b in zip(lst, lst[1:]):
                if (a.markov_state in MarkovChain.STATE_IDX
                        and b.markov_state in MarkovChain.STATE_IDX):
                    pairs.append((a.markov_state, b.markov_state))
        by_pod_out: dict = {}
        for o in sorted(outs, key=lambda r: r.timestamp):
            by_pod_out.setdefault(o.pod_key, []).append(o)
        for pod, lst in by_pod.items():
            ol = by_pod_out.get(pod, [])
            if not ol:
                continue
            for p in lst:
                later = [o for o in ol if o.timestamp >= p.timestamp]
                if later and later[0].state in MarkovChain.STATE_IDX:
                    pairs.append((p.markov_state, later[0].state))
        if pairs:
            self.refine_base_from_pairs(pairs)
            self._backtester_refined = True
        return len(pairs)

    # -- persistence (action effects only; base chain is persisted by StateStore) --
    def to_action_dict(self) -> dict:
        """Serialize the learned action-conditioned transitions for persistence.

        The base chain is intentionally excluded: it is persisted separately by
        ``StateStore`` (as the live ``MarkovChain``) and shared by reference, so
        the planner's base dynamics always match the operator's learned model.
        """
        return {
            "action_counts": self.action_counts,
            "backtester_refined": getattr(self, "_backtester_refined", False),
        }

    def load_action_counts(self, data: dict) -> None:
        """Load learned action-conditioned transitions from ``to_action_dict``."""
        if not data:
            return
        self.action_counts = data.get("action_counts", {})
        self._backtester_refined = data.get("backtester_refined", False)
        # Normalise loaded counts to plain lists-of-lists (JSON round-trip).
        for action, matrix in self.action_counts.items():
            self.action_counts[action] = [[float(v) for v in row] for row in matrix]

    # -- continual learning (#10) ------------------------------------------
    def total_support(self) -> float:
        """Total observed transitions across all actions (data-volume proxy)."""
        return float(
            sum(
                sum(sum(r) for r in matrix)
                for matrix in self.action_counts.values()
            )
        )

    def state_support(self, state: str) -> float:
        """Total observed transitions originating from ``state`` (all actions)."""
        fi = MarkovChain.STATE_IDX.get(state, 0)
        return float(
            sum(matrix[fi][ti] for matrix in self.action_counts.values() for ti in range(6))
        )

    def is_confident(self, state: str, min_support: float) -> bool:
        """True if enough transitions have been seen from ``state`` to plan."""
        if min_support <= 0:
            return True
        return self.state_support(state) >= min_support

    def decay(self, alpha: float = 0.99) -> float:
        """Exponential forgetting: scale all learned counts by ``alpha`` in [0,1].

        Lets the model forget stale transitions so recent behaviour dominates
        (continual learning). Returns the total transition-weight removed. An
        ``alpha`` of 1.0 disables forgetting.
        """
        if alpha >= 1.0:
            return 0.0
        removed = 0.0
        for matrix in self.action_counts.values():
            for i in range(6):
                for j in range(6):
                    v = matrix[i][j]
                    if v:
                        new_v = v * alpha
                        removed += v - new_v
                        matrix[i][j] = new_v if new_v >= 1e-6 else 0.0
        return removed


class Planner:
    """Short-horizon (Scenario-MPC-style) planner.

    Greedily chooses, at each step, the action that minimises *expected cost*:
    ``P(FAILED) * failure_cost + action_cost``, subject to per-action cooldowns.
    This is a decision-theoretic (value-of-intervention) objective rather than a
    flat penalty: the planner intervenes only when the expected reduction in
    failure risk exceeds the cost of acting. Produces an *advisory* plan only —
    it never executes anything.
    """

    def __init__(
        self,
        model: Optional[ActionAwareTransitionModel] = None,
        cooldowns: Optional[dict] = None,
        cost_weight: float = 0.02,
        action_costs: Optional[dict] = None,
        failure_cost: float = 1.0,
    ):
        self.model = model or ActionAwareTransitionModel()
        # action -> last-used timestamp (seconds); "none" is always allowed.
        self.cooldowns = cooldowns or {}
        self.cooldown_seconds = 300
        # Backwards-compatible uniform cost; preferred per-action cost model below.
        self.cost_weight = cost_weight
        self.action_costs = action_costs or DEFAULT_ACTION_COSTS
        self.failure_cost = failure_cost

    def _action_cost(self, action: str) -> float:
        """Estimated cost of taking ``action`` (0 for "none")."""
        if action == "none":
            return 0.0
        return self.action_costs.get(action, self.cost_weight)

    def _action_allowed(self, action: str, now: float) -> bool:
        if action == "none":
            return True
        last = self.cooldowns.get(action)
        if last is None:
            return True
        return (now - last) >= self.cooldown_seconds

    def plan(self, current_state: str, horizon: int = 3) -> Plan:
        """Return a plan minimising expected cost (risk * failure_cost + action cost)."""
        dist = {s: 0.0 for s in MarkovChain.STATES}
        dist[current_state] = 1.0
        plan = Plan(current_state=current_state, horizon=horizon)
        now = time.time()

        for step in range(horizon):
            before = self.model.failed_probability(dist)
            state_before = max(dist, key=lambda s: dist[s])
            best_action, best_score, best_fail, best_dist = "none", float("inf"), before, None
            for action in ACTION_TYPES:
                if not self._action_allowed(action, now):
                    continue
                cand = self.model.step_distribution(dist, action)
                fail = self.model.failed_probability(cand)
                # Expected cost = P(FAILED) * failure_cost + action cost.
                score = fail * self.failure_cost + self._action_cost(action)
                if best_dist is None or score < best_score:
                    best_action, best_score, best_fail, best_dist = action, score, fail, cand
            if best_dist is None:
                best_dist = dist
            interval = self.model.predictive_interval(best_dist, best_action)

            rationale = _step_rationale(
                step, best_action, state_before, before, best_fail,
                self._action_cost(best_action), self.failure_cost,
            )
            plan.steps.append(
                PlanStep(
                    step=step,
                    action=best_action,
                    expected_failed_prob_before=before,
                    expected_failed_prob_after=best_fail,
                    expected_state_distribution=best_dist,
                    expected_failed_interval_after=interval,
                    rationale=rationale,
                )
            )
            dist = best_dist
            # Mark chosen (non-none) action as used for cooldown purposes.
            if best_action != "none":
                self.cooldowns[best_action] = now

        plan.rationale = (
            f"Greedy Scenario-MPC lookahead over {horizon} steps minimising expected "
            "P(FAILED); intervention chosen only when it strictly reduces risk "
            "beyond the action-cost penalty. Output is passed to RemediationManager "
            "for execution under SafetyPolicy."
        )
        return plan


class ValueIterationPolicy:
    """Stationary optimal policy via value iteration over the action-MDP.

    Treats each Markov state as a node and ``transition_matrix(action)`` as the
    action-conditioned dynamics. Reward is ``-failure_cost * P(FAILED) -
    action_cost`` (the same decision-theoretic objective as :class:`Planner`),
    so the resulting stationary policy minimises long-run expected cost. This is
    the model-based RL / dynamic-programming counterpart to the myopic
    Scenario-MPC :class:`Planner`: an exact solution for the infinite-horizon
    problem rather than a short look-ahead. Use it as an alternative action
    selector, or to seed / sanity-check the planner.
    """

    def __init__(
        self,
        model: Optional[ActionAwareTransitionModel] = None,
        actions: Optional[list] = None,
        failure_cost: float = 1.0,
        action_costs: Optional[dict] = None,
        gamma: float = 0.95,
        threshold: float = 1e-4,
        max_iter: int = 500,
    ):
        self.model = model or ActionAwareTransitionModel()
        self.actions = actions or ACTION_TYPES
        self.failure_cost = failure_cost
        self.action_costs = action_costs or DEFAULT_ACTION_COSTS
        self.gamma = gamma
        self.threshold = threshold
        self.max_iter = max_iter
        self.V = {s: 0.0 for s in MarkovChain.STATES}
        self.policy = {s: "none" for s in MarkovChain.STATES}
        self.solve()

    def _action_cost(self, action: str) -> float:
        if action == "none":
            return 0.0
        return self.action_costs.get(action, 0.0)

    def _reward(self, state: str, action: str) -> float:
        dist = {s: 0.0 for s in MarkovChain.STATES}
        dist[state] = 1.0
        after = self.model.step_distribution(dist, action)
        pf = self.model.failed_probability(after)
        return -self.failure_cost * pf - self._action_cost(action)

    def _q(self, state: str, action: str, V: dict) -> float:
        idx = MarkovChain.STATE_IDX[state]
        row = self.model.transition_matrix(action)[idx]
        expected_next = sum(row[j] * V[MarkovChain.STATES[j]] for j in range(6))
        return self._reward(state, action) + self.gamma * expected_next

    def solve(self) -> None:
        """Run value iteration until the value function converges."""
        V = {s: 0.0 for s in MarkovChain.STATES}
        for _ in range(self.max_iter):
            delta = 0.0
            new_V = {}
            for s in MarkovChain.STATES:
                best = max(
                    (self._q(s, a, V) for a in self.actions), default=float("-inf")
                )
                new_V[s] = best if best > float("-inf") else 0.0
                delta = max(delta, abs(new_V[s] - V[s]))
            V = new_V
            if delta < self.threshold:
                break
        self.V = V
        self.policy = {
            s: max(self.actions, key=lambda a: self._q(s, a, V)) for s in MarkovChain.STATES
        }

    def action(self, state: str) -> str:
        """Return the optimal stationary action for ``state``."""
        return self.policy.get(state, "none")

    def value(self, state: str) -> float:
        return self.V.get(state, 0.0)

    def plan(self, start_state: str, horizon: int = 3) -> Plan:
        """Build an execution-ready plan by rolling out the stationary policy."""
        dist = {s: 0.0 for s in MarkovChain.STATES}
        dist[start_state] = 1.0
        plan = Plan(current_state=start_state, horizon=horizon)
        for step in range(horizon):
            before = self.model.failed_probability(dist)
            current = max(dist, key=lambda s: dist[s])
            a = self.action(current)
            after = self.model.step_distribution(dist, a)
            interval = self.model.predictive_interval(dist, a)
            plan.steps.append(
                PlanStep(
                    step=step,
                    action=a,
                    expected_failed_prob_before=before,
                    expected_failed_prob_after=self.model.failed_probability(after),
                    expected_state_distribution={
                        k: round(v, 4) for k, v in after.items()
                    },
                    expected_failed_interval_after=interval,
                )
            )
            dist = after
        return plan


def advisory_plan(current_state: str, horizon: int = 3,
                   base_chain: Optional[MarkovChain] = None,
                   cooldowns: Optional[dict] = None) -> dict:
    """Convenience helper: build a model + planner and return an execution-ready plan dict."""
    model = ActionAwareTransitionModel(base_chain=base_chain)
    planner = Planner(model=model, cooldowns=cooldowns)
    return planner.plan(current_state, horizon).to_dict()
