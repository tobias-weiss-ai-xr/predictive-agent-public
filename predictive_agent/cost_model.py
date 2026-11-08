"""Calibrate the decision-theoretic cost model from incident / SLO data (#5).

The planner minimises expected cost = ``P(FAILED) * failure_cost + action_cost``.
Both terms live in the same business-cost units so the trade-off is meaningful:

  * a pod failure costs ``failure_cost`` (e.g. an SLO breach ≈ outage minutes × $/min);
  * performing action ``a`` costs ``action_cost[a]`` (its disruption, e.g. its
    mean downtime × $/min, scaled by ``ACTION_DISRUPTION_FRACTION`` since a
    remediation disruption is typically far cheaper than a full outage).

Calibrating from real incidents makes the planner's intervention decisions
reflect actual operational cost instead of hand-tuned constants.
"""

import logging
from collections.abc import Iterable
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def calibrate_costs(
    incidents: Iterable[dict],
    slo_cost_per_minute: float = 1.0,
    typical_outage_minutes: float = 30.0,
    disruption_fraction: float = 0.1,
    default_action_cost: float = 0.05,
    action_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], float]:
    """Derive (action_costs, failure_cost) from incident history.

    Args:
        incidents: iterable of dicts with at least ``action`` (str) and
            ``downtime_minutes`` (float). Optional keys ``slo_violation``
            (bool) and ``severity`` (float, 0-1) are accepted but not required.
        slo_cost_per_minute: business cost of one minute of outage / SLO breach.
        typical_outage_minutes: mean duration of a full pod failure (the
            reference against which a remediation disruption is scaled).
        disruption_fraction: a remediation action's disruption is treated as at
            most this fraction of a full outage's cost.
        default_action_cost: cost assigned to actions never seen in incidents.
        action_names: full list of known action names (to pre-seed defaults).

    Returns:
        (action_costs, failure_cost) calibrated to the supplied incident data.
    """
    failure_cost = slo_cost_per_minute * typical_outage_minutes

    sums: Dict[str, float] = {}
    counts: Dict[str, float] = {}
    for inc in incidents:
        action = inc.get("action")
        dt = inc.get("downtime_minutes")
        if action is None or dt is None:
            continue
        try:
            dt = float(dt)
        except (TypeError, ValueError):
            continue
        sums[action] = sums.get(action, 0.0) + dt
        counts[action] = counts.get(action, 0.0) + 1.0

    action_costs: Dict[str, float] = {}
    for name in action_names or []:
        action_costs[name] = default_action_cost
    for action, n in counts.items():
        mean_dt = sums[action] / n
        action_costs[action] = slo_cost_per_minute * mean_dt * disruption_fraction

    logger.info(
        "Calibrated cost model from %d incident records: failure_cost=%.3f, "
        "action_costs=%s",
        int(sum(counts.values())), failure_cost, action_costs,
    )
    return action_costs, failure_cost
