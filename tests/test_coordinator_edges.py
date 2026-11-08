"""Edge-case tests for wave sequencing in predictive_agent/coordinator.py.

Covers the per-node wave-packing logic of ``MultiAgentCoordinator._sequence``:

1. an empty per_pod map yields no waves without raising;
2. unknown/missing state strings sort after the known priorities and never raise
   (CRITICAL/FAILED come first);
3. two pods on the same node that cannot share a wave land in separate waves;
4. pods on different nodes with the same priority can share a wave;
5. a single pod yields exactly one wave with one item;
6. sequenced items carry pod/action/node fields through unchanged.
"""

from predictive_agent.coordinator import MultiAgentCoordinator, _priority


def _coord(max_actions_per_node: int = 2) -> MultiAgentCoordinator:
    """Coordinator bound to no manager — wave sequencing needs none."""
    return MultiAgentCoordinator(manager=None, max_actions_per_node=max_actions_per_node)


def _waves(pod_specs, actions, max_actions_per_node: int = 2):
    """Run _sequence over pod_specs {key: (state, node)} and actions {key: [...]}."""
    coord = _coord(max_actions_per_node=max_actions_per_node)
    pod_info = {
        key: {"state": state, "node": node}
        for key, (state, node) in pod_specs.items()
    }
    per_pod = {key: {"actions": actions[key]} for key in actions}
    return coord._sequence(pod_info, per_pod)


def test_empty_per_pod_returns_no_waves_without_raising():
    coord = _coord()
    assert coord._sequence({}, {}) == []


def test_priority_table_known_values():
    assert _priority("CRITICAL") == 5
    assert _priority("FAILED") == 5
    assert _priority("STRESSED") == 4
    assert _priority("DEGRADED") == 3
    assert _priority("RECOVERED") == 1
    assert _priority("HEALTHY") == 0


def test_unknown_state_sorts_after_critical_failed():
    spec = {
        "p-critical": ("CRITICAL", "n1"),
        "p-failed": ("FAILED", "n1"),
        "p-mystery": ("MYSTERY_STATE", "n1"),
    }
    actions = {key: ["pod_restart"] for key in spec}
    waves = _waves(spec, actions, max_actions_per_node=4)
    order = [it["pod"] for wave in waves for it in wave]
    # CRITICAL/FAILED first (priority 5), unknown state sorts after them.
    assert order[0] == "p-critical"
    assert order[1] == "p-failed"
    assert order[2] == "p-mystery"
    assert _priority("MYSTERY_STATE") == 0


def test_missing_state_sorts_after_known_without_raising():
    # Blank state string and state entirely absent must not raise.
    spec = {
        "p-critical": ("CRITICAL", "n1"),
        "p-blank": ("", "n1"),
    }
    actions = {"p-critical": ["pod_restart"], "p-blank": ["pod_restart"]}
    waves = _waves(spec, actions, max_actions_per_node=2)
    order = [it["pod"] for wave in waves for it in wave]
    assert order[0] == "p-critical"
    assert order[1] == "p-blank"

    coord = _coord()
    pod_info = {"p-no-state": {"node": "n1"}}  # no "state" key at all
    per_pod = {"p-no-state": {"actions": ["pod_restart"]}}
    waves = coord._sequence(pod_info, per_pod)  # must not raise
    assert len(waves) == 1
    assert waves[0][0]["priority"] == 0


def test_same_node_pods_that_cannot_share_land_in_separate_waves():
    # max_actions_per_node=1 → the two pods on the same node cannot share a wave.
    spec = {"p1": ("CRITICAL", "n1"), "p2": ("CRITICAL", "n1")}
    actions = {key: ["pod_restart"] for key in spec}
    waves = _waves(spec, actions, max_actions_per_node=1)
    assert len(waves) == 2
    assert all(len(wave) == 1 for wave in waves)
    assert {wave[0]["pod"] for wave in waves} == {"p1", "p2"}


def test_per_node_cap_packs_overflow_into_next_wave():
    spec = {
        "p1": ("CRITICAL", "n1"),
        "p2": ("CRITICAL", "n1"),
        "p3": ("CRITICAL", "n1"),
    }
    actions = {key: ["pod_restart"] for key in spec}
    waves = _waves(spec, actions, max_actions_per_node=2)
    assert [len(wave) for wave in waves] == [2, 1]


def test_different_nodes_same_priority_share_a_wave():
    spec = {
        "p1": ("CRITICAL", "n1"),
        "p2": ("CRITICAL", "n2"),
    }
    actions = {key: ["pod_restart"] for key in spec}
    waves = _waves(spec, actions, max_actions_per_node=1)
    assert len(waves) == 1
    assert len(waves[0]) == 2
    assert {it["pod"] for it in waves[0]} == {"p1", "p2"}


def test_single_pod_yields_exactly_one_wave_with_one_item():
    waves = _waves(
        {"only": ("STRESSED", "n9")},
        {"only": ["pod_restart"]},
        max_actions_per_node=2,
    )
    assert len(waves) == 1
    assert len(waves[0]) == 1
    assert waves[0][0] == {
        "pod": "only",
        "action": "pod_restart",
        "node": "n9",
        "priority": _priority("STRESSED"),
    }


def test_items_carry_pod_action_node_fields_unchanged():
    spec = {
        "p-a": ("DEGRADED", "node-7"),
        "p-b": ("STRESSED", None),  # no node — never blocks others
    }
    actions = {"p-a": ["pod_restart"], "p-b": ["node_cordon"]}
    waves = _waves(spec, actions, max_actions_per_node=1)
    flat = [it for wave in waves for it in wave]
    by_pod = {it["pod"]: it for it in flat}
    assert len(by_pod) == 2
    assert by_pod["p-a"]["action"] == "pod_restart"
    assert by_pod["p-a"]["node"] == "node-7"
    assert by_pod["p-a"]["priority"] == _priority("DEGRADED")
    assert by_pod["p-b"]["action"] == "node_cordon"
    assert by_pod["p-b"]["node"] is None
    assert by_pod["p-b"]["priority"] == _priority("STRESSED")
