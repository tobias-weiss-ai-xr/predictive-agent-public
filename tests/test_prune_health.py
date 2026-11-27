"""Prune end: cached predictions + notify-throttle stamps must not grow
unbounded, and removed pods must not stay "cached" forever (they would
pollute decision_layer_mean_confidence and leak memory)."""

import predictive_agent.reconcile_phases as phases
from predictive_agent.predictor import Predictor


def _pred(pod_key: str, risk: float):
    return type(
        "R",
        (),
        {"pod_key": pod_key, "risk_score": risk, "confidence": 0.8},
    )()


def test_prune_predictions_removes_missing_keeps_active():
    p = Predictor(risk_threshold=0.5)
    p._predictions = {"ns/a": _pred("ns/a", 0.9), "ns/b": _pred("ns/b", 0.8), "ns/c": _pred("ns/c", 0.3)}
    p.prune_predictions({"ns/a", "ns/c"})
    assert set(p._predictions) == {"ns/a", "ns/c"}


def test_prune_predictions_clears_when_no_active_keys():
    p = Predictor(risk_threshold=0.5)
    p._predictions = {"ns/a": _pred("ns/a", 0.9), "ns/b": _pred("ns/b", 0.4)}
    p.prune_predictions(set())
    assert p._predictions == {}


def test_prune_predictions_keeps_everything_when_all_active():
    p = Predictor(risk_threshold=0.5)
    p._predictions = {"ns/a": _pred("ns/a", 0.9), "ns/b": _pred("ns/b", 0.4)}
    p.prune_predictions({"ns/a", "ns/b"})
    assert set(p._predictions) == {"ns/a", "ns/b"}


def test_autonomy_notify_prune_drops_old_keeps_recent():
    now = 2_000_000.0
    d = {
        "ns/a": ("review_required", now - 10),           # fresh
        "ns/b": ("review_required", now - 3601),         # older than 1h
        "ns/c": ("plan", now - 7200),                    # much older
        "remediation_exhausted|aggregate": ("remediation_exhausted", now - 30),  # fresh
    }
    phases._prune_autonomy_notify(d, now, ttl_s=3600)
    assert set(d) == {"ns/a", "remediation_exhausted|aggregate"}


def test_autonomy_notify_prune_handles_empty():
    d = {}
    phases._prune_autonomy_notify(d, 1_000_000.0, ttl_s=3600)
    assert d == {}
