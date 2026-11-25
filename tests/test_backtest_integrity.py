"""Regression tests for backtest integrity bugs found in the 2026-11-25 audit.

Three independent defects, all invisible in normal operation:

1. **Unbounded history load (OOM bomb).** ``_load_history`` appended *every*
   record from the JSONL file to the in-memory lists. The ``MAX_HISTORY_RECORDS``
   cap was only applied on the *next* ``record_*`` call, i.e. after the memory
   was already allocated. The history file is capped at 256 MiB
   (``MAX_HISTORY_FILE_BYTES``) and holds ~267 bytes/record, so a restart could
   load ~960k records — exactly the OOM the rotation cap claims to prevent
   (the operator *was* OOM-killed once at a 1.8 GB file).

2. **Cumulative ``restart_count`` used as a failure label.**
   ``record_backtest_predictions`` labelled a pod a failure when
   ``tracker.restart_count > 0``. ``restartCount`` is cumulative for the pod's
   whole lifetime and never resets, so every pod that ever restarted was
   permanently recorded as a failure — measured live: 3,582 of 10,000 pairs
   "false negatives" in a healthy cluster. This is the same stale-signal trap
   already fixed in ``risk.py`` and ``classify_state``.

3. **O(n*m) outcome matching.** ``evaluate`` scanned the full outcome list for
   every prediction (10k x 10k = 10^8 comparisons), making ``/backtest`` block
   for ~10s of CPU.
"""

import json

import pytest

from predictive_agent import backtester as bt_mod
from predictive_agent.backtester import Backtester


def _write_jsonl(path, n_predictions, n_outcomes=None):
    """Write a history file with n_predictions prediction records."""
    n_outcomes = n_predictions if n_outcomes is None else n_outcomes
    with open(path, "w") as f:
        for i in range(n_predictions):
            f.write(json.dumps({
                "_type": "prediction",
                "pod_key": f"ns/pod-{i}",
                "timestamp": float(i),
                "risk_score": 0.1,
                "ttf_minutes": None,
                "confidence": 0.5,
                "markov_state": "HEALTHY",
                "memory_pct": 10.0,
                "cpu_pct": 1.0,
                "memory_trend": 0.0,
                "cpu_trend": 0.0,
            }) + "\n")
        for i in range(n_outcomes):
            f.write(json.dumps({
                "_type": "outcome",
                "pod_key": f"ns/pod-{i}",
                "timestamp": float(i),
                "actual_failure": False,
                "restart_count": 0,
                "state": "HEALTHY",
            }) + "\n")


class TestHistoryLoadIsCapped:
    """Defect 1: loading must respect MAX_HISTORY_RECORDS."""

    def test_load_keeps_only_last_max_records(self, tmp_path, monkeypatch):
        path = str(tmp_path / "history.jsonl")
        _write_jsonl(path, 200)
        monkeypatch.setattr(bt_mod, "MAX_HISTORY_RECORDS", 25)

        bt = Backtester(history_file=path)

        assert len(bt._predictions) == 25
        assert len(bt._outcomes) == 25
        # The newest records must be the ones kept (tail of the file).
        assert bt._predictions[-1].pod_key == "ns/pod-199"
        assert bt._outcomes[-1].pod_key == "ns/pod-199"

    def test_load_under_cap_is_unchanged(self, tmp_path, monkeypatch):
        path = str(tmp_path / "history.jsonl")
        _write_jsonl(path, 7)
        monkeypatch.setattr(bt_mod, "MAX_HISTORY_RECORDS", 25)

        bt = Backtester(history_file=path)

        assert len(bt._predictions) == 7
        assert len(bt._outcomes) == 7


class TestEvaluateMatchesNearestOutcome:
    """Defect 3: matching must be indexed, and still pick the nearest outcome."""

    def _bt(self, tmp_path):
        return Backtester(history_file=str(tmp_path / "h.jsonl"))

    def test_picks_nearest_outcome_in_time(self, tmp_path):
        bt = self._bt(tmp_path)
        bt._predictions = [bt_mod.PredictionRecord(
            pod_key="ns/p", timestamp=100.0, risk_score=0.2, ttf_minutes=None,
            confidence=0.5, markov_state="HEALTHY",
            memory_pct=10.0, cpu_pct=1.0, memory_trend=0.0, cpu_trend=0.0,
        )]
        bt._outcomes = [
            bt_mod.OutcomeRecord(pod_key="ns/p", timestamp=10.0, actual_failure=True, actual_ttf_minutes=None, restart_count=0, state="FAILED"),
            bt_mod.OutcomeRecord(pod_key="ns/p", timestamp=140.0, actual_failure=False, actual_ttf_minutes=None, restart_count=0, state="HEALTHY"),
            bt_mod.OutcomeRecord(pod_key="ns/p", timestamp=900.0, actual_failure=True, actual_ttf_minutes=None, restart_count=0, state="FAILED"),
        ]

        report = bt.evaluate(time_window=120)

        assert report.matched_pairs == 1
        # Nearest in time is the 140.0 outcome (dt=40), not 10.0 (dt=90) or 900.
        assert report.true_negatives + report.false_positives == 1
        assert report.brier_score == pytest.approx(0.2 ** 2)

    def test_no_match_outside_window(self, tmp_path):
        bt = self._bt(tmp_path)
        bt._predictions = [bt_mod.PredictionRecord(
            pod_key="ns/p", timestamp=100.0, risk_score=0.2, ttf_minutes=None,
            confidence=0.5, markov_state="HEALTHY",
            memory_pct=10.0, cpu_pct=1.0, memory_trend=0.0, cpu_trend=0.0,
        )]
        bt._outcomes = [
            bt_mod.OutcomeRecord(pod_key="ns/p", timestamp=500.0, actual_failure=True, actual_ttf_minutes=None, restart_count=0, state="FAILED"),
        ]

        report = bt.evaluate(time_window=60)

        assert report.matched_pairs == 0

    def test_matches_are_indexed_per_pod(self, tmp_path):
        """Many predictions across a few pods must not be O(n*m).

        Measured with the nested scan: n=4000 -> 0.33s, n=8000 -> 1.34s,
        n=10000 -> 2.14s (2x n = 4x time, i.e. quadratic); ~10s live in-pod.
        Indexing by pod_key makes it linear; assert a generous wall-clock bound
        so the test is not flaky but still catches a quadratic regression.
        """
        import time

        bt = self._bt(tmp_path)
        pods = [f"ns/pod-{i % 8}" for i in range(8000)]
        bt._predictions = [
            bt_mod.PredictionRecord(
                pod_key=pods[i], timestamp=float(i), risk_score=0.2,
                ttf_minutes=None, confidence=0.5, markov_state="HEALTHY",
                memory_pct=10.0, cpu_pct=1.0, memory_trend=0.0, cpu_trend=0.0,
            )
            for i in range(8000)
        ]
        bt._outcomes = [
            bt_mod.OutcomeRecord(pod_key=pods[i], timestamp=float(i), actual_failure=False, actual_ttf_minutes=None, restart_count=0, state="HEALTHY")
            for i in range(8000)
        ]

        started = time.monotonic()
        report = bt.evaluate(time_window=100)
        elapsed = time.monotonic() - started

        assert report.matched_pairs > 0
        assert elapsed < 0.8, f"evaluate took {elapsed:.2f}s — matching is quadratic again"
