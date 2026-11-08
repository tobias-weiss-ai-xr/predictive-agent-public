"""Edge-case tests for predictive_agent.backtester.

Focuses on history-file robustness and the in-memory / on-disk caps and
boundaries that the happy-path suite (test_backtester.py) does not hit:

  * corrupt / partial / truncated JSONL lines must be skipped, never crash a load
  * records with unknown (or missing) ``_type`` values are ignored gracefully
  * clear_history() empties only the in-memory buffers, never the file
  * writes must never raise, even when the parent directory disappears mid-run
  * outcome fields survive a write → reload round-trip intact
  * evaluate() returns a well-formed report even with an empty prediction side
"""
import json
import shutil

import pytest

from predictive_agent.backtester import Backtester


def _prediction_line(pod_key="ns/pod-1", timestamp=1000.0, risk=0.7, ttf=15):
    return json.dumps({
        "_type": "prediction",
        "pod_key": pod_key,
        "timestamp": timestamp,
        "risk_score": risk,
        "ttf_minutes": ttf,
        "confidence": 0.8,
        "markov_state": "STRESSED",
        "memory_pct": 40.0,
        "cpu_pct": 55.0,
        "memory_trend": 0.1,
        "cpu_trend": 0.2,
    })


def _outcome_line(pod_key="ns/pod-1", timestamp=1010.0, failure=True, restarts=3,
                  state="CRITICAL", ttf=None):
    return json.dumps({
        "_type": "outcome",
        "pod_key": pod_key,
        "timestamp": timestamp,
        "actual_failure": failure,
        "actual_ttf_minutes": ttf,
        "restart_count": restarts,
        "state": state,
    })


@pytest.fixture
def hist_path(tmp_path):
    """Return a fresh, non-existent JSONL history path in a temp dir."""
    return tmp_path / "backtest-history.jsonl"


class TestHistoryFileRobustness:
    def test_corrupt_lines_skipped_and_good_lines_recovered(self, hist_path):
        """Corrupt/partial JSONL lines are skipped; good lines still load."""
        hist_path.write_text("\n".join([
            _prediction_line(timestamp=1000.0),
            '{"pod_key": "ns/pod-2", "risk_score": 0.9, "ttf_minutes": 1',   # truncated
            "this is not json at all",                                        # garbage
            '[1, 2,',                                                         # partial array
            "{'single': 'quoted-json'}",                                      # invalid JSON
            _outcome_line(timestamp=1010.0),
            "",                                                               # blank line
        ]) + "\n")

        bt = Backtester(history_file=str(hist_path))  # must not raise

        assert bt.get_prediction_count() == 1
        assert bt.get_outcome_count() == 1
        # Fields of recovered good lines are intact
        assert bt._predictions[0].pod_key == "ns/pod-1"
        assert bt._predictions[0].risk_score == 0.7
        assert bt._outcomes[0].pod_key == "ns/pod-1"
        assert bt._outcomes[0].actual_failure is True
        assert bt._outcomes[0].restart_count == 3

    def test_truncated_trailing_line_without_newline_recovered(self, hist_path):
        """A final line cut off mid-write (no trailing newline) is tolerated."""
        hist_path.write_text(
            _prediction_line() + "\n" + '{"_type": "outcome", "pod_key": "ns/pod-1", "actu'
        )

        bt = Backtester(history_file=str(hist_path))

        assert bt.get_prediction_count() == 1
        assert bt.get_outcome_count() == 0

    def test_entirely_garbage_history_file_does_not_crash(self, hist_path):
        """A history file with only corrupt lines still yields a usable backtester."""
        hist_path.write_text("not json\n{\"partial\": \n<<<>>>\n")

        bt = Backtester(history_file=str(hist_path))

        assert bt.get_prediction_count() == 0
        assert bt.get_outcome_count() == 0
        # Still fully usable afterwards
        bt.record_prediction(pod_key="ns/pod-1", risk_score=0.3, ttf_minutes=None,
                             confidence=0.5, markov_state="HEALTHY")
        assert bt.get_prediction_count() == 1

    def test_unknown_type_records_are_skipped_gracefully(self, hist_path):
        """Records whose _type is unknown (or missing) are skipped, not fatal."""
        hist_path.write_text("\n".join([
            json.dumps({"_type": "decision", "pod_key": "ns/a", "value": 1}),
            json.dumps({"_type": "action", "pod_key": "ns/b", "action": "restart"}),
            json.dumps({"pod_key": "ns/c", "risk_score": 0.5}),  # no _type at all
            _prediction_line(pod_key="ns/good", timestamp=5.0, risk=0.1, ttf=None),
        ]) + "\n")

        bt = Backtester(history_file=str(hist_path))

        assert bt.get_prediction_count() == 1
        assert bt.get_outcome_count() == 0
        assert bt._predictions[0].pod_key == "ns/good"
        # Still usable after loading
        bt.record_prediction(pod_key="ns/x", risk_score=0.2, ttf_minutes=None,
                             confidence=0.5, markov_state="HEALTHY")
        assert bt.get_prediction_count() == 2


class TestHistoryCapsAndLifecycle:
    def test_clear_history_empties_memory_but_file_remains(self, hist_path):
        """clear_history() drops in-memory records; disk file is untouched."""
        bt = Backtester(history_file=str(hist_path))
        bt.record_prediction(pod_key="ns/pod-1", risk_score=0.7, ttf_minutes=15,
                             confidence=0.8, markov_state="STRESSED")
        bt.record_outcome(pod_key="ns/pod-1", actual_failure=True,
                          restart_count=2, state="CRITICAL")
        assert bt.get_prediction_count() == 1
        assert bt.get_outcome_count() == 1

        bt.clear_history()

        assert bt.get_prediction_count() == 0
        assert bt.get_outcome_count() == 0
        # File on disk is still there and non-empty
        assert hist_path.exists()
        assert hist_path.stat().st_size > 0
        # A fresh instance re-reads the persisted records
        bt2 = Backtester(history_file=str(hist_path))
        assert bt2.get_prediction_count() == 1
        assert bt2.get_outcome_count() == 1

    def test_write_after_parent_dir_deleted_mid_run_does_not_raise(self, hist_path):
        """Records appended after the parent dir vanishes do not raise."""
        data_dir = hist_path.parent
        bt = Backtester(history_file=str(hist_path))
        bt.record_outcome(pod_key="ns/pod-1", actual_failure=False,
                          restart_count=0, state="HEALTHY")
        assert bt.get_outcome_count() == 1

        # Simulate the PVC / data dir disappearing mid-run
        shutil.rmtree(data_dir)

        bt.record_outcome(pod_key="ns/pod-1", actual_failure=True,
                          restart_count=1, state="CRITICAL")
        bt.record_prediction(pod_key="ns/pod-1", risk_score=0.9, ttf_minutes=5,
                             confidence=0.9, markov_state="CRITICAL")
        # No exception raised; in-memory state still tracks both records
        assert bt.get_outcome_count() == 2
        assert bt.get_prediction_count() == 1

    def test_write_when_parent_path_blocked_drops_silently(self, hist_path):
        """If the parent dir is un-usable, records drop silently (DEBUG), no raise."""
        data_dir = hist_path.parent
        bt = Backtester(history_file=str(hist_path))
        bt.record_prediction(pod_key="ns/pod-1", risk_score=0.5, ttf_minutes=10,
                             confidence=0.6, markov_state="HEALTHY")
        assert bt.get_prediction_count() == 1

        # Replace the parent dir with a plain file → makedirs/open must fail,
        # but _append_to_file swallows the OSError at DEBUG level.
        shutil.rmtree(data_dir)
        data_dir.write_text("blocked")

        bt.record_prediction(pod_key="ns/pod-1", risk_score=0.9, ttf_minutes=5,
                             confidence=0.9, markov_state="CRITICAL")
        assert bt.get_prediction_count() == 2


class TestOutcomePersistenceAndEvaluateEdges:
    def test_outcome_fields_survive_reload(self, hist_path):
        """actual_failure / restart_count / state / ttf persist across reload."""
        bt = Backtester(history_file=str(hist_path))
        bt.record_outcome(pod_key="ns/pod-1", actual_failure=True, restart_count=7,
                          state="CRITICAL", actual_ttf_minutes=3.5)

        bt2 = Backtester(history_file=str(hist_path))

        assert bt2.get_outcome_count() == 1
        outcome = bt2._outcomes[0]
        assert outcome.pod_key == "ns/pod-1"
        assert outcome.actual_failure is True
        assert outcome.restart_count == 7
        assert outcome.state == "CRITICAL"
        assert outcome.actual_ttf_minutes == 3.5

    def test_evaluate_with_only_outcomes_no_division_by_zero(self, hist_path):
        """evaluate() with outcomes but zero predictions is well-formed."""
        bt = Backtester(history_file=str(hist_path))
        for i in range(3):
            bt.record_outcome(pod_key=f"ns/pod-{i}", actual_failure=True,
                              restart_count=1, state="CRITICAL")
        assert bt.get_prediction_count() == 0
        assert bt.get_outcome_count() == 3

        report = bt.evaluate()  # must not raise (no division by zero)

        assert report.total_predictions == 0
        assert report.total_outcomes == 3
        assert report.matched_pairs == 0
        assert report.brier_score == 0.0
        assert report.true_positives == 0
        assert report.false_positives == 0
        assert report.false_negatives == 0
        assert report.true_negatives == 0
        assert report.ttf_errors == []
        assert report.calibration == {}

    def test_evaluate_with_only_predictions_no_division_by_zero(self, hist_path):
        """evaluate() with predictions but zero outcomes is well-formed."""
        bt = Backtester(history_file=str(hist_path))
        bt.record_prediction(pod_key="ns/pod-1", risk_score=0.9, ttf_minutes=10,
                             confidence=0.9, markov_state="CRITICAL")
        assert bt.get_outcome_count() == 0

        report = bt.evaluate()

        assert report.total_predictions == 1
        assert report.total_outcomes == 0
        assert report.matched_pairs == 0
        assert report.brier_score == 0.0
        assert report.to_dict()["brier_score"] == 0.0
