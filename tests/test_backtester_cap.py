"""Cap on the backtest history JSONL file.

Regression: backtest-history.jsonl grew unbounded (only an in-memory
record cap existed via BACKTEST_MAX_HISTORY). At 1.8GB the singleton load()
at startup pulled the whole file into RAM (~2.1GB anon-rss) and the operator
was OOMKilled against its memory limit in a restart loop.

Fix: MAX_HISTORY_FILE_BYTES (env BACKTEST_HISTORY_MAX_MB, default 256).
When appending would exceed the cap, the file is rotated to <history>.1
(old backup replaced), so disk usage and the startup load stay bounded and
the newest records are preserved.
"""
import os

from predictive_agent import backtester as bt_mod
from predictive_agent.backtester import Backtester


def test_default_cap_is_256mb():
    """The default cap must be 256 MiB unless the env knob overrides it."""
    if "BACKTEST_HISTORY_MAX_MB" not in os.environ:
        assert bt_mod.MAX_HISTORY_FILE_BYTES == 256 * 1024 * 1024


def test_history_file_is_capped_on_overflow(tmp_path, monkeypatch):
    """Appends must not push the main history file past the cap: overflow
    rotates the file to <history>.1 instead of growing it."""
    monkeypatch.setattr(bt_mod, "MAX_HISTORY_FILE_BYTES", 100)
    target = tmp_path / "backtest-history.jsonl"
    bt = Backtester(history_file=str(target))
    for i in range(50):
        bt.record_prediction(
            pod_key=f"ns/pod-{i}", risk_score=0.7, ttf_minutes=15,
            confidence=0.8, markov_state="STRESSED",
        )
    # Each JSONL line is ~150 bytes, so 50 appends would be ~7KB uncapped.
    # With the cap the main file must stay within cap + one line.
    assert target.stat().st_size <= bt_mod.MAX_HISTORY_FILE_BYTES + 1024
    # The old chunk lives on in the single rotated backup.
    assert os.path.exists(str(target) + ".1")


def test_rotation_keeps_newest_records_loadable(tmp_path, monkeypatch):
    """After a rotation the most recent records must still survive a reload
    from the main (capped) file — startup memory stays bounded AND data is
    not lost."""
    monkeypatch.setattr(bt_mod, "MAX_HISTORY_FILE_BYTES", 100)
    target = tmp_path / "backtest-history.jsonl"
    bt = Backtester(history_file=str(target))
    bt.record_prediction(pod_key="ns/first", risk_score=0.5, ttf_minutes=10,
                         confidence=0.6, markov_state="HEALTHY")
    for i in range(100):
        bt.record_prediction(pod_key=f"ns/later-{i}", risk_score=0.7,
                             ttf_minutes=15, confidence=0.8, markov_state="STRESSED")
    bt2 = Backtester(history_file=str(target))
    assert bt2.get_prediction_count() >= 1
    # The newest record must be present in the main file.
    assert any(r.pod_key == "ns/later-99" for r in bt2._predictions)


def test_small_file_never_rotated(tmp_path, monkeypatch):
    """Files comfortably under the cap must be appended untouched (no
    spurious rotation / data loss on normal operation)."""
    monkeypatch.setattr(bt_mod, "MAX_HISTORY_FILE_BYTES", 10 * 1024 * 1024)
    target = tmp_path / "backtest-history.jsonl"
    bt = Backtester(history_file=str(target))
    bt.record_prediction(pod_key="ns/small", risk_score=0.5, ttf_minutes=10,
                         confidence=0.6, markov_state="HEALTHY")
    assert target.exists()
    assert "ns/small" in target.read_text()
    assert not os.path.exists(str(target) + ".1")
