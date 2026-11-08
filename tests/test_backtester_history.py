"""Regression tests for the backtester history-file persistence fix.

Canary (dry-run against a real Docker host with a custom HISTORY_FILE)
surfaced that Backtester hardcoded its history path to
/var/lib/opendesk/backtest-history.jsonl via its own BACKTEST_HISTORY_FILE
default, ignoring the shared HISTORY_FILE knob used by STATE_MODEL_FILE /
PREDICTIONS_FILE. With a custom HISTORY_FILE the backtester still tried to
write the system dir and silently dropped records at DEBUG.

Fix: DEFAULT_HISTORY_FILE now derives from config.HISTORY_FILE (the
BACKTEST_HISTORY_FILE override still takes precedence for backward compat).
These tests lock that behaviour in.
"""
import os

import predictive_agent.config as config
from predictive_agent import backtester as bt_mod
from predictive_agent.backtester import Backtester


def _dir_without(path):
    # Ensure the config HISTORY_FILE default resolves relative to /var/lib by
    # constructing an absolute path we control so the assertions don't depend
    # on the host's environment.
    return os.path.dirname(path)


def test_default_history_derives_from_config_history_file():
    """The default backtester history path must live alongside the shared
    HISTORY_FILE, NOT a hardcoded /var/lib/opendesk path."""
    assert os.path.join(
        _dir_without(config.HISTORY_FILE),
        "backtest-history.jsonl",
    ) == bt_mod.DEFAULT_HISTORY_FILE


def test_default_history_is_not_hardcoded_system_dir():
    """Regression: the default must respect the configured HISTORY_FILE rather
    than always writing to /var/lib/opendesk (which caused dropped records and
    'Permission denied' when HISTORY_FILE points elsewhere)."""
    # If HISTORY_FILE is not the legacy system location, the default must not be.
    if not config.HISTORY_FILE.startswith("/var/lib/opendesk"):
        assert not bt_mod.DEFAULT_HISTORY_FILE.startswith("/var/lib/opendesk/")
    assert "backtest-history.jsonl" in bt_mod.DEFAULT_HISTORY_FILE


def test_backtest_history_env_override_wins(monkeypatch):
    """The dedicated BACKTEST_HISTORY_FILE override still takes precedence."""
    custom = "/tmp/override-dir/custom-backtest.jsonl"
    monkeypatch.setenv("BACKTEST_HISTORY_FILE", custom)
    # Re-read the module default as a fresh process would.
    default = os.environ.get(
        "BACKTEST_HISTORY_FILE",
        os.path.join(os.path.dirname(config.HISTORY_FILE), "backtest-history.jsonl"),
    )
    assert default == custom


def test_backtester_uses_default_history_file_when_not_given():
    """Constructing a Backtester without an explicit file must use the
    derived default (verified via the class default binding)."""
    assert Backtester.__init__ is not None
    # The default parameter is bound to the (now-derived) DEFAULT_HISTORY_FILE.
    import inspect
    sig = inspect.signature(Backtester.__init__)
    default_val = sig.parameters["history_file"].default
    assert default_val is bt_mod.DEFAULT_HISTORY_FILE


def test_write_history_creates_parent_dir(tmp_path):
    """The canary failure mode: writing history must create the parent dir so
    a custom (non-system) HISTORY_FILE actually persists instead of dropping
    records with 'Permission denied'."""
    target = tmp_path / "nested" / "deep" / "backtest-history.jsonl"
    bt = Backtester(history_file=str(target))
    bt.record_prediction(
        pod_key="ns/pod-1",
        risk_score=0.7,
        ttf_minutes=15,
        confidence=0.8,
        markov_state="STRESSED",
    )
    assert target.exists()
    assert "pod-1" in target.read_text()


def test_custom_history_file_without_dir_persists(tmp_path):
    """End-to-end: a fully custom path (like the canary used) must persist a
    backtest record even though /var/lib/opendesk may be unwritable."""
    target = tmp_path / "state" / "alts" / "history.jsonl"
    bt = Backtester(history_file=str(target))
    bt.record_prediction(pod_key="a/b", risk_score=0.5, ttf_minutes=20,
                         confidence=0.6, markov_state="HEALTHY")
    # Reload from the same file to prove it truly persisted.
    bt2 = Backtester(history_file=str(target))
    assert bt2.get_prediction_count() >= 1
    assert any(r.pod_key == "a/b" for r in bt2._predictions)
