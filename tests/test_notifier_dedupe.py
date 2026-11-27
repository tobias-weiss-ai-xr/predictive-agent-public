"""C10 — alert-noise dedupe at the NotificationManager boundary.

Last-resort filter on top of the reconcile-layer per-key throttle windows:
(1) an exact repeat of an identical alert within NOTIFY_DEDUPE_WINDOW is
dropped, and (2) a burst cap of NOTIFY_BURST_LIMIT deliveries per
NOTIFY_BURST_WINDOW stops any N-webhook restart flood. Deduped/capped calls
still land in history (marked deduped=True) so nothing is silently lost.
"""

from unittest import mock

import pytest

from predictive_agent import config, notifier
from predictive_agent.notifier import NotificationManager


@pytest.fixture
def clock(monkeypatch):
    """Injectable time.time() for deterministic window arithmetic."""
    state = {"now": 1_000_000.0}

    class _Clock:
        @staticmethod
        def time():
            return state["now"]

    monkeypatch.setattr(notifier, "_time", _Clock)
    return state


def _mgr(**kw):
    webhook = mock.MagicMock()
    webhook.send.return_value = True
    m = NotificationManager(webhook_notifier=webhook, channel="matrix", **kw)
    return m, webhook


# ─── identical-alert dedupe ────────────────────────────────────────────────
def test_identical_alert_dropped_within_window(clock):
    m, webhook = _mgr()
    first = m.notify("remediation", "pod-1", 0.9, "pod_restart", "crash loop")
    assert first == (False, True)
    clock["now"] += 60  # inside the 300s window
    second = m.notify("remediation", "pod-1", 0.9, "pod_restart", "crash loop")
    assert second == (False, False), "identical repeat within window must be dropped"
    assert webhook.send.call_count == 1
    assert m.history[-1]["deduped"] is True
    assert m.history[-1]["email_sent"] is False and m.history[-1]["webhook_sent"] is False
    assert m.deduped_count == 1


def test_identical_alert_resent_after_window(clock):
    m, webhook = _mgr()
    m.notify("remediation", "pod-1", 0.9, "pod_restart", "crash loop")
    clock["now"] += 301
    m.notify("remediation", "pod-1", 0.9, "pod_restart", "crash loop")
    assert webhook.send.call_count == 2, "repeat after the window must be delivered"


def test_identical_alert_not_refreshed_by_dropped_repeat(clock):
    """A dropped repeat must not extend the dedupe stamp — the alert can
    fire again as soon as the original window has elapsed."""
    m, _ = _mgr()
    m.notify("remediation", "pod-1", 0.9, "pod_restart", "crash loop")
    clock["now"] += 60
    m.notify("remediation", "pod-1", 0.9, "pod_restart", "crash loop")  # dropped
    clock["now"] += 250  # 310s after the ORIGINAL send -> must deliver again
    assert m.notify("remediation", "pod-1", 0.9, "pod_restart", "crash loop") == (False, True)


def test_different_status_not_deduped(clock):
    """'resolved' must never be collapsed into a prior 'firing' — it closes
    the Alertmanager incident."""
    m, webhook = _mgr()
    m.notify("plan", "api-server", 0.9, "", "at risk", status="firing")
    clock["now"] += 10
    m.notify("plan", "api-server", 0.9, "", "at risk", status="resolved")
    assert webhook.send.call_count == 2


def test_different_details_not_deduped(clock):
    m, webhook = _mgr()
    m.notify("remediation", "pod-1", 0.9, "pod_restart", "details A")
    clock["now"] += 10
    m.notify("remediation", "pod-1", 0.9, "pod_restart", "details B")
    assert webhook.send.call_count == 2


def test_dedupe_disabled_by_config(clock):
    m, webhook = _mgr()
    with mock.patch.object(config, "NOTIFY_DEDUPE_ENABLED", False):
        m.notify("remediation", "pod-1", 0.9, "pod_restart", "crash loop")
        clock["now"] += 10
        m.notify("remediation", "pod-1", 0.9, "pod_restart", "crash loop")
    assert webhook.send.call_count == 2


# ─── burst cap ─────────────────────────────────────────────────────────────
def test_burst_cap_drops_after_limit(clock):
    m, webhook = _mgr()
    with mock.patch.object(config, "NOTIFY_BURST_LIMIT", 3), \
         mock.patch.object(config, "NOTIFY_BURST_WINDOW", 600):
        for i in range(5):
            m.notify("alert", f"pod-{i}", 0.8, "act")  # all distinct fingerprints
    assert webhook.send.call_count == 3, "only NOTIFY_BURST_LIMIT deliveries allowed"
    assert m.capped_count == 2
    # All five calls are auditable in history; last two marked deduped.
    history = m.get_history(limit=10)
    assert len(history) == 5
    assert [h["deduped"] for h in history] == [False, False, False, True, True]


def test_burst_cap_resets_after_window(clock):
    m, webhook = _mgr()
    with mock.patch.object(config, "NOTIFY_BURST_LIMIT", 2), \
         mock.patch.object(config, "NOTIFY_BURST_WINDOW", 600):
        for i in range(2):
            m.notify("alert", f"pod-{i}", 0.8, "act")
        clock["now"] += 601
        m.notify("alert", "pod-2", 0.8, "act")
    assert webhook.send.call_count == 3


# ─── stats / observability ─────────────────────────────────────────────────
def test_sent_entries_recorded_as_not_deduped():
    m, _ = _mgr()
    m.notify("remediation", "pod-1", 0.9, "pod_restart", "crash loop")
    assert m.history[-1]["deduped"] is False
    assert m.sent_count == 1


def test_dedupe_stats():
    m, _ = _mgr()
    m.notify("alert", "p1", 0.8, "act")
    m.notify("alert", "p1", 0.8, "act")  # identical -> deduped
    stats = m.dedupe_stats()
    assert stats["sent_count"] == 1
    assert stats["deduped_count"] == 1
    assert stats["capped_count"] == 0
    assert stats["window_s"] == config.NOTIFY_DEDUPE_WINDOW
