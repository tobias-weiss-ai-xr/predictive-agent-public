"""Test state persistence."""
import os
import tempfile

from predictive_agent.persistence import StateStore


def test_state_store_creation():
    """Test StateStore creation."""
    store = StateStore("/tmp/test-state-model.json")
    assert store.state_model_file == "/tmp/test-state-model.json"


def test_state_store_save_load_markov():
    """Test saving and loading Markov chain state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sm_file = os.path.join(tmpdir, "state-model.json")
        pred_file = os.path.join(tmpdir, "predictions.json")
        store = StateStore(sm_file, pred_file)

        # Save some state
        from predictive_agent.markov import MarkovChain
        mc = MarkovChain()
        mc.record_transition("HEALTHY", "DEGRADED")
        mc.record_transition("DEGRADED", "STRESSED")
        store.save_markov(mc)

        # Load it back
        mc2 = store.load_markov()
        assert mc2.total_transitions == 2


def test_state_store_save_load_predictions():
    """Test saving and loading predictions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sm_file = os.path.join(tmpdir, "state-model.json")
        pred_file = os.path.join(tmpdir, "predictions.json")
        store = StateStore(sm_file, pred_file)

        predictions = [
            {"pod": "ns/pod-0", "risk_score": 0.85, "ttf_minutes": 12},
            {"pod": "ns/pod-1", "risk_score": 0.3, "ttf_minutes": None},
        ]
        store.save_predictions(predictions)

        loaded = store.load_predictions()
        assert len(loaded) == 2
        assert loaded[0]["pod"] == "ns/pod-0"
        assert loaded[0]["risk_score"] == 0.85


def test_state_store_nonexistent_files():
    """Test loading from nonexistent files returns defaults."""
    store = StateStore("/nonexistent/state.json", "/nonexistent/pred.json")
    mc = store.load_markov()
    assert mc is not None
    assert mc.total_transitions == 0

    preds = store.load_predictions()
    assert preds == []


def test_state_store_save_markov_no_dir():
    """Test saving when directory doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sm_file = os.path.join(tmpdir, "subdir", "state-model.json")
        pred_file = os.path.join(tmpdir, "predictions.json")
        store = StateStore(sm_file, pred_file)

        from predictive_agent.markov import MarkovChain
        mc = MarkovChain()
        mc.record_transition("HEALTHY", "HEALTHY")
        store.save_markov(mc)  # Should create directory

        assert os.path.exists(sm_file)


# ── Seasonal state file ─────────────────────────────────────────────────────
# Separate file from state-model.json because state-model.json is rewritten
# every cycle and the seasonal blob is far larger; it is written on its own
# (throttled) cadence.

def test_save_and_load_seasonal_roundtrip(tmp_path):
    from predictive_agent.persistence import StateStore
    store = StateStore(str(tmp_path / "state-model.json"))
    payload = {"ns/p1": {"mem": {"ts": [1, 2], "vals": [1.0, 2.0]},
                         "cpu": {"ts": [1, 2], "vals": [3.0, 4.0]}}}
    store.save_seasonal(payload)
    assert (tmp_path / "seasonal-state.json").exists()
    assert store.load_seasonal() == payload


def test_load_seasonal_missing_or_corrupt_returns_empty(tmp_path):
    from predictive_agent.persistence import StateStore
    store = StateStore(str(tmp_path / "state-model.json"))
    assert store.load_seasonal() == {}
    (tmp_path / "seasonal-state.json").write_text("{not json")
    assert store.load_seasonal() == {}
    (tmp_path / "seasonal-state.json").write_text("[1,2,3]")
    assert store.load_seasonal() == {}


def test_seasonal_persist_is_throttled_and_drops_removed_pods(tmp_path, monkeypatch):
    """Seasonal state is saved on a slow cadence (not every cycle) and only
    for pods still tracked, so the file cannot grow without bound."""
    from unittest.mock import MagicMock

    from predictive_agent import config
    from predictive_agent import reconcile_phases as phases
    from predictive_agent.persistence import StateStore
    from predictive_agent.state_model import StateModel

    monkeypatch.setattr(config, "SEASONAL_PERSIST_ENABLED", True)
    monkeypatch.setattr(config, "SEASONAL_PERSIST_EVERY", 3)

    ctx = MagicMock()
    ctx.state_store = StateStore(str(tmp_path / "state-model.json"))
    ctx.state_model = StateModel()
    ctx.remediation_manager = None
    ctx.predictor = None
    ctx.self_health = None
    # Two tracked pods (seasonal must be enabled for a tracker to persist).
    for name in ("p1", "p2"):
        t = ctx.state_model.track_pod("ns", name)
        t.seasonal_mode = "shadow"
        t.update(memory_mib=100.0, memory_limit_mib=1000, cpu_m=50.0,
                 restart_count=0, log_errors=0, node_pressure=False)

    saved = []
    real = ctx.state_store.save_seasonal
    ctx.state_store.save_seasonal = lambda d: (saved.append(list(d)), real(d))

    # Cycles 1 and 2: throttled, nothing written. Cycle 3: written.
    for cycle in (1, 2):
        phases._persist_seasonal(ctx, cycle)
        assert saved == []
    phases._persist_seasonal(ctx, 3)
    assert saved and set(saved[0]) == {"ns/p1", "ns/p2"}

    # Pod p2 disappears -> next save must evict it.
    del ctx.state_model.pods["ns/p2"]
    phases._persist_seasonal(ctx, 6)
    assert set(saved[-1]) == {"ns/p1"}


def test_persist_seasonal_disabled_or_off_mode_writes_nothing(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from predictive_agent import config
    from predictive_agent import reconcile_phases as phases
    from predictive_agent.persistence import StateStore
    from predictive_agent.state_model import StateModel

    ctx = MagicMock()
    ctx.state_store = StateStore(str(tmp_path / "state-model.json"))
    ctx.state_model = StateModel()
    t = ctx.state_model.track_pod("ns", "p1")
    t.seasonal_mode = "shadow"
    t.update(memory_mib=100.0, memory_limit_mib=1000, cpu_m=50.0,
             restart_count=0, log_errors=0, node_pressure=False)

    monkeypatch.setattr(config, "SEASONAL_PERSIST_ENABLED", False)
    phases._persist_seasonal(ctx, 100)
    assert not (tmp_path / "seasonal-state.json").exists()


def test_restart_path_resumes_seasonal_warmup_end_to_end(tmp_path, monkeypatch):
    """The real failure mode: warm up half-way, 'restart', keep observing.

    Exercises the whole chain the operator uses -- reconcile persist ->
    StateStore file -> bootstrap load -> StateModel.track_pod restore.
    """
    from unittest.mock import MagicMock

    from predictive_agent import config
    from predictive_agent import reconcile_phases as phases
    from predictive_agent import state_model as state_model_mod
    from predictive_agent.persistence import StateStore
    from predictive_agent.state_model import StateModel

    monkeypatch.setattr(config, "SEASONAL_PERSIST_ENABLED", True)
    monkeypatch.setattr(config, "SEASONAL_PERSIST_EVERY", 1)
    # Small period so a fit is reachable inside the test (production is 1440).
    monkeypatch.setattr(state_model_mod, "_SEASONAL_TREND_MODE", "shadow")
    monkeypatch.setattr(state_model_mod, "_SEASONAL_PERIOD", 24)
    sm_file = str(tmp_path / "state-model.json")

    # ── run 1: warm up half the required window, then persist (cycle 1) ──
    ctx1 = MagicMock()
    ctx1.state_store = StateStore(sm_file)
    ctx1.state_model = StateModel()
    t1 = ctx1.state_model.track_pod("ns", "p1")
    t1.seasonal_mode = "shadow"
    need = t1.seasonal_cpu.samples_required
    assert t1.seasonal_cpu.season_len == 24
    for i in range(max(1, need // 2)):
        t1.update(memory_mib=100.0 + (i % 24), memory_limit_mib=1000,
                  cpu_m=50.0 + (i % 12), restart_count=0, log_errors=0,
                  node_pressure=False)
    warm_half = t1.seasonal_cpu.samples_seen
    assert 0 < warm_half < need
    phases._persist_seasonal(ctx1, 1)

    # ── 'restart': fresh state model, state loaded from the same PVC ──
    store2 = StateStore(sm_file)
    ctx2 = MagicMock()
    ctx2.state_store = store2
    ctx2.state_model = StateModel()
    ctx2.state_model.seasonal_restore = store2.load_seasonal()
    assert "ns/p1" in ctx2.state_model.seasonal_restore

    # track_pod itself consumes the restored payload for this pod.
    t2 = ctx2.state_model.track_pod("ns", "p1")
    assert "ns/p1" not in ctx2.state_model.seasonal_restore, "restore must be consumed"
    t2.seasonal_mode = "shadow"
    assert t2.seasonal_cpu.samples_seen == warm_half, "warm-up must survive restart"

    # ── run 2: finish the window -> the model finally fits ──
    for i in range(need + 10):
        t2.update(memory_mib=100.0 + (i % 24), memory_limit_mib=1000,
                  cpu_m=50.0 + (i % 12), restart_count=0, log_errors=0,
                  node_pressure=False)
    assert t2.seasonal_cpu.fitted, "warm-up stitched across restart must fit"
    assert t2.seasonal_cpu.warmup_progress >= 1.0
