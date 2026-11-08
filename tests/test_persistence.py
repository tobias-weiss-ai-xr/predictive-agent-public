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
