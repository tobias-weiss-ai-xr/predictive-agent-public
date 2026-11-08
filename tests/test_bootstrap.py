"""Tests for predictive_agent.bootstrap (PA-S3b).

Exercises the component bootstrap sequence without running main(): the
docker vs kubernetes action-registration branch, knowledge-graph
unavailability, OPA failure fallback, LLM backend selection, and persisted
state loading. All constructors/factories are injected via BootstrapHooks
so the tests don't touch real sockets.
"""

from unittest.mock import MagicMock

import pytest

import predictive_agent.bootstrap as bootstrap
import predictive_agent.config as config
from predictive_agent.bootstrap import BootstrapHooks


def _make_hooks(runtime="kubernetes", **over):
    base = dict(
        detect_runtime=lambda: runtime,
        get_watch_selectors=lambda: {},
        state_store_cls=MagicMock(),
        predictor_cls=lambda **kw: MagicMock(risk_threshold=kw.get("risk_threshold")),
        state_model_cls=lambda: MagicMock(),
        create_remediation_manager=lambda: MagicMock(),
        docker_socket_client_cls=lambda sock: MagicMock(),
        create_notifier=lambda: "notifier-instance",
        get_kg_client=lambda: None,
        get_policy_engine=lambda **kw: MagicMock(),
    )
    base.update(over)
    return BootstrapHooks(**base)


def test_bootstrap_kubernetes_registers_kubectl_actions():
    seen = []
    def make_mgr():
        m = MagicMock()
        m.register_action.side_effect = lambda a: seen.append(type(a).__name__)
        m.get_stats.return_value = {"registered_actions": 5}
        m.rl_policy = None
        return m
    hooks = _make_hooks(runtime="kubernetes", create_remediation_manager=make_mgr)
    _cs = bootstrap.initialize_components("kubernetes", hooks)
    # Kubernetes path: PodRestartAction registered (container restart only in docker)
    assert "PodRestartAction" in seen
    assert "ContainerRestartAction" not in seen


def test_bootstrap_docker_registers_docker_restart():
    seen = []
    def make_mgr():
        m = MagicMock()
        m.register_action.side_effect = lambda a: seen.append(type(a).__name__)
        m.get_stats.return_value = {"registered_actions": 5}
        m.rl_policy = None
        return m
    hooks = _make_hooks(runtime="docker", create_remediation_manager=make_mgr)
    _cs = bootstrap.initialize_components("docker", hooks)
    # Docker path: the double detect_runtime()==docker branch registers a
    # ContainerRestartAction AND the runtime=="docker" guard adds another.
    assert seen.count("ContainerRestartAction") >= 1
    assert "NodeCordonAction" in seen


def test_bootstrap_kg_unavailable_sets_none():
    hooks = _make_hooks(get_kg_client=lambda: None)
    _cs = bootstrap.initialize_components("kubernetes", hooks)
    assert _cs.kg_client is None


def test_bootstrap_kg_available_when_healthy():
    kg = MagicMock()
    kg.health.return_value = True
    hooks = _make_hooks(get_kg_client=lambda: kg)
    _cs = bootstrap.initialize_components("kubernetes", hooks)
    assert _cs.kg_client is kg


def _boom(**kw):
    raise RuntimeError("opa down")


def test_bootstrap_opa_error_falls_back_to_none():
    hooks = _make_hooks(get_policy_engine=_boom)
    _cs = bootstrap.initialize_components("kubernetes", hooks)
    assert _cs.policy_engine is None


def test_bootstrap_persisted_state_loaded():
    store = MagicMock()
    store.load_markov.return_value = "markov-chain"
    store.load_decision_model.return_value = {}  # falsy → no load log, but no crash
    store.load_q_table.return_value = None
    hooks = _make_hooks(
        state_store_cls=lambda **kw: store,
        create_remediation_manager=lambda: MagicMock(rl_policy=None,
                                                     get_stats=lambda: {"registered_actions": 5}),
    )
    _cs = bootstrap.initialize_components("kubernetes", hooks)
    store.load_markov.assert_called_once()


def test_bootstrap_llm_backend_ollama():
    # Patch the lazy imports inside _initialize_llm_analyzer via the llm mod.
    import predictive_agent.llm as llm_mod
    fake_backend = MagicMock()
    fake_backend.OLLAMA = "ollama"
    analyzer_cls = MagicMock()
    orig = config.LLM_BACKEND
    try:
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(llm_mod, "LLMAnalyzer", analyzer_cls)
            mp.setattr(llm_mod, "LLMBackend", fake_backend)
            mp.setattr(config, "LLM_BACKEND", "ollama")
            cs = bootstrap.initialize_components("kubernetes", _make_hooks(
                create_remediation_manager=lambda: MagicMock(rl_policy=None,
                                                             get_stats=lambda: {"registered_actions": 5})))
        analyzer_cls.assert_called_once()
        assert cs.llm_analyzer is not None
    finally:
        config.LLM_BACKEND = orig


def test_bootstrap_llm_unknown_backend_returns_none():
    orig = config.LLM_BACKEND
    try:
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(config, "LLM_BACKEND", "bogus")
            cs = bootstrap.initialize_components("kubernetes", _make_hooks(
                create_remediation_manager=lambda: MagicMock(rl_policy=None,
                                                             get_stats=lambda: {"registered_actions": 5})))
        assert cs.llm_analyzer is None
    finally:
        config.LLM_BACKEND = orig


def test_bootstrap_returns_component_set():
    cs = bootstrap.initialize_components("kubernetes", _make_hooks(
        create_remediation_manager=lambda: MagicMock(rl_policy=None,
                                                     get_stats=lambda: {"registered_actions": 5})))
    for f in ("state_model", "predictor", "state_store", "remediation_manager",
              "notifier", "backtester"):
        assert hasattr(cs, f)


@pytest.mark.parametrize("backend_str,enum_name", [
    ("openai", "OPENAI"),
    ("saia", "SAIA"),
    ("tud", "TUD"),
])
def test_bootstrap_llm_backend_api_backends(backend_str, enum_name):
    """openai/saia/tud backends construct an analyzer with the right enum."""
    import predictive_agent.llm as llm_mod
    fake_backend = MagicMock()
    fake_backend.OLLAMA = "ollama"
    setattr(fake_backend, enum_name, backend_str)
    analyzer_cls = MagicMock()
    orig = config.LLM_BACKEND
    try:
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(llm_mod, "LLMAnalyzer", analyzer_cls)
            mp.setattr(llm_mod, "LLMBackend", fake_backend)
            mp.setattr(config, "LLM_BACKEND", backend_str)
            cs = bootstrap.initialize_components("kubernetes", _make_hooks(
                create_remediation_manager=lambda: MagicMock(rl_policy=None,
                                                             get_stats=lambda: {"registered_actions": 5})))
        assert cs.llm_analyzer is analyzer_cls.return_value
        kwargs = analyzer_cls.call_args.kwargs
        assert kwargs["backend"] == backend_str
        # url/api_key/model wired from the backend's config values (may be
        # empty in envs without credentials — the wiring is what we assert).
        url_key = {"openai": "OPENAI_API_URL", "saia": "SAIA_API_URL", "tud": "TUD_API_URL"}[backend_str]
        key_key = {"openai": "OPENAI_API_KEY", "saia": "SAIA_API_KEY", "tud": "TUD_API_KEY"}[backend_str]
        assert kwargs["url"] == getattr(config, url_key)
        assert kwargs["api_key"] == getattr(config, key_key)
    finally:
        config.LLM_BACKEND = orig


def test_bootstrap_llm_analyzer_init_error_returns_none():
    """An analyzer constructor failure must disable LLM analysis, not crash boot."""
    import predictive_agent.llm as llm_mod
    analyzer_cls = MagicMock(side_effect=RuntimeError("backend unreachable"))
    orig = config.LLM_BACKEND
    try:
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(llm_mod, "LLMAnalyzer", analyzer_cls)
            mp.setattr(llm_mod, "LLMBackend", MagicMock(OLLAMA="ollama"))
            mp.setattr(config, "LLM_BACKEND", "ollama")
            cs = bootstrap.initialize_components("kubernetes", _make_hooks(
                create_remediation_manager=lambda: MagicMock(rl_policy=None,
                                                             get_stats=lambda: {"registered_actions": 5})))
        assert cs.llm_analyzer is None
    finally:
        config.LLM_BACKEND = orig


# ── symmetric error paths: each optional subsystem must degrade, not abort ──

def test_bootstrap_corrupt_markov_state_degrades():
    store = MagicMock()
    store.load_markov.side_effect = RuntimeError("corrupt")
    hooks = _make_hooks(state_store_cls=lambda **kw: store)
    _cs = bootstrap.initialize_components("kubernetes", hooks)  # must not raise
    store.load_markov.assert_called_once()


def test_bootstrap_calibration_monitor_failure_degrades():
    # CalibrationMonitor import/construct failure → warning, startup continues
    import sys
    real = sys.modules.pop("predictive_agent.calibration", None)
    sys.modules["predictive_agent.calibration"] = None  # import raises ImportError
    try:
        hooks = _make_hooks()
        _cs = bootstrap.initialize_components("kubernetes", hooks)
    finally:
        sys.modules.pop("predictive_agent.calibration", None)
        if real is not None:
            sys.modules["predictive_agent.calibration"] = real


def test_bootstrap_qlearning_failure_degrades():
    store = MagicMock()
    store.load_q_table.side_effect = RuntimeError("corrupt q")
    hooks = _make_hooks(state_store_cls=lambda **kw: store)
    _cs = bootstrap.initialize_components("kubernetes", hooks)  # must not raise
