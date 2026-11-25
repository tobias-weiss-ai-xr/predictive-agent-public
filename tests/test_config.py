"""Config loading tests: env-driven operator version."""
import importlib
import importlib.util
import os
from pathlib import Path

# Loading config.py as a fresh module under a controlled environment, so we
# can assert `OPERATOR_VERSION` honours the deployment env instead of the
# hard-coded fallback. config.py is re-imported here dynamically to avoid
# caching a stale module across test runs.
_MODULE_PATH = Path(__file__).resolve().parent.parent / "predictive_agent" / "config.py"


def _load_config_env(env_overrides):
    saved = {k: os.environ.get(k) for k in env_overrides}
    for k, v in env_overrides.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    spec = importlib.util.spec_from_file_location("config_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        # restore environment for other tests
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return module


def test_operator_version_reads_env():
    m = _load_config_env({"OPERATOR_VERSION": "4.1.8"})
    assert m.OPERATOR_VERSION == "4.1.8"


def test_operator_version_absent_env_falls_back():
    m = _load_config_env({"OPERATOR_VERSION": None})
    assert m.OPERATOR_VERSION == "4.1.0"


def test_operator_name_and_namespace_env():
    m = _load_config_env({"OPERATOR_NAME": "pa-test", "OPERATOR_NAMESPACE": "ns-test"})
    assert m.OPERATOR_NAME == "pa-test"
    assert m.OPERATOR_NAMESPACE == "ns-test"
