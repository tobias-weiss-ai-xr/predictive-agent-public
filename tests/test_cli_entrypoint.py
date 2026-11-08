"""Regression test for the ``predictive_agent`` CLI entrypoint (task 4.1).

Covers ``python -m predictive_agent`` (the `__main__` path that was at
0% coverage). The entrypoint has no ``--help``: it calls ``main()``
unconditionally, which starts the reconcile loop + HTTP servers, so we
assert IMPORTABILITY only (that the module and its ``main`` callable
resolve without ImportError / ModuleNotFoundError) — never actually
running the loop in the test process.
"""

import subprocess
import sys


def test_entrypoint_main_is_callable():
    """``predictive_agent.__main__`` exposes a callable ``main``."""
    from predictive_agent import __main__ as entrypoint
    assert callable(entrypoint.main)


def test_subprocess_import_does_not_raise_importerror():
    """`python -m predictive_agent` must import cleanly (no missing deps).

    We import only — never execute the entrypoint body — so the test is
    fast and side-effect free (no port binds, no reconcile loop).
    """
    result = subprocess.run(
        [sys.executable, "-c",
         "import predictive_agent.__main__"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "ImportError" not in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
