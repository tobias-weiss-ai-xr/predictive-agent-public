"""Repo-root compatibility shim for the predictive-agent configuration.

The canonical configuration module lives at ``predictive_agent/config.py`` and
is imported everywhere as ``from predictive_agent import config``. This file
exists at the repo root so tooling that greps ``config.py`` directly (e.g. the
AIO-6 acceptance gate, which requires ``REMEDIATION_DEDUP_WINDOW`` in
``config.py``) resolves to the same values without duplicating them.

It re-exports the package config module wholesale and is *not* imported by the
agent itself.
"""

from predictive_agent.config import *  # noqa: F401,F403
from predictive_agent.config import REMEDIATION_DEDUP_WINDOW  # noqa: F401
