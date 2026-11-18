#!/usr/bin/env bash
set -euo pipefail

# Predictive Agent v4.1.0 — Predictive Kubernetes Health Monitor
# Entrypoint: prepares runtime dirs then execs the Python reconcile loop.

echo "[INFO] === Predictive Agent v${OPERATOR_VERSION:-4.0.0} starting ==="
echo "[INFO] LLM Backend: ${LLM_BACKEND:-not set}"
echo "[INFO] Ollama URL: ${OLLAMA_URL:-not set}"
echo "[INFO] Ollama Model: ${OLLAMA_MODEL:-not set}"
echo "[INFO] Watch namespaces: ${OPERATOR_WATCH_NAMESPACES:-openedusuite,default,llm}"
echo "[INFO] Reconcile interval: ${RECONCILE_INTERVAL:-60}s"
echo "[INFO] Health probe: ${OPERATOR_HEALTH_PROBE_BIND_ADDRESS:-0.0.0.0:8081}"
echo "[INFO] Metrics bind: ${OPERATOR_METRICS_BIND_ADDRESS:-0.0.0.0:8080}"

# Create runtime directories (state PVC, logs, cache, kube config)
mkdir -p /var/lib/predictive-agent /var/log/predictive-agent /var/cache/predictive-agent /run/predictive-agent /tmp /home/predictive-agent/.kube

# Set PYTHONPATH so the predictive_agent package is importable
export PYTHONPATH=/opt/predictive-agent:${PYTHONPATH:-}

# Execute the Python operator (python3 is in PATH from the nix image Env).
exec python3 -m predictive_agent.main "$@"
