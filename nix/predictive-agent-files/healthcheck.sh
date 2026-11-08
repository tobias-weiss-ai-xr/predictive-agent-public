#!/usr/bin/env bash
set -euo pipefail

# openDesk Predictive Agent v4.0 — container healthcheck
# Probes the HTTP health server (default port 8081).
#   liveness  -> GET /healthz
#   readiness -> GET /ready
#
# The health port is derived from OPERATOR_HEALTH_PROBE_BIND_ADDRESS (host:port).

HEALTH_ADDR="${OPERATOR_HEALTH_PROBE_BIND_ADDRESS:-0.0.0.0:8081}"
HEALTH_PORT="${HEALTH_ADDR##*:}"

case "${1:-liveness}" in
  liveness)
    if curl -sf "http://localhost:${HEALTH_PORT}/healthz" >/dev/null 2>&1; then
      echo "OK: liveness"
      exit 0
    fi
    echo "FAIL: liveness"
    exit 1
    ;;
  readiness)
    if curl -sf "http://localhost:${HEALTH_PORT}/ready" >/dev/null 2>&1; then
      echo "OK: readiness"
      exit 0
    fi
    echo "FAIL: readiness"
    exit 1
    ;;
  *)
    echo "Usage: $0 {liveness|readiness}"
    exit 1
    ;;
esac
