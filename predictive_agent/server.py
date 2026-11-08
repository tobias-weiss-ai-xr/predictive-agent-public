"""HTTP server exposing health, metrics, predictions, and state endpoints.

The server is started with :func:`start_server`, which accepts an optional
:class:`~predictive_agent.state_model.StateModel` and
:class:`~predictive_agent.predictor.Predictor`. When provided, the API
endpoints serve *real* data (live pod states, real predictions, real Prometheus
gauges). When omitted (e.g. in the test suite), the endpoints return sensible
empty defaults so the server always responds correctly.
"""

import json
import socketserver
import threading
import time
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from predictive_agent import config, main
from predictive_agent.benchmark import run_benchmark, scenarios_from_backtester
from predictive_agent.collector import (
    COLLECTOR_STATE_DEGRADED,
    COLLECTOR_STATE_DOWN,
    COLLECTOR_STATE_HEALTHY,
    get_collector_state,
    get_collector_state_numeric,
)
from predictive_agent.markov import MarkovChain
from predictive_agent.metrics import collect_decision_metrics, render_prometheus
from predictive_agent.pomdp import ACTION_TYPES, ActionAwareTransitionModel, ValueIterationPolicy


# ─── Shared server context ───────────────────────────────────────────────────
class _ServerState:
    """Mutable container shared between request handlers and the operator.

    Populated by :func:`start_server` so that handlers can serve real data
    without relying on hard module-level globals that are awkward to reset
    between test runs / server restarts.
    """

    __slots__ = (
        "state_model",
        "predictor",
        "cache",
        "reconcile_callback",
        "history",
        "start_time",
        "remediation_manager",
        "notifier",
        "backtester",
        "policy_engine",
        "reconcile_count",
        "reconcile_duration",
        "at_risk_count",
        "llm_calls",
        "llm_errors",
        "state_saves",
        "remediation_plans",
    )

    def __init__(self):
        self.state_model = None
        self.predictor = None
        self.cache = None
        self.reconcile_callback = None
        self.history = []
        self.start_time = time.time()
        self.remediation_manager = None
        self.notifier = None
        self.backtester = None
        self.policy_engine = None
        self.reconcile_count = 0
        self.reconcile_duration = 0.0
        self.at_risk_count = 0
        self.llm_calls = 0
        self.llm_errors = 0
        self.state_saves = 0
        self.remediation_plans: dict = {}


# Default context used when start_server is called without arguments (e.g. the
# test suite). start_server replaces this with a fresh, populated instance.
_context = _ServerState()


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _all_predictions(predictor):
    """Return all stored PredictionResult objects from a Predictor."""
    if predictor is None:
        return []
    preds = getattr(predictor, "_predictions", None)
    if not preds:
        # Fall back to the public at-risk API if the internal store is empty
        getter = getattr(predictor, "get_at_risk", None)
        if callable(getter):
            try:
                return list(getter())
            except Exception:
                return []
        return []
    return list(preds.values())


def _serialize_prediction(result):
    """Convert a PredictionResult into a JSON-serializable dict.

    Exposes ``ttf`` as an alias for ``ttf_minutes`` per the HTTP API contract.
    """
    if is_dataclass(result) and not isinstance(result, type):
        data = asdict(result)
    else:
        data = dict(getattr(result, "__dict__", {}))
    if "ttf_minutes" in data:
        data.setdefault("ttf", data["ttf_minutes"])
    pk = getattr(result, "pod_key", None)
    if pk is None and isinstance(result, dict):
        pk = result.get("pod_key")
    # Surface the planned intervention sequence computed during reconcile so
    # operators see *what the agent intends to do*, not just the risk score.
    data["planned_actions"] = _context.remediation_plans.get(pk)
    return data


# ─── Request handler ─────────────────────────────────────────────────────────
class RequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler serving all metrics/API and health endpoints."""

    def _send_response(self, data, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        if isinstance(data, (dict, list)):
            self.wfile.write(json.dumps(data).encode())
        else:
            self.wfile.write(data.encode())

    # ── Routing ──────────────────────────────────────────────────────────────
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/healthz":
            self._send_response({"status": "ok"})
        elif path == "/ready":
            # P4: Degraded-Mode Protocol - return 503 if collector is degraded or down
            collector_state = get_collector_state()
            if collector_state in (COLLECTOR_STATE_DEGRADED, COLLECTOR_STATE_DOWN):
                self._send_response(
                    {"status": "not_ready", "collector_state": collector_state},
                    status=503
                )
            else:
                self._send_response({"status": "ready", "collector_state": collector_state})
        elif path == "/metrics":
            self._send_response(self._metrics_text(),
                                content_type="text/plain; version=0.0.4; charset=utf-8")
        elif path == "/status":
            self._send_response(self._status_dict())
        elif path == "/predictions":
            self._send_response(self._predictions_dict())
        elif path == "/state":
            self._send_response(self._state_dict())
        elif path == "/history":
            self._send_response(_context.history if _context.history else [])
        elif path == "/reanalyze":
            data, status = self._reanalyze()
            self._send_response(data, status=status)
        elif path == "/cache":
            self._send_response(self._cache_dict())
        elif path == "/remediate":
            self._send_response(self._remediate_get())
        elif path == "/remediation-log":
            self._send_response(self._remediation_log_get())
        elif path == "/remediation/state":
            self._send_response(self._remediation_state_get())
        elif path == "/plan":
            data, status = self._plan_get()
            self._send_response(data, status=status)
        elif path == "/world-model":
            data, status = self._world_model_get()
            self._send_response(data, status=status)
        elif path == "/notifications":
            self._send_response(self._notifications_get())
        elif path == "/backtest":
            self._send_response(self._backtest_get())
        elif path == "/benchmark":
            data, status = self._benchmark_get()
            self._send_response(data, status=status)
        elif path == "/review":
            data, status = self._review_get()
            self._send_response(data, status=status)
        elif path == "/shadow":
            data, status = self._shadow_get()
            self._send_response(data, status=status)
        elif path == "/calibration":
            data, status = self._calibration_get()
            self._send_response(data, status=status)
        elif path == "/decision-metrics":
            data, status = self._decision_metrics_get()
            self._send_response(data, status=status)
        elif path == "/cost-model":
            data, status = self._cost_model_get()
            self._send_response(data, status=status)
        elif path == "/policy/health":
            self._send_response(self._policy_health())
        elif path == "/policy/stats":
            self._send_response(self._policy_stats())
        else:
            self._send_response({"error": "Not Found"}, status=404)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/reanalyze":
            data, status = self._reanalyze()
            self._send_response(data, status=status)
        elif path == "/remediate":
            data, status = self._remediate_post()
            self._send_response(data, status=status)
        elif path == "/plan":
            data, status = self._plan_post()
            self._send_response(data, status=status)
        elif path == "/cost-model":
            data, status = self._cost_model_post()
            self._send_response(data, status=status)
        elif path == "/coordinate":
            data, status = self._coordinate_post()
            self._send_response(data, status=status)
        elif path == "/review":
            data, status = self._review_post()
            self._send_response(data, status=status)
        else:
            self._send_response({"error": "Not Found"}, status=404)

    # ── Endpoint payloads ────────────────────────────────────────────────────
    # State name to numeric mapping for Prometheus
    _STATE_MAP = {
        "HEALTHY": 0, "STABLE": 1, "DEGRADED": 2,
        "WARNING": 3, "CRITICAL": 4, "FAILED": 5,
    }

    def _metrics_text(self):
        """Prometheus-format metrics with real gauges from state/predictor."""
        sm = _context.state_model
        pred = _context.predictor

        # Thread-safe snapshot of pods
        if sm is not None:
            pods_snapshot = getattr(sm, 'get_pods_snapshot', lambda: dict(sm.pods))()
            pods_tracked = len(pods_snapshot)
        else:
            pods_snapshot = {}
            pods_tracked = 0
        predictions = _all_predictions(pred)
        predictions_count = len(predictions)
        risk_score = max((p.risk_score for p in predictions), default=0.0)
        uptime = int(time.time() - _context.start_time)
        at_risk = _context.at_risk_count
        reconcile_count = _context.reconcile_count
        reconcile_duration = _context.reconcile_duration
        llm_calls = _context.llm_calls
        llm_errors = _context.llm_errors
        state_saves = _context.state_saves
        collector_state = get_collector_state()
        collector_up = 1 if collector_state == COLLECTOR_STATE_HEALTHY else 0

        lines = [
            "# HELP opendesk_predictive_agent_collector_up Whether metric/pod collection currently succeeds (1=HEALTHY)",
            "# TYPE opendesk_predictive_agent_collector_up gauge",
            f"opendesk_predictive_agent_collector_up {collector_up}",
            "# HELP opendesk_predictive_agent_pods_tracked Number of pods currently tracked",
            "# TYPE opendesk_predictive_agent_pods_tracked gauge",
            f"opendesk_predictive_agent_pods_tracked {pods_tracked}",
            "# HELP opendesk_predictive_agent_pods_at_risk Number of pods at risk (risk >= threshold)",
            "# TYPE opendesk_predictive_agent_pods_at_risk gauge",
            f"opendesk_predictive_agent_pods_at_risk {at_risk}",
            "# HELP opendesk_predictive_agent_predictions_count Number of active predictions",
            "# TYPE opendesk_predictive_agent_predictions_count gauge",
            f"opendesk_predictive_agent_predictions_count {predictions_count}",
            "# HELP opendesk_predictive_agent_risk_score Highest current risk score (0-1)",
            "# TYPE opendesk_predictive_agent_risk_score gauge",
            f"opendesk_predictive_agent_risk_score {risk_score}",
            "# HELP opendesk_predictive_agent_uptime_seconds Uptime in seconds",
            "# TYPE opendesk_predictive_agent_uptime_seconds gauge",
            f"opendesk_predictive_agent_uptime_seconds {uptime}",
            "# HELP opendesk_predictive_agent_reconcile_total Total reconcile cycles",
            "# TYPE opendesk_predictive_agent_reconcile_total counter",
            f"opendesk_predictive_agent_reconcile_total {reconcile_count}",
            "# HELP opendesk_predictive_agent_reconcile_duration_seconds Duration of last reconcile in seconds",
            "# TYPE opendesk_predictive_agent_reconcile_duration_seconds gauge",
            f"opendesk_predictive_agent_reconcile_duration_seconds {reconcile_duration}",
            "# HELP opendesk_predictive_agent_llm_calls_total Total LLM analysis calls",
            "# TYPE opendesk_predictive_agent_llm_calls_total counter",
            f"opendesk_predictive_agent_llm_calls_total {llm_calls}",
            "# HELP opendesk_predictive_agent_llm_errors_total Total LLM analysis errors",
            "# TYPE opendesk_predictive_agent_llm_errors_total counter",
            f"opendesk_predictive_agent_llm_errors_total {llm_errors}",
            "# HELP opendesk_predictive_agent_state_saves_total Total state model saves to disk",
            "# TYPE opendesk_predictive_agent_state_saves_total counter",
            f"opendesk_predictive_agent_state_saves_total {state_saves}",
            "# HELP predictive_collector_state Collector state gauge (0=down, 1=degraded, 2=healthy)",
            "# TYPE predictive_collector_state gauge",
            f"predictive_collector_state {get_collector_state_numeric()}",
            "# HELP opendesk_predictive_agent_pod_risk_score Per-pod risk score (0-1)",
            "# TYPE opendesk_predictive_agent_pod_risk_score gauge",
        ]
        # Per-pod risk score metrics
        for p in predictions:
            pod_key = p.pod_key if hasattr(p, 'pod_key') else p.get('pod_key', 'unknown')
            risk = p.risk_score if hasattr(p, 'risk_score') else p.get('risk_score', 0)
            lines.append(f"opendesk_predictive_agent_pod_risk_score{{pod=\"{pod_key}\"}} {risk}")
        # Risk score histogram
        risk_values = [p.risk_score for p in predictions if hasattr(p, 'risk_score')]
        buckets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        lines.append("# HELP opendesk_predictive_agent_risk_score_bucket Distribution of risk scores")
        lines.append("# TYPE opendesk_predictive_agent_risk_score_bucket histogram")
        for le in buckets:
            count = sum(1 for r in risk_values if r <= le)
            lines.append(f"opendesk_predictive_agent_risk_score_bucket{{le=\"{le}\"}} {count}")
        lines.append('opendesk_predictive_agent_risk_score_bucket{le="+Inf"} ' + str(len(risk_values)))
        lines.append(f"opendesk_predictive_agent_risk_score_sum {sum(risk_values):.4f}")
        lines.append(f"opendesk_predictive_agent_risk_score_count {len(risk_values)}")
        # Per-pod detailed metrics (CPU trend, memory trend, restart count, state, confidence)
        if sm is not None:
            lines.append("# HELP opendesk_predictive_agent_pod_cpu_trend Per-pod CPU trend (millicores/min)")
            lines.append("# TYPE opendesk_predictive_agent_pod_cpu_trend gauge")
            lines.append("# HELP opendesk_predictive_agent_pod_mem_trend Per-pod memory trend (MiB/min)")
            lines.append("# TYPE opendesk_predictive_agent_pod_mem_trend gauge")
            lines.append("# HELP opendesk_predictive_agent_pod_restart_count Per-pod restart count")
            lines.append("# TYPE opendesk_predictive_agent_pod_restart_count gauge")
            lines.append("# HELP opendesk_predictive_agent_pod_state Per-pod state (0=healthy, 5=failed)")
            lines.append("# TYPE opendesk_predictive_agent_pod_state gauge")
            lines.append("# HELP opendesk_predictive_agent_kalman_confidence Per-pod Kalman filter confidence (0-1)")
            lines.append("# TYPE opendesk_predictive_agent_kalman_confidence gauge")
            lines.append("# HELP opendesk_predictive_agent_markov_transition Per-pod Markov transition probability")
            lines.append("# TYPE opendesk_predictive_agent_markov_transition gauge")
            lines.append("# HELP opendesk_predictive_agent_pod_anomaly_score Per-pod anomaly score (0=normal, >3=anomalous)")
            lines.append("# TYPE opendesk_predictive_agent_pod_anomaly_score gauge")
            for pod_key, tracker in pods_snapshot.items():
                cpu_trend = getattr(tracker, 'cpu_trend', 0.0) or 0.0
                mem_trend = getattr(tracker, 'memory_trend', 0.0) or 0.0
                restart_count = getattr(tracker, 'restart_count', 0)
                state_num = self._STATE_MAP.get(getattr(tracker, 'state', 'HEALTHY'), 0)
                # Find matching prediction for confidence and markov state
                confidence = 0.0
                markov_p = 0.0
                anomaly_score = 0.0
                for p in predictions:
                    pk = p.pod_key if hasattr(p, 'pod_key') else p.get('pod_key', '')
                    if pk == pod_key:
                        confidence = p.confidence if hasattr(p, 'confidence') else p.get('confidence', 0.0)
                        markov_p = p.markov_p_critical if hasattr(p, 'markov_p_critical') else 0.0
                        anomaly_score = p.anomaly_score if hasattr(p, 'anomaly_score') else 0.0
                        break
                lines.append(f"opendesk_predictive_agent_pod_cpu_trend{{pod=\"{pod_key}\"}} {cpu_trend}")
                lines.append(f"opendesk_predictive_agent_pod_mem_trend{{pod=\"{pod_key}\"}} {mem_trend}")
                lines.append(f"opendesk_predictive_agent_pod_restart_count{{pod=\"{pod_key}\"}} {restart_count}")
                lines.append(f"opendesk_predictive_agent_pod_state{{pod=\"{pod_key}\"}} {state_num}")
                lines.append(f"opendesk_predictive_agent_kalman_confidence{{pod=\"{pod_key}\"}} {confidence}")
                lines.append(f"opendesk_predictive_agent_markov_transition{{pod=\"{pod_key}\"}} {markov_p}")
                lines.append(f"opendesk_predictive_agent_pod_anomaly_score{{pod=\"{pod_key}\"}} {anomaly_score}")
        # Remediation metrics (REM-8)
        rem = _context.remediation_manager
        if rem is not None:
            stats = rem.get_stats()
            lines.extend([
                "# HELP opendesk_predictive_agent_remediation_actions_total Total remediation actions",
                "# TYPE opendesk_predictive_agent_remediation_actions_total counter",
                f"opendesk_predictive_agent_remediation_actions_total {stats['total_actions']}",
                "# HELP opendesk_predictive_agent_remediation_successful_total Successful remediation actions",
                "# TYPE opendesk_predictive_agent_remediation_successful_total counter",
                f"opendesk_predictive_agent_remediation_successful_total {stats['successful_actions']}",
                "# HELP opendesk_predictive_agent_remediation_failed_total Failed remediation actions",
                "# TYPE opendesk_predictive_agent_remediation_failed_total counter",
                f"opendesk_predictive_agent_remediation_failed_total {stats['failed_actions']}",
                "# HELP opendesk_predictive_agent_remediation_dry_run_total Dry-run remediation actions",
                "# TYPE opendesk_predictive_agent_remediation_dry_run_total counter",
                f"opendesk_predictive_agent_remediation_dry_run_total {stats['dry_run_actions']}",
            ])
            # Binary observability metric: predictive_remediation_total{action, outcome}
            from predictive_agent.remediator import get_remediation_metrics
            rem_metrics = get_remediation_metrics()
            if rem_metrics:
                lines.append("# HELP predictive_remediation_total Total remediation decisions by action and outcome")
                lines.append("# TYPE predictive_remediation_total counter")
                for (action, outcome), count in rem_metrics.items():
                    lines.append(f"predictive_remediation_total{{action=\"{action}\",outcome=\"{outcome}\"}} {count}")
            else:
                # Always emit the metric type even if no decisions yet
                lines.append("# HELP predictive_remediation_total Total remediation decisions by action and outcome")
                lines.append("# TYPE predictive_remediation_total counter")
        decision_metrics = collect_decision_metrics(_context.remediation_manager, _context.predictor)
        if decision_metrics:
            lines.append("")
            lines.append(render_prometheus(decision_metrics))
        return "\n".join(lines) + "\n"

    def _status_dict(self):
        sm = _context.state_model
        pred = _context.predictor
        pods_tracked = len(sm.pods) if sm is not None else 0
        predictions = _all_predictions(pred)
        predictions_count = len(predictions)
        at_risk = [p for p in predictions if p.risk_score >= (pred.risk_threshold if pred else 0.5)]
        uptime = int(time.time() - _context.start_time)
        return {
            "version": config.OPERATOR_VERSION,
            "operator": config.OPERATOR_NAME,
            "status": "running",
            "pod_count": pods_tracked,
            "predictions_count": predictions_count,
            "at_risk_count": len(at_risk),
            "uptime_seconds": uptime,
        }

    def _predictions_dict(self):
        predictions = [_serialize_prediction(p) for p in _all_predictions(_context.predictor)]
        # Sort by risk score descending (highest risk first)
        predictions.sort(key=lambda p: p.get("risk_score", 0), reverse=True)
        return {
            "predictions": predictions,
            "total": len(predictions),
        }

    def _state_dict(self):
        sm = _context.state_model
        if sm is not None:
            data = sm.to_dict()
            data["total_pods_tracked"] = len(sm.pods)
            # Expose markov chain info under a clear alias
            if "markov" in data:
                data["markov_chain"] = data["markov"]
            return data
        return {
            "pods": {},
            "states": "stable",
            "markov_chain": None,
            "total_pods_tracked": 0,
        }

    def _cache_dict(self):
        cache = _context.cache
        if cache is None:
            return {"cache": {}, "total": 0}
        if isinstance(cache, dict):
            return {"cache": cache, "total": len(cache)}
        try:
            items = dict(cache)
            return {"cache": items, "total": len(items)}
        except Exception:
            return {"cache": {}, "total": 0}

    def _reanalyze(self):
        """Trigger a reconcile cycle via callback, returning (data, status)."""
        cb = _context.reconcile_callback
        if cb is not None:
            try:
                result = cb()
                return {"status": "ok", "reanalyze": "triggered", "result": result}, 200
            except Exception as exc:  # noqa: BLE001 - surface error to caller
                return {"status": "error", "error": str(exc)}, 503
        return {"status": "ok", "reanalyze": "triggered"}, 200

    def _remediate_get(self):
        """GET /remediate — return remediation config, stats, and audit trail."""
        rem = _context.remediation_manager
        if rem is None:
            return {"status": "disabled", "message": "Remediation not initialized"}
        stats = rem.get_stats()
        audit_trail = rem.get_audit_trail(limit=50)
        return {
            "status": "enabled" if not rem.dry_run else "dry_run",
            "dry_run": rem.dry_run,
            "risk_threshold": rem.risk_threshold,
            "registered_actions": stats["registered_actions"],
            "stats": stats,
            "audit_trail": audit_trail,
            "safety_policy": {
                "max_per_minute": rem.safety_policy.max_per_minute,
                "max_per_hour": rem.safety_policy.max_per_hour,
                "cooldown_seconds": rem.safety_policy.cooldown_seconds,
                "protected_namespaces": list(rem.safety_policy.protected_namespaces),
            },
        }

    def _remediation_log_get(self):
        """GET /remediation-log — return structured audit trail of all remediation decisions.
        
        Returns all remediation decisions with binary observability outcomes:
        - acted: action was executed successfully
        - skipped: action was not executed (below threshold, protected namespace, etc.)
        - failed: action execution failed with an error
        
        This implements the binary observability requirement from P2.
        """
        rem = _context.remediation_manager
        if rem is None:
            return {
                "status": "disabled",
                "message": "Remediation not initialized",
                "remediation_log": [],
                "total": 0,
            }
        remediation_log = rem.get_remediation_log(limit=100)
        return {
            "status": "enabled" if not rem.dry_run else "dry_run",
            "dry_run": rem.dry_run,
            "total": len(remediation_log),
            "remediation_log": remediation_log,
        }

    def _remediation_state_get(self):
        """GET /remediation/state — in-flight verification/hold/canary state.

        Surfaces the closed-loop remediation lifecycle to operators: which
        executions await verification, which are held (and for how long),
        which canary legs await widening, and pending outcome alerts.
        """
        rem = _context.remediation_manager
        if rem is None or not hasattr(rem, "get_remediation_state"):
            return {
                "status": "disabled",
                "message": "Remediation not initialized",
                "entries": [],
                "canary": {},
            }
        state = rem.get_remediation_state(limit=100)
        state["status"] = "enabled" if not rem.dry_run else "dry_run"
        return state

    def _remediate_post(self):
        """POST /remediate — manually trigger remediation for a specific pod.

        Expects JSON body: {"pod_name": "namespace/pod", "risk_score": 85.0}
        """
        rem = _context.remediation_manager
        if rem is None:
            return {"status": "error", "error": "Remediation not initialized"}, 503

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            return {"status": "error", "error": "Invalid JSON body"}, 400

        pod_name = payload.get("pod_name", "")
        risk_score = float(payload.get("risk_score", 0.0))
        if not pod_name:
            return {"status": "error", "error": "pod_name is required"}, 400

        # Look up pod state from the state model
        sm = _context.state_model
        if sm is None or pod_name not in sm.pods:
            return {"status": "error", "error": f"Pod '{pod_name}' not found in state model"}, 404

        tracker = sm.pods[pod_name]
        results = rem.evaluate(tracker, None, risk_score)
        return {
            "status": "ok",
            "pod_name": pod_name,
            "risk_score": risk_score,
            "results": [
                {
                    "action": r.action,
                    "target": r.target,
                    "success": r.success,
                    "dry_run": r.dry_run,
                    "message": r.message,
                    "timestamp": r.timestamp,
                    "command": r.command,
                }
                for r in results
            ],
        }, 200

    def _plan_get(self):
        """GET /plan?state=CRITICAL&horizon=3 — preview a remediation plan.

        Returns the POMDP planner's execution-ready action sequence for the
        given Markov state without executing anything. Refines the decision
        model from backtester observations once, if available.
        """
        rem = _context.remediation_manager
        if rem is None:
            return {"status": "disabled", "message": "Remediation not initialized"}, 200
        qs = parse_qs(urlparse(self.path).query)
        state = (qs.get("state") or [""])[0]
        try:
            horizon = int((qs.get("horizon") or ["3"])[0])
        except ValueError:
            horizon = 3
        if state not in MarkovChain.STATES:
            return {
                "status": "error",
                "error": f"invalid state '{state}'; expected one of {MarkovChain.STATES}",
            }, 400
        strategy = (qs.get("strategy") or ["planner"])[0]
        bt = _context.backtester
        if bt is not None:
            try:
                rem.refine_from_backtester(bt)
            except Exception:  # noqa: BLE001 - refinement must never break planning
                pass
        if strategy == "policy":
            policy = ValueIterationPolicy(
                rem.decision_model, failure_cost=rem.failure_cost, action_costs=rem.action_costs
            )
            plan = policy.plan(state, horizon).to_dict()
        elif strategy == "q":
            from predictive_agent.rl import QLearningPolicy

            qpolicy = QLearningPolicy(
                model=rem.decision_model, action_costs=rem.action_costs, failure_cost=rem.failure_cost
            )
            q_actions, q_pf = qpolicy.plan(state, horizon, rem.decision_model)
            steps = [
                {"step": i, "action": a, "expected_failed_prob_after": round(q_pf, 4)}
                for i, a in enumerate(q_actions)
            ] or [{"step": 0, "action": "none", "expected_failed_prob_after": round(q_pf, 4)}]
            plan = {
                "current_state": state,
                "horizon": horizon,
                "execution_ready": True,
                "steps": steps,
                "rationale": "Q-learning policy (model-free) rollout over the learned Q-table",
            }
        else:
            plan = rem.plan_remediation(state, horizon)
        return {"status": "ok", "strategy": strategy, "plan": plan}, 200

    def _world_model_get(self):
        """GET /world-model?state=CRITICAL&action=pod_restart&horizon=3

        What-if simulator: returns the per-step expected state distribution,
        P(FAILED) and its calibrated interval if ``action`` is applied for
        ``horizon`` steps from ``state``. Pure read-only reasoning aid — it never
        executes anything. Uses the live decision model when available.
        """
        qs = parse_qs(urlparse(self.path).query)
        state = (qs.get("state") or [""])[0]
        action = (qs.get("action") or ["none"])[0]
        try:
            horizon = int((qs.get("horizon") or ["3"])[0])
        except ValueError:
            horizon = 3
        if state not in MarkovChain.STATES:
            return {
                "status": "error",
                "error": f"invalid state '{state}'; expected one of {MarkovChain.STATES}",
            }, 400
        if action not in ACTION_TYPES:
            return {
                "status": "error",
                "error": f"invalid action '{action}'; expected one of {ACTION_TYPES}",
            }, 400
        rem = _context.remediation_manager
        model = rem.decision_model if rem is not None else None
        if model is None:
            model = ActionAwareTransitionModel()
        result = model.what_if(state, action, horizon)
        return {"status": "ok", **result}, 200

    def _plan_post(self):
        """POST /plan — plan (and optionally execute) a remediation sequence.

        Body: {"pod_name": "ns/pod", "risk_score": 0.9, "horizon": 3,
               "state": "CRITICAL", "execute": false}
        ``pod_name`` resolves the current state from the live state model;
        otherwise ``state`` is required. With ``execute=true`` the plan runs
        through ``RemediationManager.plan_and_execute`` (SafetyPolicy + dry_run).
        """
        rem = _context.remediation_manager
        if rem is None:
            return {"status": "error", "error": "Remediation not initialized"}, 503

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            return {"status": "error", "error": "Invalid JSON body"}, 400

        pod_name = payload.get("pod_name", "")
        risk_score = float(payload.get("risk_score", 0.0))
        try:
            horizon = int(payload.get("horizon", 3))
        except (ValueError, TypeError):
            horizon = 3
        execute = bool(payload.get("execute", False))

        sm = _context.state_model
        tracker = None
        if pod_name:
            if sm is None or pod_name not in sm.pods:
                return {"status": "error", "error": f"Pod '{pod_name}' not found in state model"}, 404
            tracker = sm.pods[pod_name]
            state = tracker.state
        else:
            state = payload.get("state", "")

        if state not in MarkovChain.STATES:
            return {
                "status": "error",
                "error": f"invalid/unknown state '{state}'; expected one of {MarkovChain.STATES}",
            }, 400

        bt = _context.backtester
        if bt is not None:
            try:
                rem.refine_from_backtester(bt)
            except Exception:  # noqa: BLE001
                pass

        plan = rem.plan_remediation(state, horizon)
        execution = None
        if execute:
            if tracker is None:
                return {"status": "error", "error": "execute=true requires pod_name"}, 400
            results = rem.plan_and_execute(
                state,
                pod_name,
                tracker,
                None,
                risk_score,
                horizon,
                namespace=getattr(tracker, "namespace", ""),
                node_name=getattr(tracker, "node_name", ""),
            )
            execution = [
                {
                    "action": r.action,
                    "target": r.target,
                    "success": r.success,
                    "dry_run": r.dry_run,
                    "message": r.message,
                    "timestamp": r.timestamp,
                    "command": r.command,
                }
                for r in results
            ]
        return {"status": "ok", "plan": plan, "execution": execution}, 200

    def _notifications_get(self):
        """GET /notifications — return recent notification history."""
        notifier = _context.notifier
        if notifier is None:
            return {"notifications": [], "total": 0, "message": "Notifier not initialized"}
        history = notifier.get_history(limit=50)
        return {"notifications": history, "total": len(history)}

    def _cost_model_get(self):
        """GET /cost-model — current decision-theoretic costs (read-only)."""
        rm = _context.remediation_manager
        if rm is None:
            return {"status": "disabled", "message": "Remediation not initialized"}, 200
        return {
            "source": getattr(rm, "cost_source", "defaults"),
            "failure_cost": rm.failure_cost,
            "action_costs": rm.action_costs or {},
        }, 200

    def _cost_model_post(self):
        """POST /cost-model — recalibrate costs from incident history (#5).

        Body: {"incidents": [{"action": str, "downtime_minutes": float, ...}],
               "slo_cost_per_minute": float, "typical_outage_minutes": float,
               "disruption_fraction": float}
        Recalculates action_costs / failure_cost from real ops data and applies
        them to the live RemediationManager.
        """
        rm = _context.remediation_manager
        if rm is None:
            return {"status": "error", "error": "Remediation not initialized"}, 503
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            return {"status": "error", "error": "Invalid JSON body"}, 400
        incidents = payload.get("incidents")
        if not isinstance(incidents, list) or not incidents:
            return {"status": "error", "error": "'incidents' list required"}, 400
        try:
            costs = rm.calibrate_from_incidents(
                incidents,
                slo_cost_per_minute=payload.get("slo_cost_per_minute"),
                typical_outage_minutes=payload.get("typical_outage_minutes"),
                disruption_fraction=payload.get("disruption_fraction"),
            )
        except Exception as e:  # noqa: BLE001 - surface calibration errors to caller
            return {"status": "error", "error": str(e)}, 400
        return {
            "status": "ok",
            "source": "calibrated",
            "failure_cost": costs["failure_cost"],
            "action_costs": costs["action_costs"],
        }, 200

    def _review_get(self):
        """GET /review — pending remediation reviews (P6 bounded autonomy)."""
        rm = _context.remediation_manager
        if rm is None:
            return {"enabled": False, "pending": [], "message": "Remediation not initialized"}, 200
        pending = rm.list_reviews()
        return {
            "enabled": True,
            "review_tier": getattr(rm, "review_tier", "none"),
            "review_ttl_s": getattr(rm, "review_ttl_s", 0),
            "pending": pending,
            "reviews_deferred": getattr(rm, "reviews_deferred", 0),
        }, 200

    def _review_post(self):
        """POST /review — approve a deferred remediation action (P6).

        Body: {"signature": "ns/target/action"} or {"id": "rev-1"}. The
        approval is valid for REMEDIATION_REVIEW_TTL_S seconds.
        """
        rm = _context.remediation_manager
        if rm is None:
            return {"status": "error", "error": "Remediation not initialized"}, 503
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw.decode("utf-8")) if raw.strip() else {}
        except (ValueError, UnicodeDecodeError):
            return {"status": "error", "error": "Invalid JSON body"}, 400
        if not isinstance(body, dict):
            return {"status": "error", "error": "JSON object body required"}, 400
        signature = str(body.get("signature", "") or "")
        review_id = str(body.get("id", "") or "")
        if not signature and not review_id:
            return {
                "status": "error",
                "error": "Provide 'signature' (ns/target/action) or 'id'",
            }, 400
        result = rm.approve_review(signature=signature, review_id=review_id)
        if result is None:
            return {"status": "error", "error": "Review id not found (or empty signature)"}, 404
        return {"status": "ok", **result, "pending": rm.list_reviews()}, 200

    def _coordinate_post(self):
        """POST /coordinate — multi-agent (Dec-POMDP) joint plan (#9).

        Body: {"pods": {<pod_key>: {"state": str, "node": str,
                                   "namespace": str, "deployment": str}},
               "horizon": int, "strategy": str}
        Each pod key may alternatively be a plain state string. Returns the
        coordinated joint plan with resolved cross-pod conflicts and per-node
        sequencing waves.
        """
        rm = _context.remediation_manager
        if rm is None:
            return {"status": "error", "error": "Remediation not initialized"}, 503
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            return {"status": "error", "error": "Invalid JSON body"}, 400
        pods = payload.get("pods")
        if not isinstance(pods, dict) or not pods:
            return {"status": "error", "error": "'pods' dict required"}, 400
        horizon = int(payload.get("horizon", config.PLANNING_HORIZON))
        strategy = payload.get("strategy", "planner")
        try:
            from predictive_agent.coordinator import MultiAgentCoordinator

            coordinator = MultiAgentCoordinator(
                rm, max_actions_per_node=config.MAX_ACTIONS_PER_NODE
            )
            result = coordinator.coordinate(pods, horizon=horizon, strategy=strategy)
        except Exception as e:  # noqa: BLE001 - surface coordination errors
            return {"status": "error", "error": str(e)}, 400
        return {"status": "ok", **result}, 200

    def _decision_metrics_get(self):
        """GET /decision-metrics — decision-layer metrics as JSON (#7)."""
        metrics = collect_decision_metrics(_context.remediation_manager, _context.predictor)
        return {name: value for (name, _type, _help, value) in metrics}, 200

    def _calibration_get(self):
        """Surface calibration monitoring + safety-guardrail status (read-only)."""
        rm = _context.remediation_manager
        monitor = getattr(rm, "calibration_monitor", None) if rm is not None else None
        degraded = bool(getattr(rm, "degraded", False)) if rm is not None else False
        if monitor is None:
            return {
                "enabled": False,
                "degraded": degraded,
                "summary": None,
            }, 200
        try:
            return {
                "enabled": True,
                "degraded": degraded,
                "summary": monitor.summary(),
            }, 200
        except Exception as e:
            return {"enabled": True, "degraded": degraded, "error": str(e)}, 500

    def _shadow_get(self):
        """Surface the shadow-mode validation harness summary (read-only)."""
        harness = getattr(main, "_shadow_harness", None)
        if harness is None:
            return {"enabled": False, "summary": None}, 200
        try:
            return {"enabled": True, "summary": harness.summary()}, 200
        except Exception as e:
            return {"enabled": True, "error": str(e)}, 500

    def _benchmark_get(self):
        """GET /benchmark — 4-strategy decision benchmark over real backtester
        scenarios (falls back to synthetic scenarios when none are recorded)."""
        rm = _context.remediation_manager
        if rm is None:
            return {"status": "disabled", "message": "Remediation manager not initialized"}, 200
        bt = _context.backtester
        scenarios = scenarios_from_backtester(bt) if bt is not None else []
        source = "backtester" if scenarios else "synthetic"
        if not scenarios:
            scenarios = [(s, 0.85) for s in MarkovChain.STATES]
        try:
            horizon = config.PLANNING_HORIZON
            report = run_benchmark(rm, scenarios, horizon=horizon)
            return {
                "source": source,
                "horizon": horizon,
                "scenarios": [s.to_dict() for s in report.scenarios],
                "summary": report.summary(),
            }, 200
        except Exception as e:
            return {"status": "error", "error": str(e)}, 500

    def _backtest_get(self):
        """GET /backtest — return prediction accuracy evaluation report."""
        bt = _context.backtester
        if bt is None:
            return {"status": "disabled", "message": "Backtester not initialized"}
        try:
            report = bt.evaluate()
            return report.to_dict()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _policy_health(self):
        # GET /policy/health — OPA engine liveness
        engine = _context.policy_engine
        if engine is None:
            return {"status": "disabled", "message": "OPA integration not enabled"}
        try:
            return {"opa_healthy": engine.check_health(), "enabled": engine._enabled}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _policy_stats(self):
        # GET /policy/stats — OPA engine statistics
        engine = _context.policy_engine
        if engine is None:
            return {"status": "disabled", "message": "OPA integration not enabled"}
        return engine.get_stats()

    def log_message(self, format, *args):
        # Suppress standard logging to keep test output clean
        return


# ─── Server wrapper ──────────────────────────────────────────────────────────
class HTTPServer(ThreadingHTTPServer):
    """Threaded wrapper for the HTTP server to allow easier shutdown in tests."""

    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 128

    def server_bind(self):
        # http.server's default server_bind() calls socket.getfqdn() on the
        # bind address — a reverse-DNS lookup that can block for tens of
        # seconds per server when the host's resolver cannot answer (e.g.
        # 0.0.0.0 on networks with slow DNS). The agent serves JSON APIs and
        # never uses server_name, so bind directly and skip the lookup.
        socketserver.TCPServer.server_bind(self)  # bind + listen, no DNS
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = port

    def shutdown_server(self):
        self.shutdown()


def start_server(metrics_port, health_port, state_model=None, predictor=None,
                 cache=None, reconcile_callback=None, history=None,
                 remediation_manager=None, notifier=None, backtester=None,
                 policy_engine=None):
    """Start the metrics/API and health HTTP servers.

    Args:
        metrics_port: Port for the metrics + API server.
        health_port: Port for the health probe server.
        state_model: Optional :class:`StateModel` serving real pod state.
        predictor: Optional :class:`Predictor` serving real predictions.
        cache: Optional LLM analysis cache (dict) for the ``/cache`` endpoint.
        reconcile_callback: Optional callable triggered by ``/reanalyze``.
        history: Optional list of analysis-history entries for ``/history``.
        remediation_manager: Optional :class:`RemediationManager` for /remediate.
        notifier: Optional :class:`NotificationManager` for /notifications.
        backtester: Optional :class:`Backtester` for /backtest.

    Returns:
        The metrics HTTPServer instance. Calling ``shutdown()`` on it stops
        both the metrics and health servers.
    """
    # Populate the shared context with real data sources
    _context.state_model = state_model
    _context.predictor = predictor
    _context.cache = cache
    _context.reconcile_callback = reconcile_callback
    _context.history = history if history is not None else []
    _context.remediation_manager = remediation_manager
    _context.notifier = notifier
    _context.backtester = backtester
    _context.policy_engine = policy_engine
    _context.start_time = time.time()

    # Metrics and API server (use HTTPServer subclass with allow_reuse_address)
    metrics_server = HTTPServer(("0.0.0.0", metrics_port), RequestHandler)
    metrics_thread = threading.Thread(target=metrics_server.serve_forever, daemon=True)
    metrics_thread.start()

    # Health server
    health_server = HTTPServer(("0.0.0.0", health_port), RequestHandler)
    health_thread = threading.Thread(target=health_server.serve_forever, daemon=True)
    health_thread.start()

    # Ensure shutting down the returned handle stops both servers and
    # releases both ports (server_close) — without close, the bound sockets
    # linger until GC and a later bind on the same port fails (EADDRINUSE).
    original_shutdown = metrics_server.shutdown

    def shutdown_both():
        health_server.shutdown()
        original_shutdown()
        health_server.server_close()
        metrics_server.server_close()

    metrics_server.shutdown = shutdown_both  # type: ignore[method-assign]

    return metrics_server
