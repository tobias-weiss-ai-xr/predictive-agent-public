"""Backtesting harness for prediction accuracy evaluation.

Records predictions and evaluates them against actual outcomes (pod restarts,
OOM kills, state transitions). Computes Brier score, calibration, and TTF
accuracy.

Usage:
    from predictive_agent.backtester import Backtester

    bt = Backtester()
    bt.record_prediction(pod_key, risk_score, ttf_minutes, markov_state)
    bt.record_outcome(pod_key, actual_failure, actual_ttf_minutes)
    report = bt.evaluate()
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from predictive_agent import config

logger = logging.getLogger(__name__)

# Backtest history lives alongside the other persisted state. Prefer the
# shared HISTORY_FILE knob (same one STATE_MODEL_FILE / PREDICTIONS_FILE use)
# so a deployment can point all persistence at one writable dir; keep the
# dedicated BACKTEST_HISTORY_FILE override for backwards compatibility.
DEFAULT_HISTORY_FILE = os.environ.get(
    "BACKTEST_HISTORY_FILE",
    os.path.join(
        os.path.dirname(config.HISTORY_FILE),
        "backtest-history.jsonl",
    ),
)

# Maximum number of prediction/outcome records to keep in memory.
# Prevents unbounded memory growth over long uptimes.
MAX_HISTORY_RECORDS = int(os.environ.get("BACKTEST_MAX_HISTORY", "10000"))

# Cap on the JSONL history file size (env BACKTEST_HISTORY_MAX_MB, default
# 256 MiB). When appending would exceed the cap, the file is rotated to
# <history>.1 (old backup replaced), so disk usage AND the all-in-memory
# startup load stay bounded — the operator was OOM-killed in a restart loop
# once backtest-history.jsonl reached 1.8 GB (whole file loaded at start).
MAX_HISTORY_FILE_BYTES = int(os.environ.get("BACKTEST_HISTORY_MAX_MB", "256")) * 1024 * 1024


@dataclass
class PredictionRecord:
    """A single prediction snapshot at a point in time."""
    pod_key: str
    timestamp: float
    risk_score: float
    ttf_minutes: Optional[int]
    confidence: float
    markov_state: str
    memory_pct: float
    cpu_pct: float
    memory_trend: float
    cpu_trend: float


@dataclass
class OutcomeRecord:
    """Actual outcome for a pod at a point in time."""
    pod_key: str
    timestamp: float
    actual_failure: bool  # Did pod fail (OOM, CrashLoop, restart)?
    actual_ttf_minutes: Optional[float]  # Actual time to failure (if any)
    restart_count: int
    state: str


@dataclass
class BacktestReport:
    """Evaluation report comparing predictions to outcomes."""
    total_predictions: int = 0
    total_outcomes: int = 0
    matched_pairs: int = 0
    brier_score: float = 0.0
    false_positives: int = 0
    false_negatives: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    ttf_errors: list = field(default_factory=list)
    calibration: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Backtester:
    """Backtesting harness for prediction accuracy.

    Records prediction snapshots and actual outcomes, then evaluates
    prediction quality using Brier score, confusion matrix, and TTF error.
    """

    def __init__(self, history_file: str = DEFAULT_HISTORY_FILE):
        self.history_file = history_file
        self._predictions: list[PredictionRecord] = []
        self._outcomes: list[OutcomeRecord] = []
        self._load_history()

    def record_prediction(
        self,
        pod_key: str,
        risk_score: float,
        ttf_minutes: Optional[int],
        confidence: float,
        markov_state: str,
        memory_pct: float = 0.0,
        cpu_pct: float = 0.0,
        memory_trend: float = 0.0,
        cpu_trend: float = 0.0,
    ) -> None:
        """Record a prediction snapshot for a pod."""
        record = PredictionRecord(
            pod_key=pod_key,
            timestamp=time.time(),
            risk_score=risk_score,
            ttf_minutes=ttf_minutes,
            confidence=confidence,
            markov_state=markov_state,
            memory_pct=memory_pct,
            cpu_pct=cpu_pct,
            memory_trend=memory_trend,
            cpu_trend=cpu_trend,
        )
        self._predictions.append(record)
        # Cap in-memory history to prevent unbounded growth
        if len(self._predictions) > MAX_HISTORY_RECORDS:
            self._predictions = self._predictions[-MAX_HISTORY_RECORDS:]
        self._append_to_file("prediction", record)

    def record_outcome(
        self,
        pod_key: str,
        actual_failure: bool,
        restart_count: int = 0,
        state: str = "HEALTHY",
        actual_ttf_minutes: Optional[float] = None,
    ) -> None:
        """Record the actual outcome for a pod."""
        record = OutcomeRecord(
            pod_key=pod_key,
            timestamp=time.time(),
            actual_failure=actual_failure,
            actual_ttf_minutes=actual_ttf_minutes,
            restart_count=restart_count,
            state=state,
        )
        self._outcomes.append(record)
        # Cap in-memory history to prevent unbounded growth
        if len(self._outcomes) > MAX_HISTORY_RECORDS:
            self._outcomes = self._outcomes[-MAX_HISTORY_RECORDS:]
        self._append_to_file("outcome", record)

    def evaluate(self, time_window: float = 3600) -> BacktestReport:
        """Evaluate prediction accuracy.

        Matches each prediction to the nearest outcome within time_window
        seconds and computes metrics.

        Args:
            time_window: Maximum seconds between prediction and outcome to match.

        Returns:
            BacktestReport with Brier score, confusion matrix, and TTF errors.
        """
        report = BacktestReport(
            total_predictions=len(self._predictions),
            total_outcomes=len(self._outcomes),
        )

        if not self._predictions or not self._outcomes:
            return report

        # Match predictions to outcomes by pod_key and time proximity
        matched = []
        for pred in self._predictions:
            best_outcome = None
            best_dt = float("inf")
            for outcome in self._outcomes:
                if outcome.pod_key != pred.pod_key:
                    continue
                dt = abs(outcome.timestamp - pred.timestamp)
                if dt < best_dt and dt <= time_window:
                    best_dt = dt
                    best_outcome = outcome
            if best_outcome is not None:
                matched.append((pred, best_outcome))

        report.matched_pairs = len(matched)
        if not matched:
            return report

        # Brier score: mean((predicted_prob - actual)^2)
        brier_sum = 0.0
        for pred, outcome in matched:
            actual = 1.0 if outcome.actual_failure else 0.0
            brier_sum += (pred.risk_score - actual) ** 2

            # Confusion matrix (threshold = 0.5)
            predicted_positive = pred.risk_score >= 0.5
            actual_positive = outcome.actual_failure

            if predicted_positive and actual_positive:
                report.true_positives += 1
            elif predicted_positive and not actual_positive:
                report.false_positives += 1
            elif not predicted_positive and actual_positive:
                report.false_negatives += 1
            else:
                report.true_negatives += 1

            # TTF error
            if pred.ttf_minutes is not None and outcome.actual_ttf_minutes is not None:
                ttf_error = abs(pred.ttf_minutes - outcome.actual_ttf_minutes)
                report.ttf_errors.append({
                    "pod_key": pred.pod_key,
                    "predicted_ttf": pred.ttf_minutes,
                    "actual_ttf": outcome.actual_ttf_minutes,
                    "error_minutes": ttf_error,
                })

        report.brier_score = brier_sum / len(matched)

        # Calibration: group predictions into bins and check reliability
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        calibration = {}
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            in_bin = [
                (p, o) for p, o in matched
                if lo <= p.risk_score < hi
            ]
            if in_bin:
                avg_pred = sum(p.risk_score for p, _ in in_bin) / len(in_bin)
                actual_rate = sum(1 for _, o in in_bin if o.actual_failure) / len(in_bin)
                calibration[f"{lo:.1f}-{hi:.1f}"] = {
                    "count": len(in_bin),
                    "avg_predicted": round(avg_pred, 4),
                    "actual_rate": round(actual_rate, 4),
                }
        report.calibration = calibration

        return report

    def get_prediction_count(self) -> int:
        """Return number of recorded predictions."""
        return len(self._predictions)

    def get_outcome_count(self) -> int:
        """Return number of recorded outcomes."""
        return len(self._outcomes)

    def clear_history(self) -> None:
        """Clear in-memory history (does not delete file)."""
        self._predictions.clear()
        self._outcomes.clear()

    def _append_to_file(self, record_type: str, record) -> None:
        """Append a record to the JSONL history file, rotating to <history>.1
        once the file would exceed MAX_HISTORY_FILE_BYTES (single choke point:
        predictions and outcomes both write here)."""
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            data = asdict(record) if hasattr(record, "__dataclass_fields__") else record.__dict__
            data["_type"] = record_type
            line = json.dumps(data) + "\n"
            if os.path.exists(self.history_file) and \
                    os.path.getsize(self.history_file) + len(line.encode()) > MAX_HISTORY_FILE_BYTES:
                backup = self.history_file + ".1"
                if os.path.exists(backup):
                    os.remove(backup)  # keep only one bounded backup
                os.replace(self.history_file, backup)
            with open(self.history_file, "a") as f:
                f.write(line)
        except OSError as e:
            logger.debug("Could not write backtest history: %s", e)

    def _load_history(self) -> None:
        """Load existing history from JSONL file."""
        if not os.path.exists(self.history_file):
            return
        try:
            with open(self.history_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rtype = data.pop("_type", None)
                    if rtype == "prediction":
                        self._predictions.append(PredictionRecord(
                            pod_key=data.get("pod_key", ""),
                            timestamp=data.get("timestamp", 0.0),
                            risk_score=data.get("risk_score", 0.0),
                            ttf_minutes=data.get("ttf_minutes"),
                            confidence=data.get("confidence", 0.0),
                            markov_state=data.get("markov_state", "HEALTHY"),
                            memory_pct=data.get("memory_pct", 0.0),
                            cpu_pct=data.get("cpu_pct", 0.0),
                            memory_trend=data.get("memory_trend", 0.0),
                            cpu_trend=data.get("cpu_trend", 0.0),
                        ))
                    elif rtype == "outcome":
                        self._outcomes.append(OutcomeRecord(
                            pod_key=data.get("pod_key", ""),
                            timestamp=data.get("timestamp", 0.0),
                            actual_failure=data.get("actual_failure", False),
                            actual_ttf_minutes=data.get("actual_ttf_minutes"),
                            restart_count=data.get("restart_count", 0),
                            state=data.get("state", "HEALTHY"),
                        ))
        except OSError as e:
            logger.debug("Could not load backtest history: %s", e)
