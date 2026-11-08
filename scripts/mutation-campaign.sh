#!/bin/bash
# Nightly mutation campaigns on the safety-critical core (scoring + remediator).
# mutmut 3 has no per-run config override, so each campaign rewrites the
# [tool.mutmut] section (source_paths + test selection), runs, tallies, and
# restores pyproject. mutants/ is wiped between runs: mutmut caches by SOURCE
# hash only and would not re-run after test changes.
#
# Output: mutation-report.json + exit 1 if a campaign's kill rate falls below
# its floor. remediator floor tightened to 85 after the survivor-killer test
# battery (2026-09-17: 89.6% measured, 1527/1704, 0 untested); scoring floor
# tightened to 85 after its survivor-killer battery + a real Kalman FP10 bug
# fix (2026-09-17: 87.7% measured, 922/1051, 0 untested; remaining survivors
# are calibration tweaks, structural-zero algebra, and fallback defaults).
# pomdp floor set to 90 after its survivor-killer battery (2026-09-18:
# 94.4% measured, 726/769, 0 untested; remaining survivors are docstrings,
# dead guard branches ("none" is always an allowed fallback), init values
# overwritten by solve(), max() defaults for never-empty action lists, and
# two provably-equivalent boundary mutants (is_confident min_support==0,
# decay alpha==1.0 identity loop).
set -euo pipefail
cd "$(dirname "$0")/.."
REPORT="mutation-report.json"

swap_scope() {
    # guard against argument-order bugs: select = tests/*, sources = predictive_agent/*
    case "$1" in tests/*) ;; *) echo "swap_scope: arg1 must be the test selection" >&2; exit 1;; esac
    case "$2" in predictive_agent/*) ;; *) echo "swap_scope: arg2 must be the source paths" >&2; exit 1;; esac
    python3 - "$1" "$2" <<'PYEOF'
import re, sys
select, sources = sys.argv[1], sys.argv[2]
src = open("pyproject.toml").read()
sel_toml = "pytest_add_cli_args_test_selection = [%s]" % ", ".join(f'"{t}"' for t in select.split(","))
src_toml = 'source_paths = ["predictive_agent"]\nonly_mutate = [%s]' % ", ".join(f'"{t}"' for t in sources.split(","))
assert "[tool.mutmut]" in src, "pyproject.toml lost its [tool.mutmut] section"
src = re.sub(r"source_paths = \[[^\]]*\]", src_toml, src, count=1, flags=re.S)
src = re.sub(r"pytest_add_cli_args_test_selection = \[[^\]]*\]", sel_toml, src, count=1, flags=re.S)
open("pyproject.toml", "w").write(src)
PYEOF
}

run_campaign() {
    local name="$1" floor="$2" sources="$3" select="$4"
    echo "═══ mutation campaign: $name"
    swap_scope "$select" "$sources"
    rm -rf mutants
    set +e
    python3 -m mutmut run --max-children 4 > "mutmut-$name.log" 2>&1
    python3 -m mutmut results > "mutants-$name.txt" 2>&1
    set -e
    git checkout -- pyproject.toml
    python3 - "$name" "$floor" <<'PYEOF'
import json, os, re, sys
name, floor = sys.argv[1], float(sys.argv[2])
log = open(f"mutmut-{name}.log", encoding="utf-8", errors="replace").read()
# Final tally line, e.g. "... 3343/3343  🎉 1274 🫥 1019 ⏰ 16 🤔 0 🙁 848 🔇 0 🧙 0"
tally = re.findall(r"(\d+)/(\d+)\s+🎉\s*(\d+).*?🫥\s*(\d+).*?⏰\s*(\d+).*?🤔\s*(\d+).*?🙁\s*(\d+)", log)
if not tally:
    raise SystemExit(f"{name}: could not parse mutmut tally — campaign failed?")
_, total, killed, untested, timeouts, suspicious, survived = (int(x) for x in tally[-1])
rate = killed / total * 100 if total else 0.0
print(f"{name}: {killed}/{total} killed ({rate:.0f}%), {survived} survived, {untested} untested")
try:
    report = json.load(open("mutation-report.json"))
except Exception:
    report = {}
report[name] = {"total": total, "killed": killed, "survived": survived,
                "untested": untested, "kill_rate": round(rate, 1), "floor": floor}
json.dump(report, open("mutation-report.json", "w"), indent=2)
if rate < floor:
    raise SystemExit(f"KILL-RATE FLOOR BREACHED: {name} {rate:.1f}% < {floor}%")
PYEOF
}

# Floors: tighten from the first nightly run's numbers.
run_campaign scoring 85 \
    "predictive_agent/risk.py,predictive_agent/markov.py,predictive_agent/kalman.py,predictive_agent/predictor.py" \
    "tests/test_risk.py,tests/test_markov.py,tests/test_kalman.py,tests/test_kalman_anomaly.py,tests/test_kalman_params.py,tests/test_predictor.py,tests/test_state_hysteresis.py"

run_campaign remediator 85 \
    "predictive_agent/remediator.py,predictive_agent/actions" \
    "tests/test_remediator.py,tests/test_docker_remediation.py,tests/test_integration_remediation.py,tests/test_bounded_autonomy.py,tests/test_calibration_guardrails.py,tests/test_continual_learning.py,tests/test_crashloop_escalation.py,tests/test_mutation_harvest.py,tests/test_action_guards.py,tests/test_cost_model.py"

run_campaign pomdp 90 \
    "predictive_agent/pomdp.py" \
    "tests/test_pomdp.py"

echo "campaigns complete: $(cat "$REPORT")"
