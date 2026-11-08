# Deploying predictive-agent as a Nix container image

The agent is distributed as a reproducible Nix-built OCI image
(`predictive-agent:v4.0.8-nix`). The image runs the same `predictive_agent.main`
reconcile loop used in Kubernetes, but can also run **Docker-native** (monitor
and remediate containers on the host Docker daemon via the socket).

## What's in the image
- Python 3 (+ `psutil`) from nixpkgs
- `curl`, `bash`, `coreutils`, `gnugrep`, `gnused`, `procps`, `cacert`
- `kubectl` (Kubernetes mode) and `docker` client (Docker-native mode)
- The full `predictive_agent` package, including the POMDP decision layer
- Entrypoint → `python3 -m predictive_agent.main`
- Health endpoints `/healthz` (liveness) and `/ready` (readiness) on port 8081

## Prerequisites
- **Nix** with flakes: https://nixos.org/download.html
- A **docker daemon**. On macOS use Colima (Lima + QEMU):
  ```bash
  brew install colima
  colima start                 # creates ~/.colima/default/docker.sock
  docker context use colima    # optional
  ```
- The image is built for **`x86_64-linux`**. Build on a Linux host or a remote
  builder. On Apple-Silicon macOS, configure a Linux builder (e.g. via
  `nix build` with `--system x86_64-linux` and a registered binfmt/builder), or
  build inside a Linux VM.

## Build + load
```bash
./scripts/build-nix-image.sh
# equivalent to:
#   nix build .#predictive-agent -o result-image
#   docker load -i result-image
docker images | grep predictive-agent   # predictive-agent  v4.0.8-nix
```

## Run (Docker-native, recommended for local spin-up)
```bash
# macOS / colima:
# Mount /var/run/docker.sock — this is resolved INSIDE the Colima VM where the
# daemon listens. Do NOT mount the Mac-side proxy path
# ~/.colima/default/docker.sock (it cannot be bind-mounted into a container).
docker compose up

# Linux (default socket):
docker compose up
```
The compose file mounts the host Docker socket read-only and runs the agent in
**dry-run remediation** (safe). Metrics are on `:8080`, health on `:8081`.

Or run directly:
```bash
docker run --rm -p 8080:8080 -p 8081:8081 \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  -e OPERATOR_RUNTIME=docker -e LLM_BACKEND=none \
  -e OPERATOR_WATCH_NAMESPACES=predagent \
  predictive-agent:v4.0.8-nix
```

### Watching namespaces in Docker mode
There is no "namespace" in plain Docker. The agent derives one from the
container's `com.docker.compose.project` label, or the prefix before `_` in the
container name (e.g. `predagent-demo-nginx` → namespace `predagent`). Set
`OPERATOR_WATCH_NAMESPACES` to the compose projects you care about.

## Run (Kubernetes)
The kustomized manifests under `k8s/` deploy the agent into the
`openedusuite` namespace (set in `k8s/kustomization.yaml`). Update the image
tag in `k8s/deployment.yaml` to `predictive-agent:v4.0.8-nix` (or your registry
path) and apply with kustomize:
```bash
kubectl apply -k k8s/
```
This installs RBAC, PVC, ConfigMap, Deployment, Service and the
ServiceMonitor. (The Deployment also carries the health/metrics endpoints
used by the probes.)

## Reproducible chaos / remediation demo

To watch the decision layer actually remediate a container, spin up a
crash-looping container via the `chaos` compose profile. The agent (watching
the `predagent` namespace) will flag it at-risk and restart it:

```bash
# Build + load the image first (see above), then:
REMEDIATION_DRY_RUN=false docker compose --profile chaos up
# watch the logs:
docker logs -f predagent_chaos
# expected (default CRASHLOOP_POLICY=escalate):
#   Pod docker/predagent_chaos at risk: score=0.99 state=CRITICAL
#   Remediation: none on docker/predagent_chaos — planner: no beneficial action ...
#   Remediation exhausted for docker/predagent_chaos: 3 consecutive at-risk ...
#   (the agent escalates instead of restart-storming a deterministic crash-loop)
#
# For a *continuous* restart demo (churn over silence), set the policy:
#   CRASHLOOP_POLICY=persist_restart REMEDIATION_DRY_RUN=false \
#     docker compose --profile chaos up
```

The agent defaults to `REMEDIATION_DRY_RUN=true` (safe). Set it to `false`
(on the `predictive-agent` service) to let it actually execute actions. After
a restart it engages a cooldown (~5 min) so it won't restart-storm. Stop the
demo with `docker compose --profile chaos down`.

## Key environment variables
| Variable | Default (image) | Notes |
|---|---|---|
| `OPERATOR_RUNTIME` | `kubectl` | `docker` for Docker-native mode (or leave empty + mount socket) |
| `DOCKER_SOCKET` | `/var/run/docker.sock` | Socket mounted into the container |
| `OPERATOR_WATCH_NAMESPACES` | `opendesk,opendesk-edu,default,llm` | Comma list; in Docker mode these are compose-project names |
| `COLLECTOR_MODE` | `kubectl` | Mirrors `OPERATOR_RUNTIME`; auto-detected if unset |
| `LLM_BACKEND` | `ollama` | `none` for pure predictive/planner mode (no LLM calls) |
| `PLANNING_ENABLED` | `true` | POMDP planner on by default |
| `CRASHLOOP_POLICY` | `escalate` | `escalate` pages a human after N futile at-risk cycles; `persist_restart` keeps restarting on every cooldown |
| `CRASHLOOP_ESCALATE_AFTER` | `3` | consecutive at-risk `none` cycles before escalation |
| `REMEDIATION_ENABLED` | `true` | Enables remediation path |
| `REMEDIATION_DRY_RUN` | `true` | **Safe default** — set `false` to actually execute actions |
| `REMEDIATION_RISK_THRESHOLD` | `0.7` | Min risk to consider remediation |
| `MAX_ACTIONS_PER_RECONCILE` | `10` | Circuit breaker per cycle |
| `RECONCILE_INTERVAL` | `60` | Seconds between cycles |
| `OPERATOR_METRICS_BIND_ADDRESS` | `0.0.0.0:8080` | Prometheus metrics + JSON decision metrics |
| `OPERATOR_HEALTH_PROBE_BIND_ADDRESS` | `0.0.0.0:8081` | `/healthz` + `/ready` |

The full set lives in `predictive_agent/config.py`.

## Decision layer in the container
The POMDP decision layer (predict → plan → remediate → learn) is enabled by
default. Calibration, shadow-mode, RL and Dec-POMDP coordination are wired in
but conservative by default (shadow off, dry-run on, RL only when enabled via
the API). See `GET /plan?state=CRITICAL` and `GET /decision-metrics` once
running.
