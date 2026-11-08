# Alternative build path (no Nix required). The canonical, reproducible image
# is the Nix one in nix/predictive-agent.nix — see docs/DEPLOY_NIX.md.
# This Dockerfile mirrors it for quick local testing on any docker host.
FROM python:3.12-slim

# kubectl is REQUIRED for kubernetes-mode collection (pods, metrics, logs).
# docker-cli is EQUALLY required for docker-mode collection (the edge-node agent
# watches Docker containers via /var/run/docker.sock). Shipping the image
# without a runtime's CLI silently blinds that runtime (0 pods tracked with
# no warnings) — the incident class this pin exists for. Pinned + sha-verified.
# Both binaries ship in the image; the runtime is chosen on the host.
ARG KUBECTL_VERSION=v1.36.3
ARG KUBECTL_SHA256=ebbd080e7c2e275093b55915722043257eb24004363e20acb3c4d71919f88336

RUN KUBECTL_VERSION=${KUBECTL_VERSION} KUBECTL_SHA256=${KUBECTL_SHA256} python3 - <<'PY'
import hashlib, os, urllib.request
ver = os.environ["KUBECTL_VERSION"]
expected = os.environ["KUBECTL_SHA256"]
url = f"https://dl.k8s.io/release/{ver}/bin/linux/amd64/kubectl"
dest = "/usr/local/bin/kubectl"
req = urllib.request.Request(url, headers={"User-Agent": "predictive-agent-build"})
with urllib.request.urlopen(req, timeout=180) as r:
    data = r.read()
actual = hashlib.sha256(data).hexdigest()
if actual != expected:
    raise SystemExit(f"kubectl checksum mismatch: expected {expected} got {actual}")
with open(dest, "wb") as fh:
    fh.write(data)
os.chmod(dest, 0o755)
print(f"installed kubectl {ver} ({len(data)} bytes, sha256 ok)")
PY

# docker-cli static binary from download.docker.com (deb mirrors are blocked on
# the build host). Only the client is extracted; dockerd/runc stay in the tgz.
ARG DOCKER_VERSION=29.9.1
ARG DOCKER_SHA256=c158ecc55adb3afb01bbd3539ed3b997d20c26dbdc418080506cc1bf5243ff57

RUN DOCKER_VERSION=${DOCKER_VERSION} DOCKER_SHA256=${DOCKER_SHA256} python3 - <<'PY'
import hashlib, io, os, tarfile, urllib.request
ver = os.environ["DOCKER_VERSION"]
expected = os.environ["DOCKER_SHA256"]
url = f"https://download.docker.com/linux/static/stable/x86_64/docker-{ver}.tgz"
req = urllib.request.Request(url, headers={"User-Agent": "predictive-agent-build"})
with urllib.request.urlopen(req, timeout=180) as r:
    data = r.read()
actual = hashlib.sha256(data).hexdigest()
if actual != expected:
    raise SystemExit(f"docker checksum mismatch: expected {expected} got {actual}")
with tarfile.open(fileobj=io.BytesIO(data)) as tf:
    member = next(m for m in tf.getmembers() if m.name == "docker/docker")
    f = tf.extractfile(member)
    if f is None:
        raise SystemExit("docker/docker not found in archive")
    with open("/usr/local/bin/docker", "wb") as out:
        out.write(f.read())
os.chmod("/usr/local/bin/docker", 0o755)
print(f"installed docker CLI {ver} ({len(data)} bytes, sha256 ok)")
PY

RUN pip install --no-cache-dir psutil

WORKDIR /opt/predictive-agent

COPY predictive_agent /opt/predictive-agent/predictive_agent
COPY nix/predictive-agent-files/entrypoint.sh /opt/predictive-agent/entrypoint.sh
COPY nix/predictive-agent-files/healthcheck.sh /opt/predictive-agent/healthcheck.sh
RUN chmod +x /opt/predictive-agent/entrypoint.sh /opt/predictive-agent/healthcheck.sh

ENV PYTHONPATH=/opt/predictive-agent \
    OPERATOR_VERSION=4.0.8 \
    OPERATOR_RUNTIME=kubectl \
    COLLECTOR_MODE=kubectl \
    DOCKER_SOCKET=/var/run/docker.sock \
    OPERATOR_WATCH_NAMESPACES=opendesk,opendesk-edu,default,llm \
    LLM_BACKEND=ollama \
    RECONCILE_INTERVAL=60 \
    OPERATOR_METRICS_BIND_ADDRESS=0.0.0.0:8080 \
    OPERATOR_HEALTH_PROBE_BIND_ADDRESS=0.0.0.0:8081 \
    PLANNING_ENABLED=true \
    PLANNING_HORIZON=3 \
    REMEDIATION_ENABLED=true \
    REMEDIATION_DRY_RUN=true \
    REMEDIATION_RISK_THRESHOLD=0.7 \
    PREDICTION_BASE_RISK=0.15

EXPOSE 8080 8081

ENTRYPOINT ["/opt/predictive-agent/entrypoint.sh"]
