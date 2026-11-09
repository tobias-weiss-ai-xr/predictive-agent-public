# SPDX-License-Identifier: Apache-2.0
# Predictive Agent v4.0.8.1 — Predictive Kubernetes Health Monitor with Performance Optimizations
# Image: predictive-agent:v4.0.8.1-nix
# Built with: Nix dockerTools.buildLayeredImage using nixpkgs
# Contains: python3, curl, bash, kubectl, coreutils, gnugrep, gnused, procps, cacert
# Entrypoint: runs predictive_agent.main (the reconcile loop)
# Healthcheck: curl /healthz and /ready on the health-probe port (default 8081)
#
# Based on the v3.1 nix build at:
#   opendesk-nix/nix/images/dev-agent.nix
# adapted for the v4.0 stdlib-only Python package layout (predictive_agent/).

{ pkgs ? import <nixpkgs> { system = "x86_64-linux"; } }:

let
  entrypointSh = pkgs.writeText "entrypoint.sh" (builtins.readFile ./predictive-agent-files/entrypoint.sh);
  healthcheckSh = pkgs.writeText "healthcheck.sh" (builtins.readFile ./predictive-agent-files/healthcheck.sh);

  # The v4.0.8.1 predictive_agent package (stdlib-only Python) with optimizations.
  # Copy all modules: main package, actions, optimizer, cache, llm_batch, benchmark tests.
  # Also copy the tests/benchmark directory for potential in-container testing.
  predictiveAgentPackage = pkgs.runCommand "predictive-agent-package" {} ''
    mkdir -p $out/opt/predictive-agent/predictive_agent/actions
    mkdir -p $out/opt/predictive-agent/predictive_agent/tests/benchmark
    
    # Main package files
    cp ${../predictive_agent}/*.py $out/opt/predictive-agent/predictive_agent/
    cp ${../predictive_agent/actions}/*.py $out/opt/predictive-agent/predictive_agent/actions/
    
    # Optimization modules (NEW in v4.0.1)
    cp ${../predictive_agent/l1_cache.py} $out/opt/predictive-agent/predictive_agent/
    cp ${../predictive_agent/llm_batch.py} $out/opt/predictive-agent/predictive_agent/
    cp ${../predictive_agent/optimize.py} $out/opt/predictive-agent/predictive_agent/
    cp ${../predictive_agent/demo_optimizations.py} $out/opt/predictive-agent/predictive_agent/
    
    # OPA Integration modules (NEW in v4.0.8.1)
    cp ${../predictive_agent/opa_client.py} $out/opt/predictive-agent/predictive_agent/
    cp ${../predictive_agent/opa_cache.py} $out/opt/predictive-agent/predictive_agent/
    cp ${../predictive_agent/opa_integration.py} $out/opt/predictive-agent/predictive_agent/
    
    # Benchmark test modules
    cp ${../tests/benchmark}/* $out/opt/predictive-agent/predictive_agent/tests/benchmark/ 2>/dev/null || true
    
    # Entrypoint and healthcheck
    cp ${entrypointSh} $out/opt/predictive-agent/entrypoint.sh
    chmod +x $out/opt/predictive-agent/entrypoint.sh
    cp ${healthcheckSh} $out/opt/predictive-agent/healthcheck.sh
    chmod +x $out/opt/predictive-agent/healthcheck.sh
  '';

  # Real /etc/passwd and /etc/group. We avoid dockerTools.fakeNss because
  # buildLayeredImage does not include fakeNss's symlink targets in the image
  # layers, which breaks any NSS lookup (e.g. pwd.getpwnam). The operator runs
  # as root (0:0); these entries satisfy tools that read /etc/passwd directly.
  etcFiles = pkgs.runCommand "predictive-agent-etc" {} ''
    mkdir -p $out/etc
    echo 'root:x:0:0:root:/root:/bin/bash' > $out/etc/passwd
    echo 'opendesk:x:1000:1000:opendesk:/home/opendesk:/bin/bash' >> $out/etc/passwd
    echo 'nobody:x:65534:65534:nobody:/:/sbin/nologin' >> $out/etc/passwd
    echo 'root:x:0:' > $out/etc/group
    echo 'opendesk:x:1000:' >> $out/etc/group
    echo 'nobody:x:65534:' >> $out/etc/group
  '';

in
pkgs.dockerTools.buildLayeredImage {
  name = "predictive-agent";
  tag = "v4.0.8.1-nix";

  contents = with pkgs; [
    python3
    curl
    bash
    kubectl
    docker-client   # needed for Docker-native discovery (docker ps / docker inspect)
    coreutils
    gnugrep
    gnused
    procps
    cacert
    predictiveAgentPackage
    etcFiles
    # Python packages for optimizations
    pkgs.python3Packages.psutil  # For CPU monitoring
  ];

  config = {
    User = "0:0";
    WorkingDir = "/opt/predictive-agent";
    Entrypoint = [
      "${pkgs.bash}/bin/bash"
      "/opt/predictive-agent/entrypoint.sh"
    ];
    Cmd = [];
    Env = [
      "OPERATOR_VERSION=4.0.8.1"
      "OPERATOR_NAME=opendesk-predictive-agent"
      "OPERATOR_NAMESPACE=opendesk-predictive-agent"
      "OPERATOR_WATCH_NAMESPACES=opendesk,opendesk-edu,default,llm"
      "COLLECTOR_MODE=kubectl"
      "LLM_BACKEND=ollama"
      "OLLAMA_URL=http://ollama.llm.svc.cluster.local:11434"
      "OLLAMA_MODEL=qwen3-30b-a3b:latest"
      "RECONCILE_INTERVAL=60"
      "OPERATOR_METRICS_BIND_ADDRESS=0.0.0.0:8080"
      "OPERATOR_HEALTH_PROBE_BIND_ADDRESS=0.0.0.0:8081"
      "REMEDIATION_ENABLED=true"
      "REMEDIATION_DRY_RUN=true"
      "REMEDIATION_MAX_PER_MIN=5"
      "REMEDIATION_MAX_PER_HOUR=50"
      "REMEDIATION_COOLDOWN_S=300"
      "REMEDIATION_RISK_THRESHOLD=0.7"
      "REMEDIATION_PROTECTED_NS=kube-system,opendesk-predictive-agent"
      "ALERT_EMAIL_TO=alerts@example.com"
      "SMTP_HOST=smtp.example.com"
      "SMTP_PORT=587"
      "SMTP_USE_TLS=true"
      "WEBHOOK_URL="
      "WEBHOOK_TIMEOUT=10"
      "PYTHONPATH=/opt/predictive-agent:${pkgs.python3Packages.psutil}/lib/python3.14/site-packages"
      "PREDICTION_BASE_RISK=0.15"
      "PATH=${pkgs.python3}/bin:${pkgs.curl}/bin:${pkgs.bash}/bin:${pkgs.docker-client}/bin:${pkgs.coreutils}/bin:${pkgs.gnugrep}/bin:${pkgs.gnused}/bin:${pkgs.procps}/bin:${pkgs.kubectl}/bin"
      "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
      "HOME=/home/opendesk"
    ];
    ExposedPorts = {
      "8080/tcp" = {};
      "8081/tcp" = {};
    };
  };

  maxLayers = 50;
}
