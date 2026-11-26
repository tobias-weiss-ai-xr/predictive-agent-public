# SPDX-License-Identifier: Apache-2.0
# Predictive Agent v4.0.8.1 — Predictive Kubernetes Health Monitor with Performance Optimizations
# Image: predictive-agent:v4.0.8.1-nix
# Built with: Nix dockerTools.buildLayeredImage using nixpkgs
# Contains: python3, curl, bash, kubectl, coreutils, gnugrep, gnused, procps, cacert
# Entrypoint: runs predictive_agent.main (the reconcile loop)
# Healthcheck: curl /healthz and /ready on the health-probe port (default 8081)
#
# Based on the v3.1 nix build at:
#   openedusuite-nix/nix/images/dev-agent.nix
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

    # Every module in the package root (main package, optimizer, cache,
    # llm_batch, OPA integration, ...). Do NOT add individual per-file
    # `cp <path-to-one-file> $out/.../` lines: when a Nix path literal names a
    # single FILE, the store path is named `<hash>-foo.py`, and cp then keeps
    # that whole basename — producing unimportable `1nxrd...-opa_client.py`
    # duplicates in the image (7 of them until v4.1.6). The glob below already
    # copies everything, including new modules.
    cp ${../predictive_agent}/*.py $out/opt/predictive-agent/predictive_agent/
    cp ${../predictive_agent/actions}/*.py $out/opt/predictive-agent/predictive_agent/actions/
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
    echo 'predictive-agent:x:1000:1000:predictive-agent:/home/predictive-agent:/bin/bash' >> $out/etc/passwd
    echo 'nobody:x:65534:65534:nobody:/:/sbin/nologin' >> $out/etc/passwd
    echo 'root:x:0:' > $out/etc/group
    echo 'predictive-agent:x:1000:' >> $out/etc/group
    echo 'nobody:x:65534:' >> $out/etc/group
  '';

in
pkgs.dockerTools.buildLayeredImage {
  name = "predictive-agent";
  tag = "v4.1.15-nix";

  contents = with pkgs; [
    python3  # AlertWatch uses stdlib urllib only (no requests in the image)
    curl
    bash
    kubectl
    git            # FixImage gitops mode (manifest commit + push)
    openssh        # git+ssh push transport for FixImage
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
      "OPERATOR_VERSION=4.1.15"
      "OPERATOR_NAME=predictive-agent"
      "OPERATOR_NAMESPACE=predictive-agent"
      "OPERATOR_WATCH_NAMESPACES=openedusuite,default,llm"
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
      "REMEDIATION_PROTECTED_NS=kube-system,predictive-agent"
      "ALERT_EMAIL_TO=tobias.weiss@uni-marburg.de"
      "SMTP_HOST=smtp.uni-marburg.de"
      "SMTP_PORT=587"
      "SMTP_USE_TLS=true"
      "WEBHOOK_URL="
      "WEBHOOK_TIMEOUT=10"
      "PYTHONPATH=/opt/predictive-agent:${pkgs.python3Packages.psutil}/lib/python3.14/site-packages"
      "PREDICTION_BASE_RISK=0.15"
      "PATH=${pkgs.python3}/bin:${pkgs.curl}/bin:${pkgs.bash}/bin:${pkgs.docker-client}/bin:${pkgs.coreutils}/bin:${pkgs.gnugrep}/bin:${pkgs.gnused}/bin:${pkgs.procps}/bin:${pkgs.kubectl}/bin"
      "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
      "HOME=/home/predictive-agent"
    ];
    ExposedPorts = {
      "8080/tcp" = {};
      "8081/tcp" = {};
    };
  };

  maxLayers = 50;
}
