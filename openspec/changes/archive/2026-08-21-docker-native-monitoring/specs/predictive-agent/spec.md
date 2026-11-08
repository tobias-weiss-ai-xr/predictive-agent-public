## MODIFIED Requirements

### Requirement: Metrics collection
The operator SHALL collect workload metrics from the runtime. In Kubernetes mode it uses `kubectl` (`top`/`get`/`logs`) scoped to `OPERATOR_WATCH_NAMESPACES`. In Docker mode it SHALL discover containers via the read-only Docker socket using label / compose-project / name selectors (`OPERATOR_WATCH_LABELS`, `OPERATOR_WATCH_COMPOSE_PROJECTS`, `OPERATOR_WATCH_NAMES`) and track each discovered container's CPU, memory, restart, log, and status signals.

#### Scenario: Kubernetes mode
- **WHEN** deployed in a cluster with kubeconfig access
- **THEN** it uses `kubectl` to collect pod/node metrics and logs for the watched namespaces

#### Scenario: Docker mode
- **WHEN** deployed as a Docker container with the socket mounted read-only
- **THEN** it discovers containers matching the configured label / compose-project / name selectors and tracks their metrics and status signals

## ADDED Requirements

### Requirement: Docker workload discovery
The operator SHALL discover Docker containers to monitor without requiring Kubernetes namespaces. Discovery is driven by configurable selectors: a label key/value (`OPERATOR_WATCH_LABELS`), a compose project name (`OPERATOR_WATCH_COMPOSE_PROJECTS`), and/or explicit container-name patterns (`OPERATOR_WATCH_NAMES`). When no selector is configured in Docker mode it SHALL discover all running containers.

#### Scenario: Discover by label
- **WHEN** `OPERATOR_WATCH_LABELS` is set
- **THEN** only containers carrying the matching label are tracked

#### Scenario: Discover by compose project
- **WHEN** `OPERATOR_WATCH_COMPOSE_PROJECTS` is set
- **THEN** only containers belonging to the named compose projects are tracked

#### Scenario: No selector configured
- **WHEN** no watch selectors are configured in Docker mode
- **THEN** all running containers are discovered and tracked

#### Scenario: Container lifecycle
- **WHEN** a tracked container stops or is removed
- **THEN** it is removed from tracking and its state-model entry is finalized/cleaned up
