## ADDED Requirements

### Requirement: Mutation testing on extracted phase modules
The project SHALL run a mutation-testing campaign (`mutmut`) over the
extracted phase modules (`reconcile_phases.py`, `pod_collection.py`,
`bootstrap.py`) on demand (NOT per-push: the self-hosted runner is a
shared host), and SHALL publish the killed/survived counts per module.
Survivors SHALL be triaged in family batches; safety-critical mutation
families (external-command argv/timeout contracts, threshold polarity,
boolean gate polarity, persisted-payload key contracts) SHALL be fully
killed by contract-pinning tests.

#### Scenario: campaign runs and reports per-module results
- **WHEN** `python -m mutmut run` completes for the three scoped modules
- **THEN** `python -m mutmut results` reports killed/survived counts and
  the overall kill rate is recorded in the change tasks (baseline: 48.0%)
- **AND** the campaign is run ad hoc (manual/on-demand), never scheduled
  per-push, to protect the shared self-hosted runner

#### Scenario: safety-critical mutant families are killed
- **WHEN** a survivor mutates an external-command argv/timeout contract,
  a state-classification threshold comparison, a log-collection gate
  polarity, or a persisted-state payload key
- **THEN** a contract-pinning regression test is added that kills the
  entire family (e.g. exact argv assertion kills all string-casing and
  timeout±1 mutants at that call site)

#### Scenario: equivalent-mutant classes are documented, not chased
- **WHEN** survivors are dominated by log-format-string mutations,
  `getattr` default fallbacks on loosely-mocked branches, or numeric
  constants in rarely-asserted diagnostic paths
- **THEN** they are documented as accepted classes in the change tasks
  rather than pinned with brittle per-log-line assertions
