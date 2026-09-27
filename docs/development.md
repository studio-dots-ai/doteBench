# Development guide

This document defines the repository architecture and the process for extending
doteBench. User workflows are documented in [Using doteBench](usage.md).

## Ownership and data flow

The benchmark connects data construction, generation, and evaluation:

~~~text
manifest XML and evaluation annotations
        |
        +-- materialization -- complete data root
        +-- candidate compiler -- native request -- generated audio
        +-- evaluator -- per-case records -- aggregate report
~~~

<code>dotebench.instructions</code> owns XML validation, Unicode character
spans, operation regions, and deterministic source and target rendering.
<code>dotebench.compilers</code> owns one-take and sequential planning. Candidate
packages own native request construction and their service backends. Shared
speech and scoring backends live in `services/backends`; `services/utils` owns
configuration, dependency resolution, process lifecycle, and transport utilities.
Evaluation modules own XML scope selection, scoring policy, and aggregation.
Services may depend on shared root data types and utilities; they must not import
`candidates`, `evaluation`, or `materialization`, directly or transitively.
Shared backends own reusable computation and service entrypoints. Evaluators may
call backend computation functions directly. Importing a backend does not start
its service.
See [Evaluation protocol](evaluation.md) for metric implementations, model
preparation, and request contracts.

Materialization ends at a complete data root. Candidate generation and
evaluation read its manifests and content-addressed WAVs without depending on
provider recipes or materialization services.

## Repository layout

~~~text
src/dotebench/
├── instructions/       XML parsing and transcript rendering
├── compilers/          shared one-take and sequential planning
├── candidates/         model-specific compilers, adapters, and backends
├── services/
│   ├── utils/          configuration, process management, and shared utilities
│   └── backends/       one Python module per shared speech or scoring backend
├── materialization/    recipes, synthesis backends, and verification
├── evaluation/         formal scoring and aggregation
├── dataset.py          manifest loading and validation
├── execution.py        generation and evaluation runners
└── storage.py          persisted run records
data/                   benchmark manifests
release/                locks, provenance, notices, and reference identities
configs/                candidate and service configurations
envs/                   isolated evaluator and materialization environments
third_party/            pinned official upstream submodules
tests/                   offline public contract and regression tests
~~~

Each built-in candidate directory contains a README for its public setup,
model-input mapping, compiler behavior, and capability boundary.

## Add or update a candidate

A candidate integration consists of:

1. A registry entry declaring its public ID and compiler support.
2. <code>one_take.py</code> and, for editing candidates,
   <code>sequential.py</code> compiler entrypoints.
3. A model-specific <code>compile.py</code> that maps benchmark XML to native
   requests.
4. An adapter and backend boundary that classify candidate and infrastructure
   failures.
5. A flat YAML configuration under <code>configs/</code>.
6. For an imported native runtime, an official upstream gitlink, a locked
   environment, <code>runtime.json</code>, and the smallest reviewed inference
   patch.
7. Request-level tests that inspect the final native call, plus service lifecycle
   and failure-contract tests.

Candidate code may render transcripts with the public instruction parser or
implement an equivalent conversion. It cannot read manifest annotations or
frozen evaluation alignments. Auxiliary alignment belongs to the candidate and
must record its own model and cache identity.

## Compiler and runtime identities

Compiler manifests live beside each compiler entrypoint. They register static
imports, transitive source hashes, dependencies, and the aggregate SHA-256.
The fingerprint algorithm is defined in `compilation.py`. Generation checks
the manifests before startup and after execution.

~~~bash
python -m dotebench.compilation check
python -m dotebench.compilation refresh \
  --candidate ming-uniaudio \
  --compiler sequential
~~~

Refresh a manifest only after reviewing the effective model input and running
request tests. Frozen runs retain the identity recorded at creation.

Native runtime identity is separate. It covers the official commit, patch,
service adapter, common launcher, and environment lock. Startup verifies the
gitlink, upstream URL, clean source tree, patch hashes, and prepared runtime.
The service reports the captured identity through <code>/health</code>, and
generation checks it against the installed integration resources.

## Service graph

Every declaration has an instance key, endpoint, health path, startup timeout,
GPU assignment, environment, and argument-array command. Nested dependencies
start before their consumers. Repeated instance keys must resolve to identical
settings. Cycles and conflicts fail during composition.

The process manager records startup metadata and separate logs, terminates
process groups on failure or interruption, and passes only resolved direct
endpoints to the consumer. Model loading and request semantics remain in the
backend. Each backend owns its request schema, routes, concurrency policy, model
loading, and resource cleanup. Shared HTTP utilities accept explicit handlers
and lifecycle callbacks. Evaluators and candidate adapters validate the service
identities required by their own workflows.

Asset utilities prepare and verify files against caller-supplied model locks.
Callers own their model registry and expected identities; backends own checkpoint
and loader requirements.

Candidate services live beside their compilers and adapters. Shared backend
loaders can also support candidate-owned auxiliary services.

## Add or update an evaluation ability

1. Implement reusable measurement and its service entrypoint in
   `services/backends/`, using shared transport and lifecycle utilities.
2. Declare its environment, command, and direct dependencies in `configs/`.
3. Connect the ability through `evaluation/abilities.py`; put benchmark scope
   selection, scoring, and aggregation in `evaluation/`.
4. Register any evaluation model and its expected lock in `evaluation/models.py`
   and `release/evaluation-models.json`.
5. Cover computation, request validation, service lifecycle, and evaluator
   behavior with offline tests, and document preparation and scoring in the
   [Evaluation protocol](evaluation.md).

Review the public contract version whenever the scoring behavior changes.

## Public contracts and versions

The dataset version is declared by <code>release/shards.json</code>; the evaluator
protocol has its own version in <code>release/evaluation-protocol.json</code>. Python distribution
metadata has an independent package version. Compiler and runtime identities
describe their source and dependency content.

Changes to manifest semantics, XML grammar, evaluation formulas, failure
penalties, or aggregation require a new public contract version. Source-only
compiler changes require refreshed compiler identities. Dependency releases and
upstream model revisions keep their upstream versions.

## Development checks

Use the root locked environment:

~~~bash
export UV_PROJECT_ENVIRONMENT=/path/to/envs/dotebench-development
export UV_LINK_MODE=hardlink
uv sync --frozen --extra dev --extra metrics

uv run --frozen pytest
uv run --frozen python -m dotebench.compilation check
PYTHONPATH=src uv run --frozen python scripts/build_text_provenance.py --check
~~~

The default suite is CPU-only and offline. It uses small fixtures to test stable
benchmark contracts: XML semantics, manifest validation, scoring, service
orchestration, materialization identities, and public packaging. Run the compiler
and provenance validators alongside pytest.

Each model integration has one self-contained suite under
<code>tests/candidates/&lt;name&gt;/</code>. It covers that candidate's compiler,
adapter, transport, and native service boundary. Run it with the locked
candidate environment documented in that candidate's README. Shared candidate
test helpers and cross-candidate boundary checks live under
<code>tests/candidates/common/</code>. Candidate suites are included in the
source distribution and are outside the wheel package.

Full-manifest scans, model-backed smoke tests, historical comparisons, and
release-bundle audits are release validation work. Store their programs, logs,
and machine-readable results in the project artifact workspace rather than the
repository. Changes to public files run
<code>tests/publication/test_repository.py</code>.

## Release maintenance

Release builders in <code>scripts/</code> produce versioned inputs:

- <code>build_audio_bundle.py</code> creates the redistributable audio archive
  and checksum from the provenance ledger;
- <code>build_materialization_metadata.py</code> refreshes reference
  materialization metadata from a verified data root;
- <code>build_text_provenance.py</code> rebuilds and checks the public text
  provenance ledger.

Release investigation, comparison, and one-off audit programs belong in the
project artifact workspace.
