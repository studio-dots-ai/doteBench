# Contributing to doteBench

Thank you for helping improve doteBench. Please open an issue before proposing a
change to the benchmark data contract, evaluation protocol, or benchmark
results. Bug fixes, portability improvements, tests, and documentation fixes are
welcome as focused pull requests. The [development guide](docs/development.md)
defines repository ownership, extension points, identities, and version rules.

## Development setup

Use Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/):

```bash
export UV_PROJECT_ENVIRONMENT=/path/to/envs/dotebench-development
export UV_LINK_MODE=hardlink
uv sync --frozen --extra dev --extra metrics
uv run --frozen pytest
uv run --frozen python -m dotebench.compilation check
```

Keep environments, model weights, generated audio, service logs, and evaluation
runs outside the repository checkout. Do not commit local paths or machine
identifiers.

## Contract rules

- The dataset version is declared in `release/shards.json`. Data contract
  changes require a review of this identity. Evaluator protocol changes require
  a separate review of the scoring contract.
- XML is the authoritative edit instruction. Do not add a second transcript or
  parsed-operation authority to the candidate request.
- Candidate compiler identities are derived from source SHA-256. Refresh a
  compiler manifest only after reviewing its effective model input.
- Candidate code must not read frozen evaluation alignments.
- Preserve third-party notices and update provenance when adding or replacing
  any data or model source.

Run `pytest tests/publication` before submitting a change. Candidate changes
also run `tests/candidates/<name>` with that candidate's locked environment;
see its README. Pull requests should explain the user-visible behavior and list
the checks performed.
