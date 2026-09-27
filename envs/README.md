# Runtime environments

Each subdirectory is a standalone uv project with a resolved lockfile.
Evaluator setup is documented in
[Evaluation protocol](../docs/evaluation.md); materialization environments are
documented in [Data materialization](../docs/materialization.md). Candidate
environments and setup instructions live beside their integrations under
<code>src/dotebench/candidates/</code>.

Keep all created environments outside the repository checkout.
