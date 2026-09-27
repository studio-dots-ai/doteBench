# Using doteBench

This guide covers installation, data materialization, candidate generation,
evaluation, and run records. Detailed contracts are documented in
[data](data.md), [materialization](materialization.md),
and [evaluation](evaluation.md). Candidate-specific setup lives beside each
implementation and is linked from the root README.

## Install

doteBench supports Python 3.10 through 3.12 and uses
[uv](https://docs.astral.sh/uv/) for locked environments.

~~~bash
git clone --recurse-submodules https://github.com/studio-dots-ai/doteBench.git
cd doteBench
export UV_PROJECT_ENVIRONMENT=/path/to/envs/dotebench-development
export UV_LINK_MODE=hardlink
uv sync --frozen --extra dev --extra metrics
uv run --frozen dotebench --help
uv run --frozen python -m dotebench.compilation check
~~~

Keep environments, model snapshots, generated audio, service logs, and runs
outside the checkout.

## Obtain and materialize the data

Set `BUNDLE_ARCHIVE` to the audio bundle archive path. Verify it with the
companion SHA-256 file, then extract it:

~~~bash
BUNDLE_DIR=$(dirname "$BUNDLE_ARCHIVE")
BUNDLE_FILE=$(basename "$BUNDLE_ARCHIVE")
(cd "$BUNDLE_DIR" && sha256sum --check "$BUNDLE_FILE.sha256")
tar -xzf "$BUNDLE_ARCHIVE"
~~~

A complete data root also requires Common Voice 20.0 and the prepared
materialization models. Follow [Data materialization](materialization.md), then
run:

~~~bash
export DOTEBENCH_SOURCE_ROOT="$PWD"
export DOTEBENCH_MODEL_ROOT=/path/to/models
export DOTEBENCH_ENV_ROOT=/path/to/envs

uv run --frozen dotebench materialize-data \
  --bundle-root /path/to/audio-bundle \
  --common-voice-root /path/to/common-voice-20.0 \
  --output-root /path/to/materialized-data
~~~

Verify construction identities, then validate benchmark inputs:

~~~bash
uv run --frozen dotebench verify-materialization \
  --data-root /path/to/materialized-data \
  --reference release/materializations/reference.json

uv run --frozen dotebench validate-data \
  --data-root /path/to/materialized-data \
  --audio-root /path/to/materialized-data
~~~

The first command checks how the data root was constructed. The second checks
that every selected case can be loaded and used.

## Generate candidate audio

The identity candidate copies source audio for a smoke test:

~~~bash
uv run --frozen dotebench generate \
  --candidate identity \
  --compiler one_take \
  --category text --shard easy \
  --data-root /path/to/materialized-data \
  --audio-root /path/to/materialized-data \
  --run /path/to/runs/identity-text-easy
~~~

For a learned system, select a supported candidate and configuration:

~~~bash
uv run --frozen dotebench generate \
  --candidate dots-tts-edit \
  --compiler one_take \
  --config configs/dots_tts_edit.yaml \
  --data-root /path/to/materialized-data \
  --audio-root /path/to/materialized-data \
  --run /path/to/runs/dots-tts-edit
~~~

<code>one_take</code> compiles the complete XML into one model call.
<code>sequential</code> applies one XML operation per call, with each output
becoming the next input. Follow the selected integration README for its native
request mapping, configuration, and capability boundary.

## Evaluate and report

Prepare the metric models registered in
<code>release/evaluation-models.json</code>:

~~~bash
uv run --frozen dotebench prepare-evaluation-models \
  --model-root /path/to/evaluation-models
~~~

Evaluate the case selection recorded by generation and render its report:

~~~bash
uv run --frozen --extra metrics dotebench evaluate \
  --data-root /path/to/materialized-data \
  --audio-root /path/to/materialized-data \
  --run /path/to/runs/dots-tts-edit \
  --config configs/evaluation_bundle.yaml \
  --override runtime.python="$DOTEBENCH_ENV_ROOT/dotebench-metrics-cpu/bin/python" \
  --override services.qwen3_asr.python_bin="$DOTEBENCH_ENV_ROOT/dotebench-qwen-asr/bin/python" \
  --override services.qwen3_aligner.python_bin="$DOTEBENCH_ENV_ROOT/dotebench-qwen-asr/bin/python" \
  --override services.utmos.python_bin="$DOTEBENCH_ENV_ROOT/dotebench-quality-speaker/bin/python" \
  --override services.speaker_similarity.python_bin="$DOTEBENCH_ENV_ROOT/dotebench-quality-speaker/bin/python" \
  --override services.emotion.python_bin="$DOTEBENCH_ENV_ROOT/dotebench-qwen-omni/bin/python"

uv run --frozen dotebench report --run /path/to/runs/dots-tts-edit
~~~

See [Evaluation protocol](evaluation.md) for model preparation, metrics, and
failure handling.

## Run records

A run is one benchmark execution for a fixed shard selection, candidate,
compiler, and resolved configuration. Generation and evaluation share one
directory:

~~~text
run.json
audio/<case-id>.wav
generation.jsonl
evaluation.jsonl
summary.json
~~~

<code>run.json</code> records one dataset version, selected manifest checksums,
case IDs, compiler identity, candidate configuration, runtime identity, service
graph, and stage state. A run
can cover one shard or all seven shards. Generation failures remain in the
evaluation denominator. Infrastructure failures leave the run incomplete and
prevent a formal report.
