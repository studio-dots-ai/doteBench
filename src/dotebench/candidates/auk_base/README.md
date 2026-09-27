# AuK Base

This integration uses the official [AuK](https://github.com/Tencent-Hunyuan/AuK)
source pinned by <code>runtime.json</code>. Its benchmark ID is
<code>auk-base</code> and its service graph is
<code>configs/auk_base.yaml</code>.

AuK Base receives a natural-language command compiled from the selected XML
operation. Text edits specify the complete resulting transcript. Whole-utterance
emotion and pitch requests preserve duration; rate requests divide duration by
the requested factor. Local emotion, prosody, and pause requests use a
candidate-produced alignment to select the generation window. Pause edits add
200 ms or subtract up to 200 ms.

Every request uses seed 0, 32 sampling steps, CFG 2.0, and 20 ms frame
quantization. The complete source audio conditions generation.

## Compiler behavior

AuK Base supports whole-utterance generation conditioned on one command. Local
editing uses candidate-owned alignment to select a generation window.

<code>one_take</code> performs the first operation in source/XML order.
Text edits and controls covering the complete transcript regenerate the whole
waveform. Local emotion, prosody, and pause edits regenerate an aligned window
and retain the surrounding waveform.

<code>sequential</code> applies every tag separately, with text tags first and
each output becoming the next source. Each local step aligns that current source.

Local precision depends on the candidate-owned
Qwen3-ForcedAligner-0.6B service. The aligner receives only current audio,
XML-rendered source text, and language; benchmark evaluation alignments are
unavailable to it. The alignment cache binds audio, text, language, weights, and
implementation identity.

## Setup and run

Initialize the official source and create the locked external environment:

~~~bash
git submodule update --init third_party/auk
python src/dotebench/candidates/setup_environment.py \
  --candidate auk-base \
  --environment-root /path/to/envs \
  --cache-dir /path/to/uv-cache \
  --python-install-dir /path/to/uv-python

export DOTEBENCH_ENV_ROOT=/path/to/envs
UV_PROJECT_ENVIRONMENT="$DOTEBENCH_ENV_ROOT/dotebench-qwen-asr" \
  uv sync --project envs/qwen-asr --frozen

export DOTEBENCH_MODEL_ROOT=/path/to/models
dotebench prepare-evaluation-models \
  --model-root "$DOTEBENCH_MODEL_ROOT" \
  --model qwen3_forced_aligner_0_6b
dotebench prepare-evaluation-models \
  --model-root "$DOTEBENCH_MODEL_ROOT" \
  --model qwen3_forced_aligner_0_6b \
  --check
~~~

Configure:

| Variable | Purpose |
|---|---|
| <code>DOTEBENCH_CANDIDATE_PYTHON</code> | AuK service interpreter |
| <code>DOTEBENCH_MODEL_PATH</code> | AuK checkpoint directory |
| <code>DOTEBENCH_TOKENIZER_PATH</code> | Qwen2.5-Omni-3B directory used by AuK |
| <code>DOTEBENCH_WEIGHTS_MANIFEST</code> | SHA-256 manifest for supplied weights |
| <code>DOTEBENCH_ALIGNER_PYTHON</code> | <code>$DOTEBENCH_ENV_ROOT/dotebench-qwen-asr/bin/python</code> |
| <code>DOTEBENCH_MODEL_ROOT</code> | Parent of <code>Qwen3-ForcedAligner-0.6B</code> |
| <code>DOTEBENCH_SOURCE_ROOT</code> | Repository checkout |
| <code>DOTEBENCH_RUNTIME_ROOT</code> | Prepared native runtime root |
| <code>DOTEBENCH_SERVICE_OUTPUTS</code> | Service logs, cache, and scratch data |
| <code>DOTEBENCH_GPU_IDS</code> | AuK service GPU assignment |
| <code>DOTEBENCH_ALIGNER_GPU_IDS</code> | Aligner GPU assignment |

~~~bash
uv run --frozen dotebench generate \
  --candidate auk-base \
  --compiler one_take \
  --config configs/auk_base.yaml \
  --data-root /path/to/materialized-data \
  --audio-root /path/to/materialized-data \
  --run /path/to/runs/auk-base
~~~

The runtime launcher verifies the official source commit, inference patch,
environment lock, and weight manifest. The service requires mono input and
validates token capacity and generation windows.

## Tests

Prepare the patched official runtime, then run the complete AuK Base suite with
the locked interpreter:

~~~bash
export DOTEBENCH_NATIVE_RUNTIME="$(
  uv run --frozen python -m dotebench.candidates.common.runtime \
    --candidate auk-base \
    --upstream "$DOTEBENCH_SOURCE_ROOT/third_party/auk" \
    --destination "$DOTEBENCH_RUNTIME_ROOT"
)"
PYTHONPATH="$DOTEBENCH_SOURCE_ROOT/src:$DOTEBENCH_NATIVE_RUNTIME/src" \
  "$DOTEBENCH_CANDIDATE_PYTHON" -m pytest \
  tests/candidates/auk_base
~~~
