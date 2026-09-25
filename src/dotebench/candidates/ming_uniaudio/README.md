# Ming-UniAudio

This integration uses the official
[Ming-UniAudio](https://github.com/inclusionAI/Ming-UniAudio) source pinned by
<code>runtime.json</code>. Its benchmark ID is <code>ming-uniaudio</code> and
its service configuration is <code>configs/ming_uniaudio.yaml</code>.

The compiler preserves the official speech-plus-instruction conversation
format. Text-only calls request the exact XML-rendered target transcript and
enable the model's editing chain-of-thought path. Other calls compile the
selected emotion, prosody, or pause operation into one global instruction.
Compositional <code>one_take</code> input combines the complete target
transcript with the first emotion, prosody, or pause operation in source/XML
order.

Requests use seed 1895 and a 45-second audio limit. The inference patch expands
the audio semantic encoder rotary cache to 8,192 positions while preserving
existing entries.

## Compiler behavior

The native interface accepts a speech-plus-instruction conversation for a
complete utterance. Emotion, prosody, and pause requests become global controls.

<code>one_take</code> sends one native request for the complete case.
<code>sequential</code> applies text tags first, then one global emotion,
prosody, or pause instruction per call, using each output as the next input.

Evaluation always scores every operation in the original XML. A
<code>no_audio_tokens</code> response is a candidate failure.

## Setup and run

~~~bash
git submodule update --init third_party/ming-uniaudio
python src/dotebench/candidates/setup_environment.py \
  --candidate ming-uniaudio \
  --environment-root /path/to/envs \
  --cache-dir /path/to/uv-cache \
  --python-install-dir /path/to/uv-python
~~~

| Variable | Purpose |
|---|---|
| <code>DOTEBENCH_CANDIDATE_PYTHON</code> | Locked Ming-UniAudio interpreter |
| <code>DOTEBENCH_MODEL_PATH</code> | Local model checkpoint |
| <code>DOTEBENCH_SOURCE_ROOT</code> | Repository checkout |
| <code>DOTEBENCH_RUNTIME_ROOT</code> | Prepared native runtime root |
| <code>DOTEBENCH_SERVICE_OUTPUTS</code> | Service logs and generated scratch files |
| <code>DOTEBENCH_GPU_IDS</code> | Service GPU assignment |
| <code>DOTEBENCH_CANDIDATE_PORT</code> | Optional service port override |

~~~bash
uv run --frozen dotebench generate \
  --candidate ming-uniaudio \
  --compiler one_take \
  --config configs/ming_uniaudio.yaml \
  --data-root /path/to/materialized-data \
  --audio-root /path/to/materialized-data \
  --run /path/to/runs/ming-uniaudio
~~~

The launcher verifies the source commit, patch hashes, locked environment, and
runtime identity before starting the managed service.

## Tests

~~~bash
PYTHONPATH="$DOTEBENCH_SOURCE_ROOT/src" \
  "$DOTEBENCH_CANDIDATE_PYTHON" -m pytest \
  tests/candidates/ming_uniaudio
~~~
