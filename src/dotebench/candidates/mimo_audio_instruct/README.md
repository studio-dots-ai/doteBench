# MiMo-Audio-Instruct

This integration uses the official
[MiMo-Audio](https://github.com/XiaomiMiMo/MiMo-Audio) source pinned by
<code>runtime.json</code>. Its benchmark ID is
<code>mimo-audio-instruct</code> and its service configuration is
<code>configs/mimo_audio_instruct.yaml</code>.

The current audio is supplied as the voice reference. The XML-rendered target
transcript and the natural-language instruction remain separate native
arguments. Text-only calls use an empty instruction. Other calls translate the
selected emotion, prosody, or pause operation into a bilingual global
instruction.

Requests use seed 42, WAV output, and a 282-token generation limit.
Compositional <code>one_take</code> input combines the complete target transcript
with the first emotion, prosody, or pause operation in source/XML order.

## Compiler behavior

The native interface synthesizes a complete utterance from a transcript and one
instruction. Emotion, prosody, and pause requests become global controls.

<code>one_take</code> makes one instruct-TTS call for the complete case.
<code>sequential</code> applies text tags first and then one emotion, prosody, or
pause tag per call, always using the preceding output as the next voice
reference.

Evaluation scores all operations in the original XML.

## Setup and run

~~~bash
git submodule update --init third_party/mimo-audio
python src/dotebench/candidates/setup_environment.py \
  --candidate mimo-audio-instruct \
  --environment-root /path/to/envs \
  --cache-dir /path/to/uv-cache \
  --python-install-dir /path/to/uv-python
~~~

| Variable | Purpose |
|---|---|
| <code>DOTEBENCH_CANDIDATE_PYTHON</code> | Locked MiMo-Audio interpreter |
| <code>DOTEBENCH_MODEL_PATH</code> | Local model checkpoint |
| <code>DOTEBENCH_TOKENIZER_PATH</code> | Local audio-tokenizer checkpoint |
| <code>DOTEBENCH_SOURCE_ROOT</code> | Repository checkout |
| <code>DOTEBENCH_RUNTIME_ROOT</code> | Prepared native runtime root |
| <code>DOTEBENCH_SERVICE_OUTPUTS</code> | Service logs and generated scratch files |
| <code>DOTEBENCH_GPU_IDS</code> | Service GPU assignment |
| <code>DOTEBENCH_CANDIDATE_PORT</code> | Optional service port override |

~~~bash
uv run --frozen dotebench generate \
  --candidate mimo-audio-instruct \
  --compiler one_take \
  --config configs/mimo_audio_instruct.yaml \
  --data-root /path/to/materialized-data \
  --audio-root /path/to/materialized-data \
  --run /path/to/runs/mimo-audio-instruct
~~~

The launcher verifies the source commit, patch hashes, locked environment, and
runtime identity before starting the managed service.

## Tests

~~~bash
PYTHONPATH="$DOTEBENCH_SOURCE_ROOT/src" \
  "$DOTEBENCH_CANDIDATE_PYTHON" -m pytest \
  tests/candidates/mimo_audio_instruct
~~~
