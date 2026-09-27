# Step-Audio-EditX

This integration uses the official
[Step-Audio-EditX](https://github.com/stepfun-ai/Step-Audio-EditX) source pinned
by <code>runtime.json</code>. Its benchmark ID is
<code>step-audio-editx</code> and its service configuration is
<code>configs/step_audio_editx.yaml</code>.

Text-only calls use the native TTS path with the source transcript as prompt text
and the XML-rendered target transcript as output text. Emotion and speed use the
native editing templates. Pitch and pause insertion use free-form global
instructions. Pause reduction uses native VAD editing for a single operation and
the free-form silence-removal interface for a compositional request.

Compositional <code>one_take</code> input places the complete target transcript
in the editing template and selects the first emotion, prosody, or pause
operation in source/XML order.

## Compiler behavior

The native interface provides separate text-generation and whole-utterance
editing paths. Applying text edits first establishes the target transcript
before control edits modify the resulting speech.

<code>one_take</code> makes one native call for the case.
<code>sequential</code> performs text tags first through the TTS path, then
applies every emotion, prosody, and pause tag in its own call with the preceding
output.

Evaluation scores the complete XML.

## Setup and run

~~~bash
git submodule update --init third_party/step-audio-editx
python src/dotebench/candidates/setup_environment.py \
  --candidate step-audio-editx \
  --environment-root /path/to/envs \
  --cache-dir /path/to/uv-cache \
  --python-install-dir /path/to/uv-python
~~~

| Variable | Purpose |
|---|---|
| <code>DOTEBENCH_CANDIDATE_PYTHON</code> | Locked Step-Audio-EditX interpreter |
| <code>DOTEBENCH_MODEL_PATH</code> | Local model checkpoint |
| <code>DOTEBENCH_TOKENIZER_PATH</code> | Local tokenizer checkpoint |
| <code>DOTEBENCH_SOURCE_ROOT</code> | Repository checkout |
| <code>DOTEBENCH_RUNTIME_ROOT</code> | Prepared native runtime root |
| <code>DOTEBENCH_SERVICE_OUTPUTS</code> | Service logs and generated scratch files |
| <code>DOTEBENCH_GPU_IDS</code> | Service GPU assignment |
| <code>DOTEBENCH_IPC_ROOT</code> | Shared-memory path for vLLM IPC |

~~~bash
uv run --frozen dotebench generate \
  --candidate step-audio-editx \
  --compiler one_take \
  --config configs/step_audio_editx.yaml \
  --data-root /path/to/materialized-data \
  --audio-root /path/to/materialized-data \
  --run /path/to/runs/step-audio-editx
~~~

The launcher verifies the source commit, patch hashes, locked environment, and
runtime identity before starting the managed service.

## Tests

~~~bash
PYTHONPATH="$DOTEBENCH_SOURCE_ROOT/src" \
  "$DOTEBENCH_CANDIDATE_PYTHON" -m pytest \
  tests/candidates/step_audio_editx
~~~
