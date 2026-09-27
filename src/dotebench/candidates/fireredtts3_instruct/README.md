# FireRedTTS3-Instruct

This integration uses the official
[FireRedTTS3](https://github.com/FireRedTeam/FireRedTTS3) source pinned by
<code>runtime.json</code>. Its benchmark ID is
<code>fireredtts3-instruct</code> and its service configuration is
<code>configs/fireredtts3_instruct.yaml</code>.

Insert, delete, and substitute operations use the native semantic-editing
interface. The compiler expands ambiguous or repeated text to the smallest
unique lexical context. Emotion, prosody, and pause edits use one global
instruction. Rate is mapped to the nearest supported 0.1 step
from 0.5 to 2.0 on the requested side of 1.0; pitch is mapped to the nearest
nonzero integer from -6 to 6 on the requested side.

Requests use seed 1234, 10 flow steps, and CFG 1.2 over the full source audio.

## Compiler behavior

<code>one_take</code> selects the first operation by source position and XML
order. <code>sequential</code> applies every text tag first, then every emotion,
prosody, and pause tag, using each output as the next source.

Semantic editing is local. Emotion, prosody, and pause controls apply to the
complete utterance. Emotion and pause prompts extend the public single-control
interface. The original XML values remain in the audit record, and evaluation
scores every original operation.

## Setup and run

~~~bash
git submodule update --init third_party/fireredtts3
python src/dotebench/candidates/setup_environment.py \
  --candidate fireredtts3-instruct \
  --environment-root /path/to/envs \
  --cache-dir /path/to/uv-cache \
  --python-install-dir /path/to/uv-python
~~~

| Variable | Purpose |
|---|---|
| <code>DOTEBENCH_CANDIDATE_PYTHON</code> | Locked FireRedTTS3 interpreter |
| <code>DOTEBENCH_MODEL_PATH</code> | Local FireRedTTS3-Instruct checkpoint |
| <code>DOTEBENCH_SOURCE_ROOT</code> | Repository checkout |
| <code>DOTEBENCH_RUNTIME_ROOT</code> | Prepared native runtime root |
| <code>DOTEBENCH_SERVICE_OUTPUTS</code> | Service logs and generated scratch files |
| <code>DOTEBENCH_GPU_IDS</code> | Service GPU assignment |
| <code>DOTEBENCH_CANDIDATE_PORT</code> | Optional service port override |

~~~bash
uv run --frozen dotebench generate \
  --candidate fireredtts3-instruct \
  --compiler one_take \
  --config configs/fireredtts3_instruct.yaml \
  --data-root /path/to/materialized-data \
  --audio-root /path/to/materialized-data \
  --run /path/to/runs/fireredtts3-instruct
~~~

The launcher verifies the official source commit, locked environment, and
runtime identity before service startup.
Candidate generation failures are scored; service and transport failures leave
the run incomplete.

## Tests

~~~bash
PYTHONPATH="$DOTEBENCH_SOURCE_ROOT/src" \
  "$DOTEBENCH_CANDIDATE_PYTHON" -m pytest \
  tests/candidates/fireredtts3_instruct
~~~
