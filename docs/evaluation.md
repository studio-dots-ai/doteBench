# Evaluation protocol

Precise editing requires evidence that the requested change occurred while
preserving speech outside its scope. doteBench scores instruction following,
local preservation, and overall audio quality against frozen cases. XML defines
what should change; the source recording and annotations define the reference.

The scoring identity is declared in `release/evaluation-protocol.json` and
recorded in each evaluation summary. Compare runs under the same scoring
contract. Re-evaluation may reuse generated audio and valid raw measurement
caches, but writes new evaluation artifacts.

See [Using doteBench](usage.md) for the generate, evaluate, and report workflow.

## Prepare the evaluator

Evaluation uses four isolated environments:

| Environment | Abilities |
|---|---|
| <code>metrics-cpu</code> | WER, F0, and WDTW |
| <code>qwen-asr</code> | Qwen3 ASR and forced alignment |
| <code>qwen-omni</code> | Qwen3 Omni emotion classification |
| <code>quality-speaker</code> | UTMOS and WavLM ECAPA similarity |

Create them outside the checkout from the lockfiles under <code>envs/</code>.
The environment names below are also used by the service overrides:

~~~bash
export DOTEBENCH_ENV_ROOT=/path/to/envs

UV_PROJECT_ENVIRONMENT="$DOTEBENCH_ENV_ROOT/dotebench-metrics-cpu" \
  uv sync --project envs/metrics-cpu --frozen
UV_PROJECT_ENVIRONMENT="$DOTEBENCH_ENV_ROOT/dotebench-qwen-asr" \
  uv sync --project envs/qwen-asr --frozen
UV_PROJECT_ENVIRONMENT="$DOTEBENCH_ENV_ROOT/dotebench-qwen-omni" \
  uv sync --project envs/qwen-omni --frozen
UV_PROJECT_ENVIRONMENT="$DOTEBENCH_ENV_ROOT/dotebench-quality-speaker" \
  uv sync --project envs/quality-speaker --frozen
~~~

Install the pinned Praat build into the CPU service environment (Linux x86_64):

~~~bash
"$DOTEBENCH_ENV_ROOT/dotebench-metrics-cpu/bin/python" scripts/install_praat.py
~~~

The [installer](../scripts/install_praat.py) downloads the official server build,
checks the archive and executable SHA-256, and installs it as `bin/praat` in that
Python environment. Repeating the command reuses the verified installation.
The F0 service uses this environment-local executable automatically. Praat source
and license information is recorded in [NOTICE](../NOTICE).

Model assets are fixed by <code>release/evaluation-models.json</code>. Put the
registered snapshots and checkpoint files under one root, then prepare and
verify them:

~~~bash
export DOTEBENCH_MODEL_ROOT=/path/to/evaluation-models
dotebench prepare-evaluation-models --model-root "$DOTEBENCH_MODEL_ROOT"
dotebench prepare-evaluation-models \
  --model-root "$DOTEBENCH_MODEL_ROOT" \
  --check
~~~

### Model identities

Prepare the registered model files before evaluation. Preparation verifies the
supplied snapshots and checkpoints and records their identity in local receipts.
Service startup verifies those receipts and the evaluator checks the pinned
model identity. `prepare-evaluation-models --check` also rehashes file contents.

### Service configuration

<code>configs/evaluation_bundle.yaml</code> composes the eight metric services.
Prepare the external runtime dependencies described in [Runtime environments](../envs/README.md).
Set <code>DOTEBENCH_ENV_ROOT</code> and <code>DOTEBENCH_MODEL_ROOT</code>, then
supply these interpreter mappings to <code>dotebench evaluate</code>:

~~~bash
dotebench evaluate \
  --data-root /path/to/materialized-data \
  --audio-root /path/to/materialized-data \
  --run /path/to/run \
  --config configs/evaluation_bundle.yaml \
  --override runtime.python="$DOTEBENCH_ENV_ROOT/dotebench-metrics-cpu/bin/python" \
  --override services.qwen3_asr.python_bin="$DOTEBENCH_ENV_ROOT/dotebench-qwen-asr/bin/python" \
  --override services.qwen3_aligner.python_bin="$DOTEBENCH_ENV_ROOT/dotebench-qwen-asr/bin/python" \
  --override services.utmos.python_bin="$DOTEBENCH_ENV_ROOT/dotebench-quality-speaker/bin/python" \
  --override services.speaker_similarity.python_bin="$DOTEBENCH_ENV_ROOT/dotebench-quality-speaker/bin/python" \
  --override services.emotion.python_bin="$DOTEBENCH_ENV_ROOT/dotebench-qwen-omni/bin/python"
~~~

Add deployment-specific GPU overrides when the defaults do not match the local
layout. doteBench validates the resolved graph, starts dependencies in order,
records effective model and service identities, and closes owned processes on
completion or failure.

## Metrics

| Category | Instruction following |
|---|---|
| Text | Recognition error in edited text regions |
| Emotion | Requested emotion classification accuracy |
| Prosody | Absolute pitch-shift error in semitones and duration error in seconds |
| Pause | Correct direction of gap change |
| Compositional | Per-family and all-component success |

Qwen3-ASR measures content accuracy, WDTW measures local duration and F0
preservation, WavLM with ECAPA-TDNN measures speaker similarity, and UTMOS22
measures whole-utterance quality. Qwen3-Omni classifies emotion using a fixed
prompt and deterministic label order.

### Reference eligibility and edit scope

Eligibility follows the instruction and source evidence. Unmodified words
participate in duration and F0 preservation. Pitch edits remain eligible for
duration preservation, and rate edits for F0 preservation; each edited attribute
is assessed against its requested change. Emotion and text-edit regions are
excluded from acoustic preservation. Pause edits leave neighboring words
eligible for preservation.

F0 comparisons require measurable source voicing, and duration references require
positive source duration. Naturally unvoiced source words do not contribute to
F0 scores. Preservation with no eligible reference is undefined. A pitch or rate
operation without its required source reference is a data contract error.
Candidate output cannot change these source requirements.

### Output realization and measurement evidence

Local acoustic measurements share one output alignment against the frozen
target transcript; source intervals use frozen alignments. Alignment locates requested
regions but does not establish that their words were spoken. Recognition error
provides that separate content evidence.

Success requires the acoustic support needed to measure the requested edit.
Pitch requires target voicing for every source-eligible word. Rate requires
positive target duration for every source-supported word and for the region.
Pause requires realized neighboring words to anchor its gap, using the audio
boundary at utterance ends. A genuine zero gap is valid; a missing anchor is not
evidence of a reduced pause. A zero source gap remains a valid reference.

Missing required target evidence fails the operation and remains represented in
its score. Pitch averages word shifts over the source-eligible set, treating
missing target evidence as no demonstrated shift. Missing target F0 incurs the
protocol's word-level preservation penalty. Rate retains the observed duration
error even when required words collapse. Evaluation records distinguish measured
evidence from scoring conventions so a penalty is not mistaken for an acoustic
measurement.

### Metric definitions

Recognition error counts substitutions, deletions, and insertions relative to
reference words or characters. Local text error uses XML edit regions with
adjacent reference context, including deletion boundaries.
Edit and non-edit counts partition the global counts.

WDTW duration compares preserved word durations using dynamic time warping,
normalized by their combined source and target duration. F0 preservation averages
absolute semitone differences in word minimum, maximum, and mean F0, with equal
weight for each source-eligible word. F0 is extracted from the complete recording
before assigning measurements to spans.

Pitch error is the absolute difference between the requested shift and the mean
word shift computed from source and target median F0. Rate error is the absolute
difference between observed region duration and source duration divided by the
requested rate factor. Whole-sentence emotion uses the full output; local
emotion uses an aligned crop with surrounding context. Raw classifier responses
remain available with the scores.

Text component success requires zero local recognition error; emotion requires
the requested label. Pitch, rate, and pause require a change in the requested
direction that meets the protocol's minimum effect threshold. Standalone and
compositional operations share these criteria and realization requirements.

## Fair aggregation

Every requested operation contributes to operation success, and every case to
all-operations success. A case succeeds only when all its operations succeed.
Missing output cannot improve success rates by reducing their denominators.
Pitch and rate errors also weight operations equally; speaker similarity and
UTMOS weight cases equally.

Recognition error pools total error counts over total reference units. Empty
references retain insertion errors; a pool with no reference units is undefined.
WDTW pools distances and normalizers, including Compositional cases. F0 retains
all source-eligible words, including those with missing target measurements.
Category summaries pool underlying records across shards. Rounding is reserved
for presentation.

## Failure handling and reproducibility

Candidate generation failures remain scored with the protocol's failure
penalties, and every requested operation fails. Preservation penalties apply only
to eligible source objects within XML-preserved regions. The same reference
requirements apply to successful and failed generations.

Unavailable services, malformed metric responses, invalid alignments, and
corrupted stored audio are evaluation errors. They leave the run incomplete and
prevent a formal report. An explicit unvoiced measurement is a valid outcome;
an incomplete or invalid service response is not acoustic evidence. Frozen
source checksums and alignment contracts are verified for every case.

Tests cover metric definitions, preservation scope, aggregation weights, missing
target evidence, failure penalties, emotion scope, and pause boundaries. CPU
services have live HTTP lifecycle tests. Neural inference requires prepared
checkpoints and the corresponding service environments.
