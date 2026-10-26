# Evaluation protocol

Precise editing requires evidence that the requested change occurred while
preserving speech outside its scope. doteBench scores instruction following,
local preservation, and overall audio quality against frozen cases. XML defines
what should change; the source recording and annotations define the reference.

The scoring identity in `release/evaluation-protocol.json` is recorded in each
summary. Compare runs under the same contract. Re-evaluation can reuse generated
audio and valid raw measurements while writing new evaluation artifacts.

See [Using doteBench](usage.md) for the generate, evaluate, and report workflow.

## Prepare the evaluator

Evaluation uses four isolated environments:

| Environment | Abilities |
|---|---|
| <code>metrics-cpu</code> | WER, F0, and WDTW |
| <code>qwen-asr</code> | Qwen3 ASR and forced alignment |
| <code>qwen-omni</code> | Qwen3 Omni emotion classification |
| <code>quality-speaker</code> | UTMOS and WavLM ECAPA similarity |

Create environments outside the checkout using the <code>envs/</code> lockfiles:

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

Install Praat 6.1.38 into the CPU service environment (Linux x86_64):

~~~bash
"$DOTEBENCH_ENV_ROOT/dotebench-metrics-cpu/bin/python" scripts/install_praat.py
~~~

The [installer](../scripts/install_praat.py) verifies the official build’s
checksums and installs `bin/praat` for the F0 service, reusing verified
installations. See [NOTICE](../NOTICE) for source and license information.

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

Preparation verifies registered snapshots and checkpoints and records their
identity in local receipts. Service startup verifies the receipts; evaluation
checks the pinned identity. `prepare-evaluation-models --check` rehashes files.

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
| Compositional | Per-family and all-operations execution and success |

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

Execution and success require valid target realization.
Pitch requires target voicing for every source-eligible word. Rate requires
positive target duration for every source-supported word and for the region.
Pause requires realized neighboring words to anchor its gap, using the audio
boundary at utterance ends. A genuine zero gap is valid; a missing anchor is not
evidence of a reduced pause. A zero source gap remains a valid reference.

Missing required target evidence fails both outcomes and remains in both
denominators. Pitch averages word shifts over the source-eligible set, treating
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

Pitch and rate execution require a directional effect. Its fixed minimum is
capped by the success interval’s lower magnitude bound so small requested
effects count. Success requires magnitude within a bounded proportional tolerance
around the request: signed semitone shifts for pitch and changes in speaking-speed
factor from unchanged speed for rate. Success implies execution;
overshooting can execute while failing success.

Text requires zero local recognition error, emotion the requested label, and
pause a directional gap change meeting its minimum effect threshold. Their
execution and success outcomes coincide. Standalone and compositional operations
share all criteria and realization requirements.

## Fair aggregation

Execution and success are aggregated separately by operation and family and as
all-operations outcomes per case, for standalone and compositional edits. Each
case passes an outcome only when every operation passes it. All requested
operations and cases remain in the denominators. Pitch and rate errors weight
operations equally; speaker similarity and UTMOS weight cases equally.

Recognition error pools total error counts over total reference units. Empty
references retain insertion errors; a pool with no reference units is undefined.
WDTW pools distances and normalizers, including Compositional cases. F0 retains
all source-eligible words, including those with missing target measurements.
Category summaries pool underlying records across shards. Rounding is reserved
for presentation.

## Failure handling and reproducibility

Candidate generation failures receive failure penalties and fail both outcomes
for every requested operation. Preservation penalties apply to eligible source
objects within XML-preserved regions. All generations share reference requirements.

Unavailable services, malformed metric responses, invalid alignments, and
corrupted stored audio are evaluation errors. They leave the run incomplete and
prevent a formal report. An explicit unvoiced measurement is a valid outcome;
an incomplete or invalid service response is not acoustic evidence. Frozen
source checksums and alignment contracts are verified for every case.

Tests cover scoring, preservation, aggregation, missing evidence, failures,
emotion scope, and pause boundaries. CPU services have live HTTP lifecycle tests.
Neural inference requires prepared checkpoints and service environments.
