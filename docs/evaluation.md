# Evaluation protocol

The V1.0 scoring protocol is identified as `v1_0` in
`release/evaluation-protocol.json` and in evaluation run records.

See [Using doteBench](usage.md) for the complete generate, evaluate, and report
workflow. This page defines evaluator preparation and formal scoring.

The evaluator scores frozen cases using a shared set of metric abilities.
`release/evaluation-protocol.json` declares the scoring contract identity used
for scoring, failure penalties and emotion classification. The evaluation
runner records it in each summary. Manifests determine category and shard
grouping; generated results contain candidate outputs only.

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

Install Praat 6.1.38 into the CPU service environment (Linux x86_64):

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

`evaluation/models.py` owns the model registry and expected lock identities.
`services/utils/model_assets.py` prepares the supplied files against those locks
and records their inventory and identity in local receipts. Preparation verifies
existing model files; obtain the snapshots and checkpoints before running it.
Each backend owns its checkpoint filenames and loader requirements.

Service startup verifies its model ID, receipt, file inventory, sizes, and
timestamps. The evaluator validates the returned lock hash against its pinned
policy. `prepare-evaluation-models --check` also rehashes file contents.

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

Content preservation uses Qwen3-ASR recognition error. Local acoustic preservation
uses word-level duration DTW and voiced F0 statistics. Speaker similarity uses
WavLM with ECAPA-TDNN; whole-utterance quality uses UTMOS22. Emotion classification
uses Qwen3-Omni with a deterministic candidate-label order and a fixed prompt governed by the evaluator protocol.

### Recognition error

`services/backends/wer.py` owns text normalization, WER/CER kernels, and the
HTTP entrypoint. `evaluation/wer.py` attributes errors to XML regions and pools
counts across samples.

Qwen3-ASR runs through its Transformers backend in bfloat16, with one utterance
per inference call and a 512-token output limit. Service configuration records
these inference parameters alongside the pinned model revision.

Local WER uses XML text-edit regions with three reference tokens of context on
each side. A deletion contributes its zero-width target boundary and the adjacent
context. Inserted hypothesis tokens belong to the following reference token, or
the final reference token at utterance end; edit and non-edit error counts form a
partition of global counts. Recognition alignment uses the standard jiwer
edit-distance computation.

### Alignment and interval queries

Forced alignment uses Qwen3 with explicit benchmark word units. Source alignments
are frozen per sample in the manifest. Evaluation aligns each generated audio
exactly once against the frozen target text, even when the
generated speech differs from that text. Low transcription accuracy is not an
alignment rejection criterion.

One generated audio produces one alignment shared by every metric consumer.
Operation queries extract intervals from that same alignment. Source queries use
the manifest alignment. Pause queries select the left word's end and right word's
start and clamp negative gaps to zero. A zero target rate duration remains zero
in duration error and directional success calculations.

The aligner service accepts `POST /measure` with `text`, `alignment`, `queries`
and `audio_duration` for pure extraction; `alignment` is an ordered list of
`word`, `start`, `end` records and each query is the full transcript with one
`<q>...</q>` region. This request uses no audio or neural inference and returns
`answers`, each containing `start` and `end`. The evaluator calls the same pure
function locally. Audio alignment requests retain `audio`, `text`, and `language`.

### Acoustic preservation

`evaluation/preservation.py` selects XML scopes and orchestrates WDTW.
`services/backends/wdtw.py` owns duration and F0 preservation kernels and their
HTTP entrypoint. WDTW declares F0 as a service dependency and calls its API;
`services/backends/f0.py` owns the Praat worker pool.

Instruction spans define the evaluated and preserved regions. Unmodified words
participate in both duration and F0 preservation. Pitch regions remain eligible
for duration preservation, while rate regions remain eligible for F0
preservation. Pitch regions instead use the requested-shift metric for F0, and
rate regions use the requested-ratio metric for duration. Emotion and text-edit
regions are excluded from both preservation branches. Pause instructions are
zero-width, so their neighboring unmodified words remain eligible.

The evaluator selects preservation word pairs from XML node identity and the
complete source/target alignments. Repeated words do not require subsequence
matching. `evaluation.preservation.evaluate_preservation` sends the complete
recordings as `source_audio` and `target_audio`, plus four selected segment lists:
`duration_source`, `duration_target`, `f0_source`, and `f0_target`. Each segment
contains `word`, `start`, `end`, and optional `raw_word`; paired F0 lists must have
equal length. WDTW computes and returns `wdtw_dur` and `wdtw_f0`. The evaluator
adds `selection` to the audit record. XML scope selection belongs to the evaluator;
the service accepts explicit segments. Deploy the evaluator and WDTW backend
from the same source revision when updating this request contract.

F0 is measured over the complete recording before results are assigned to
requested spans. WDTW obtains its F0 measurements from the F0 service.

### F0 measurability

Source words with zero duration are removed, together with their corresponding
target words, from both preservation branches before F0 extraction. Target words
with zero duration remain in duration preservation, but are never submitted to
F0 extraction. WDTW F0 includes a word only when both sides provide finite positive
minimum, maximum and mean F0; other words contribute neither error nor normalizer.
Each measurable word contributes the mean of those three absolute semitone errors.
An empty metric has null value and zero sufficient statistics.

Pitch effects pair words within each XML operation by position. Both word durations
must be positive and both median F0 values finite and positive. The operation's
signed shift is the arithmetic mean of `12 * log2(target/source)` across these
pairs. Its error is the absolute difference from the requested shift. An operation
with no comparable pairs is unmeasurable, including when source and target have
F0 only at different word positions. Partial measurability retains the operation.
Whole-sentence pitch participates in duration preservation.

### Emotion classification

Whole-sentence emotion uses the complete output audio; local emotion uses an
aligned crop with 250 ms padding.

The emotion response parser accepts `neutral` as an alias of `calm` before
checking for a unique label. Raw classifier responses remain in evaluation records.

### Component success

Compositional Text success requires zero local recognition error; Emotion requires
the requested label. Pitch and rate components require at least 1 semitone and
10% duration change respectively, in the requested direction. Pause direction
success requires a gap change of at least 160 ms in the requested direction for
both standalone and compositional operations.

## Aggregation

All categories and shards pool sufficient statistics directly. Global and local
ER equal total substitutions, deletions and insertions divided by total reference
units (corpus ER). Empty references retain insertion errors; an aggregate with
zero reference units is null. WDTW equals total distance divided by total
normalizer, including Compositional. Category summaries pool the underlying
records rather than averaging shard summaries. Computation retains full precision;
rounding belongs to presentation.

Pitch and rate errors and component success rates weight XML operations equally.
Speaker similarity, UTMOS and all-component success weight samples equally.
An unmeasurable pitch operation contributes to neither pitch error nor component
success. All-component success tests the remaining operations; a sample with no
remaining operations has no contribution to that metric. Distance and normalizer retain the sufficient statistics for aggregation.

## Failure policy

Candidate generation failures remain scored: recognition error is 100% for a
nonempty reference, speaker similarity 0, UTMOS 1, WDTW-Dur 1, WDTW-F0 8 semitones,
and pitch error 8 semitones. All instructed components fail. Preservation penalties
apply only to XML-preserved source objects: duration uses preserved source seconds
with zero target duration; F0 uses the number of measurable preserved source words.
There is no preservation penalty when the metric has no reference objects. Local
ER failure counts use the same XML target-region partition as ordinary evaluation.
Rate failure error is source duration divided by the requested factor for every
category.

For generated audio, an empty emotion region fails locally without invoking the
classifier. A zero target rate duration remains zero, including in direction
success. A zero pause gap is a valid gap measurement. Unmeasurable F0 follows the
exclusions above and is distinct from a missing service response.

Missing metric responses, unavailable services, invalid alignments, and corrupted
stored audio are evaluation infrastructure errors. These leave the evaluation
incomplete and prevent a formal report. Frozen source checksums and alignment
contracts are verified for both successful and failed generation records.

## Verification

The test suite includes frozen reference outcomes for metric kernels, preservation
masks, aggregation weights, failure penalties, emotion input scope, and silence
boundaries. CPU metric services have live HTTP lifecycle tests. Neural service
inference requires the corresponding prepared checkpoints and service environments.
