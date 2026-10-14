# Evaluation protocol

The current scoring protocol is identified as `v1_1` in
`release/evaluation-protocol.json` and in evaluation run records. It fixes scoring
of missing target measurements. Historical `v1_0` records retain their original
identity and must not be pooled with `v1_1`. Re-evaluation can reuse generated
audio and valid raw alignment/F0 caches, but must write new evaluation artifacts.
The dataset version is unchanged.

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
start and clamp negative gaps to zero. Required pause anchors must have positive
duration. At utterance boundaries only the existing neighboring word is required;
the other endpoint is the audio boundary. Positive-duration anchors may touch or
overlap and legitimately yield a zero gap.

Rate success requires positive target duration for every source-positive-duration
word in the XML region, as well as a positive region duration. The duration ratio
and error still use the observed region envelope, including a numeric zero for a
collapsed target. This alignment condition detects missing acoustic support;
recognition metrics separately assess content accuracy, since forced alignment
does not establish that the requested words were actually spoken.

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
F0 extraction. WDTW F0 eligibility depends only on finite positive source
minimum, maximum and mean F0. Each source-valid word contributes one unit to the
normalizer. When its target statistics are available, its error is the mean of
the three absolute semitone differences. When target duration or F0 is missing,
its word error is 8 semitones, reusing the existing empty-output penalty at the
same word-weighted granularity. This is a scoring convention, not an inferred
pitch measurement or an upper bound on measured errors. The word and metric
`status` record `penalized_missing_target_f0`; raw source/target F0 statistics
remain in `word_results`, and no synthetic measured statistics are inserted.
Source-unvoiced words do not contribute. An empty source-valid set has null
metric value and zero sufficient statistics, independently of candidate output.

Pitch effects pair words within each XML operation by position. Eligibility
requires positive source word duration and finite positive source median F0;
naturally unvoiced source words are not required target measurements. For each
eligible word, use `12 * log2(target/source)` when target duration and median F0
are available, or a scoring shift of zero (no edit) otherwise. The operation's
signed shift is the arithmetic mean over this source-defined set. Its error is
the absolute difference between that mean and the requested shift. Thus complete
target loss yields shift 0 and error `abs(requested shift)`; partial loss retains
missing words in the mean. Any missing required target word makes the operation
fail, even when the resulting mean reaches the directional threshold.
`measurement_status: missing_target` distinguishes scoring fallback from a
fully measured shift (`ok`). A pitch operation with no valid source reference
raises a data/evaluation contract error. Whole-sentence pitch participates in
duration preservation.

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
Every instructed operation contributes a boolean success to Component, including
output-side measurement failures. All tests every operation in a sample; missing
output cannot reduce either denominator. Pitch fallback errors remain finite and
contribute to the operation-weighted mean. Distance and normalizer retain the
sufficient statistics for aggregation.

## Failure policy

Candidate generation failures remain scored: recognition error is 100% for a
nonempty reference, speaker similarity 0, UTMOS 1, WDTW-Dur 1, WDTW-F0 8 semitones,
and pitch shift 0 with error equal to the absolute requested shift. All instructed
components fail. Preservation penalties
apply only to XML-preserved source objects: duration uses preserved source seconds
with zero target duration; F0 uses the number of measurable preserved source words.
There is no preservation penalty when the metric has no reference objects. Local
ER failure counts use the same XML target-region partition as ordinary evaluation.
Rate failure error is source duration divided by the requested factor for every
category.

For generated audio, an empty emotion region fails locally without invoking the
classifier. A collapsed or partially missing required target rate region fails
while retaining its observed duration error. Missing or zero-duration required
pause anchors produce `measurement_status: missing_target`, raw
`target_gap_sec: null`, and a scoring `pause_delta_sec: 0` (no edit). Both
`success` and `direction_correct` are false, including in direction accuracy.
With valid anchors, a genuine zero gap is scored normally. Invalid source pause
anchors or a rate region without positive-duration source words raise a contract
error. These source requirements also apply to candidate generation failures.

Explicit null F0 statistics represent unvoiced spans. Missing statistic fields,
non-numeric/non-finite/non-positive statistics, and incomplete service responses
are infrastructure errors, not candidate missingness.

Missing metric responses, unavailable services, invalid alignments, and corrupted
stored audio are evaluation infrastructure errors. These leave the evaluation
incomplete and prevent a formal report. Frozen source checksums and alignment
contracts are verified for both successful and failed generation records.

## Verification

The test suite includes frozen reference outcomes for metric kernels, preservation
masks, aggregation weights, failure penalties, emotion input scope, and silence
boundaries. CPU metric services have live HTTP lifecycle tests. Neural service
inference requires the corresponding prepared checkpoints and service environments.
