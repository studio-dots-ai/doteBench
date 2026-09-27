# Data materialization

See [Using doteBench](usage.md) for the shortest end-to-end command sequence.
This page defines the construction inputs, model identities, services, and
verification contract.

doteBench defines benchmark metadata and deterministic audio recipes. A
materialization resolves those recipes into one complete data root:

```text
materialized-data/
├── audio/<sha256>.wav
├── data/<category>/<shard>/manifest.json
└── materialization.json
```

Generation and evaluation consume the manifests and audio in this data root.

## Recipe types

Each shard manifest contains an `audio_assets` mapping. Cases refer to one entry
with `source_audio.asset_id`. The entry is one of:

- `bundle`: copy a checksum-verified WAV from the redistributable doteBench audio
  bundle;
- `upstream`: verify a Common Voice 20.0 source MP3 and reproduce the frozen PCM16
  WAV conversion;
- `generate`: derive the source transcript from `instruction_xml`, invoke the
  named synthesis provider with its frozen seed and parameters, and optionally
  use another materialized asset as a reference prompt.

The recipes are authoritative for construction. The resulting `source_audio.path`
and `source_audio.sha256` are authoritative for all later benchmark stages.
`asset_id` remains stable when a generated WAV has a different content hash.

## Source code and model assets

The four synthesis implementations are official upstream Git submodules pinned
by `release/materialization-sources.json`:

| Provider | Official source | Service adapter | Environment |
|---|---|---|---|
| Qwen3-TTS | `third_party/qwen3-tts` | `dotebench.materialization.backends.qwen3_tts` | `envs/qwen3-tts` |
| IndexTTS-2 | `third_party/index-tts` | `dotebench.materialization.backends.indextts2` | `envs/indextts2` |
| VoxCPM2 | `third_party/voxcpm` | `dotebench.materialization.backends.voxcpm2` | `envs/voxcpm2` |
| MiMo-Audio | `third_party/mimo-audio` | `dotebench.materialization.backends.mimo_audio` | `envs/mimo-audio-materialization` |

The adapters expose `/health` and `/synthesize` and load the pinned upstream
implementations.
Service startup rejects a checkout whose origin, commit, or tracked files differ
from the release lock.

Model repository revisions and local directory names are pinned in
`release/materialization-models.json`. These weight snapshots are versioned
independently of the executable source checkouts. Put complete Hugging Face
snapshots below `DOTEBENCH_MODEL_ROOT`, then register their file identities:

```bash
dotebench prepare-materialization-models --model-root "$DOTEBENCH_MODEL_ROOT" \
  --accept-license indextts2
dotebench prepare-materialization-models --model-root "$DOTEBENCH_MODEL_ROOT" \
  --check
```

The acceptance flag records that the operator has handled IndexTTS-2's separate
model terms; it does not grant or reinterpret those terms. The command validates
existing snapshots and writes a local receipt. It does not download weights.

Create the four locked service environments outside the checkout:

```bash
git submodule update --init third_party/qwen3-tts third_party/index-tts \
  third_party/voxcpm third_party/mimo-audio
export DOTEBENCH_ENV_ROOT=/path/to/envs
for name in qwen3-tts indextts2 voxcpm2 mimo-audio-materialization; do
  UV_PROJECT_ENVIRONMENT="$DOTEBENCH_ENV_ROOT/dotebench-$name" \
    uv sync --project "envs/$name" --frozen
done
```

The Qwen3-ForcedAligner environment is the existing `envs/qwen-asr` project. It
is started only when newly generated audio differs from the reference materialization
and its frozen source alignment must be recomputed. Prepare that checkpoint with
`dotebench prepare-evaluation-models --model qwen3_forced_aligner_0_6b`; its
identity remains owned by `release/evaluation-models.json`.

## Build and run

Verify the audio bundle archive as described in [Using doteBench](usage.md).
Obtain Common Voice 20.0 from its
upstream distribution and preserve the paths recorded by each
<code>upstream</code> recipe. Then run:

```bash
export DOTEBENCH_SOURCE_ROOT="$PWD"
export DOTEBENCH_MODEL_ROOT=/path/to/models
export DOTEBENCH_ENV_ROOT=/path/to/envs

dotebench materialize-data \
  --data-root . \
  --bundle-root /path/to/audio-bundle \
  --common-voice-root /path/to/common-voice-20.0 \
  --output-root /path/to/materialized-data
```

doteBench starts one provider at a time, records the effective official source
and prepared model identities, and stops it before loading the next provider.
Each provider rehashes every file in its prepared model snapshot before loading
weights. MiMo-Audio uses its isolated Python 3.10 environment and the
locked FlashAttention 2.7.4.post1 CUDA 12/Torch 2.6 wheel.
The output receipt records every actual WAV SHA-256 and size, one canonical audio
digest, generated manifest hashes, alignment counts, and service identities.

## Verification and the reference

`release/materializations/reference.json` records reference audio identities and
canonical manifest hashes. Pass it to the verifier to compare construction
identities and outputs:

```bash
dotebench verify-materialization \
  --data-root /path/to/materialized-data \
  --reference release/materializations/reference.json
```

Bundle and upstream assets must match the reference exactly. Generated audio is
always content-addressed and remains valid when model or GPU kernels produce a
different waveform; the verifier reports those differences separately. Any
changed generated waveform is aligned again before the output manifests are
written. Verification also compares every output manifest with the packaged
release manifest: XML, case identity and order, language, recipes, seeds,
parameters, grouping annotations, and text checksums are immutable. Only the
materialized audio path and checksum may change; alignment audio identity and
segments may change only with the corresponding waveform. Provider source,
adapter, and model-lock identities are checked against the release locks. The
final aggregate digest identifies the contents of one materialization run.

The receipt records the dataset version. Reference manifest hashes use
category/shard keys; the verifier derives full paths from `release/shards.json`.

The audio provenance ledger and `release/AUDIO-NOTICE.md` define the
redistribution boundary of the bundle. Common Voice audio and the locally
generated assets are not part of that bundle.
