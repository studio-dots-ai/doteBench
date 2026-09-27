# Third-Party Licenses

This inventory describes the doteBench source repository and Python
distribution. doteBench-authored material uses the root
[Apache License 2.0](LICENSE). Third-party attribution is preserved in [NOTICE](NOTICE) and the
component notices linked below.

Unless explicitly stated, this repository does not include or distribute
third-party model weights. Their download, use, and redistribution are governed
by the respective upstream model terms.

## Dependencies

### Included source

The speaker-similarity service includes adapted source under
`src/dotebench/services/backends/resources/speaker_models/`:

| Files | Use and upstream | License and attribution |
| --- | --- | --- |
| `ecapa_tdnn.py` | Speaker verification source adapted from [Microsoft UniSpeech](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification); its attribution to lawlict/ECAPA-TDNN is preserved | CC-BY-SA-3.0, including doteBench's modifications; [`LICENSE-UniSpeech`](src/dotebench/services/backends/resources/speaker_models/LICENSE-UniSpeech) and [`NOTICE.md`](src/dotebench/services/backends/resources/speaker_models/NOTICE.md) |
| `WavLM.py` | Adapted [Microsoft WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) implementation | MIT; [`LICENSE-WavLM`](src/dotebench/services/backends/resources/speaker_models/LICENSE-WavLM) |
| `modules.py` | WavLM support code incorporating fairseq source | MIT; [`LICENSE-fairseq`](src/dotebench/services/backends/resources/speaker_models/LICENSE-fairseq) |

The wheel and source distribution carry these files, license texts, and the
speaker-model attribution notice. Checkpoint terms are supplied by their
respective providers.

### Python packages

Direct dependencies are declared in [`pyproject.toml`](pyproject.toml), with
resolved versions in [`uv.lock`](uv.lock). Service and candidate environments
have their own manifests and locks under [`envs/`](envs) and
[`src/dotebench/candidates/`](src/dotebench/candidates).

| Package | Use | License | Upstream |
| --- | --- | --- | --- |
| NumPy | Numerical arrays; required | BSD-3-Clause; bundled components also use 0BSD, MIT, Zlib and CC0-1.0 (see the resolved distribution notices) | [NumPy](https://github.com/numpy/numpy) |
| SoundFile | Audio I/O; required | BSD-3-Clause | [SoundFile](https://github.com/bastibe/python-soundfile) |
| Requests | HTTP requests; required | Apache-2.0 | [Requests](https://github.com/psf/requests) |
| Hydra Core | Configuration; required | MIT | [Hydra](https://github.com/facebookresearch/hydra) |
| jiwer | Word error rate; `metrics` | Apache-2.0 | [jiwer](https://github.com/jitsi/jiwer) |
| librosa | Audio analysis; `metrics` | ISC | [librosa](https://github.com/librosa/librosa) |
| FastAPI | Metrics service API; `metrics` | MIT | [FastAPI](https://github.com/fastapi/fastapi) |
| Uvicorn | Metrics service host; `metrics` | BSD-3-Clause | [Uvicorn](https://github.com/Kludex/uvicorn) |
| python-multipart | Multipart request parsing; `metrics` | Apache-2.0 | [python-multipart](https://github.com/Kludex/python-multipart) |
| pytest | Tests; `dev` | MIT | [pytest](https://github.com/pytest-dev/pytest) |
| Ruff | Linting; `dev` | MIT | [Ruff](https://github.com/astral-sh/ruff) |
| build | Distribution builds; `dev` | MIT | [build](https://github.com/pypa/build) |
| Hatchling | Isolated PEP 517 build backend | MIT | [Hatchling](https://github.com/pypa/hatch) |

## Models and optional capabilities

Source links below identify the pinned submodule revisions. Checkpoint identities
are recorded in [`release/evaluation-models.json`](release/evaluation-models.json)
and [`release/materialization-models.json`](release/materialization-models.json);
check the terms supplied with the specific checkpoint before enabling it.

| Component | Use | License / upstream terms |
| --- | --- | --- |
| [Praat 6.1.38](https://github.com/praat/praat.github.io/releases/tag/v6.1.38) | F0 analysis through a separate CLI process | [GPL-3.0-or-later](https://praat.org/manual/License.html) |

The [installer](scripts/install_praat.py) obtains Praat for the service environment;
the Python distributions carry the worker script, with the installer also in the
source distribution. Distributing an environment containing Praat requires
review of the applicable GPL license, attribution and corresponding-source
obligations; the process boundary alone does not determine those obligations.

| Component | Upstream source at pinned revision | Source terms and use |
| --- | --- | --- |
| AuK | [Source](https://github.com/Tencent-Hunyuan/AuK/tree/d9f30ffe4231dbc90b48cc83a35d310fece0b060) | MIT; optional candidate. Its upstream license expressly covers Tencent-published code, parameters, and weights. |
| FireRedTTS3 | [Source](https://github.com/FireRedTeam/FireRedTTS3/tree/7a1f3a7282ff184cc1c7f070556baaf5f08b5216) | Apache-2.0; optional candidate |
| IndexTTS-2 | [Source](https://github.com/index-tts/index-tts/tree/1eafa935e8887f9ba877f82cf7319fceddfc86cb) | [bilibili Model Use License Agreement](https://github.com/index-tts/index-tts/blob/1eafa935e8887f9ba877f82cf7319fceddfc86cb/LICENSE), covering its published final code and weights and setting conditions for a separate written license; materialization source |
| MiMo-Audio | [Source](https://github.com/XiaomiMiMo/MiMo-Audio/tree/62d956b4a1a45419bee5e41f477078c3684dbbcc) | Apache-2.0; optional candidate and materialization source |
| Ming-UniAudio | [Source](https://github.com/inclusionAI/Ming-UniAudio/tree/33f86a0a611b16ff1f29f6373303096d230c66dc) | MIT; optional candidate |
| Qwen3-TTS | [Source](https://github.com/QwenLM/Qwen3-TTS/tree/1ab0dd75353392f28a0d05d9ca960c9954b13c83) | Apache-2.0; materialization source |
| Step-Audio-EditX | [Source](https://github.com/stepfun-ai/Step-Audio-EditX/tree/a652e87052c109e26f616d60971376ff47a829d4) | Apache-2.0; optional candidate |
| VoxCPM | [Source](https://github.com/OpenBMB/VoxCPM/tree/5510503182e405fb78b6765d24d2e3db5d987a5c) | Apache-2.0; materialization source |

Evaluation services also use the following upstream models:

| Model | Use | Upstream terms |
| --- | --- | --- |
| [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Speech recognition | License files and model card at the locked revision |
| [Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | Word alignment | License files and model card at the locked revision |
| [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | Emotion recognition | License files and model card at the locked revision |
| [UTMOS / SpeechMOS](https://github.com/tarepan/SpeechMOS/tree/v1.2.0) | Speech quality | Upstream source and checkpoint terms |
| [WavLM speaker verification](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification) | Speaker similarity | Included source licenses are listed above; checkpoint terms follow its upstream distribution |

IndexTTS-2's agreement covers its published final code and weights, including
conditions requiring a separate written license. The adapter's acceptance flag
records acknowledgement; permission to use the component follows that agreement.
MiMo-Audio's locked materialization checkpoints use MIT terms, independently of
the source repository's Apache-2.0 license.

## Publication notes

When publishing a binary package, container, or other artifact containing
third-party software:

1. Preserve the applicable copyright notices, license texts, and upstream NOTICE
   files. The Python distributions include the bundled source licenses and notices
   linked above.
2. Update this inventory against the actual locked dependencies and include the
   license information for additional packages bundled in the artifact.
3. Distribute model weights only as permitted by their upstream licenses.

Benchmark audio and transcript terms and provenance are recorded in
[`release/DATA-NOTICE.md`](release/DATA-NOTICE.md) and
[`release/AUDIO-NOTICE.md`](release/AUDIO-NOTICE.md).
