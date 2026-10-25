# AuK

This integration uses the official [AuK](https://github.com/Tencent-Hunyuan/AuK)
Prompt Enhancer to prepare editing instructions before full-utterance generation.
Its benchmark ID is `auk-base-pe`; its service configuration is
`configs/auk_base_pe.yaml`.

## Compiler behavior

`one_take` sends the complete XML instruction to the Prompt Enhancer, which
selects a supported editing task. `sequential` uses the shared planner to prepare
one instruction per operation and feeds each generated waveform into the next step.

Instructions are cached before generation so repeated runs use the same prepared
requests. Audio preparation and duration estimation operate on each step's current
waveform. The cache uses planned source transcripts; these do not describe errors
in intermediate generated speech. Terminal refusals remain candidate failures.

## Setup and run

Initialize `third_party/auk` and create the locked external environment:

```bash
python src/dotebench/candidates/setup_environment.py \
  --candidate auk-base-pe --environment-root /path/to/envs \
  --cache-dir /path/to/uv-cache --python-install-dir /path/to/uv-python
```

Set `DOTEBENCH_MODEL_PATH`, `DOTEBENCH_TOKENIZER_PATH`,
`DOTEBENCH_WEIGHTS_MANIFEST`, `DOTEBENCH_SOURCE_ROOT`, and
`DOTEBENCH_RUNTIME_ROOT` to the supplied AuK assets and runtime directories.

Build the instruction cache with the official AuK source and a text LLM endpoint:

```bash
export LLM_API_KEY=...
python -m dotebench.candidates.auk_base_pe.build_cache \
  --auk-source /path/to/auk \
  --output /path/to/Artifacts/auk-pe-instructions \
  --base-url "$LLM_BASE_URL" --model "$LLM_MODEL_NAME"
```

Run generation with the completed cache:

```bash
export DOTEBENCH_CANDIDATE_PYTHON=/path/to/external/auk-base-pe/bin/python
dotebench generate --candidate auk-base-pe --compiler one_take \
  --audio-root /path/to/benchmark-audio --run /path/to/Artifacts/run \
  --override candidate.options.instruction_cache_dir=/path/to/Artifacts/auk-pe-instructions
```

Select `--compiler sequential` for sequential editing. Generation uses the cached
instructions and the official Prompt Enhancer's audio and duration preparation.
