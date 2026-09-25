# dots.tts.edit

This integration uses the public
[dots.tts runtime](https://github.com/studio-dots-ai/dots.tts) and
[dots.tts.edit checkpoint](https://huggingface.co/dots-studio/dots.tts.edit).
Its benchmark ID is <code>dots-tts-edit</code> and its service configuration is
<code>configs/dots_tts_edit.yaml</code>.

The benchmark keeps standards-compliant XML. The compiler renders source and
target transcripts from that XML, then converts only the tags required by the
model:

~~~xml
<pitch semitones="4">higher</pitch>
<rate factor="1.20">faster</rate>
<pause act="ins"/>
~~~

becomes:

~~~xml
<pitch, semitones=4>higher</pitch>
<rate, factor=1.20>faster</rate>
<pause act="ins" level="2"/>
~~~

Numeric spellings such as <code>1.20</code> are preserved. The public runtime
receives explicit source and target transcripts prefixed with <code>[EN]</code>
or <code>[ZH]</code>. Plain benchmark transcript rendering does not add those
tokens.

## Compiler behavior

<code>one_take</code> sends the complete instruction to one model call.
<code>sequential</code> applies text tags first, then emotion, prosody, and pause
tags, with each output becoming the next input. Both modes preserve local XML
spans and pause boundaries in the model instruction.

The fixed pause strength is part of this candidate adapter rather than the
benchmark XML. Generation uses the settings in
<code>candidate.options.generation_options</code>; the released defaults select
x-vector use automatically and fix the ODE method, step count, guidance scale,
and speaker scale.

## Setup and run

Install doteBench and the dots.tts SDK in one external environment, prepare the
checkpoint, and set:

| Variable | Purpose |
|---|---|
| <code>DOTEBENCH_CANDIDATE_PYTHON</code> | Python interpreter containing the dots.tts runtime |
| <code>DOTEBENCH_MODEL_ROOT</code> | Parent directory containing <code>dots.tts.edit</code> |
| <code>DOTEBENCH_SERVICE_OUTPUTS</code> | Service logs and temporary runtime files |

~~~bash
uv run --frozen dotebench generate \
  --candidate dots-tts-edit \
  --compiler one_take \
  --config configs/dots_tts_edit.yaml \
  --data-root /path/to/materialized-data \
  --audio-root /path/to/materialized-data \
  --run /path/to/runs/dots-tts-edit
~~~

The managed service exposes <code>/health</code> and <code>/invoke</code>.
Candidate failures remain benchmark outcomes; transport, startup, malformed
response, and unavailable-runtime errors leave the run incomplete.

## Tests

The public adapter contract is weight-free:

~~~bash
uv run --frozen pytest tests/candidates/dots_tts_edit
~~~
