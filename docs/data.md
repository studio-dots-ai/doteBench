# Data contract

This page defines the stable manifest and XML interfaces. See
[Using doteBench](usage.md) for the runnable workflow and
[Data materialization](materialization.md) for constructing the audio root.

`release/shards.json` declares the V1.0 dataset version (`v1_0`) once and lists category/shard
pairs in release order. The loader derives each `data/<category>/<shard>/manifest.json`
path and case count from this index. Audio paths and checksums come from cases.

Every manifest has `category`, `shard`, `audio_assets`, and `cases`.
Public shard names are:

| Category | Shards | Cases |
|---|---|---:|
| text | easy, hard | 209, 360 |
| emotion | neutral, intense | 312, 300 |
| prosody | default | 360 |
| pause | default | 300 |
| compositional | default | 240 |

Every case has exactly these fields:

| Field | Meaning |
|---|---|
| `id` | Stable, globally unique case identifier |
| `language` | `en` or `zh` |
| `source_audio` | Stable `asset_id`, materialized relative `path`, and actual `sha256` |
| `instruction_xml` | Authoritative edit instruction |
| `annotations` | Typed evaluation references and authored grouping labels |

The loader checks field names, release path identity, unique IDs, XML semantics, and audio
checksums. Source and target transcripts are derived by `parse_instruction(xml)`;
`Case.source_text` and `Case.target_text` expose these read-only projections.
`annotations.source_alignment` stores each sample's ordered source word alignment.
It contains `audio_sha256`, `text_sha256` (UTF-8), and
`segments`, each with `word`, `start`, and `end` in seconds. The Qwen3 aligner uses
explicit benchmark word units. Predicted intervals are intersected with the actual
audio duration; zero-duration units remain present.

All source operation intervals and pause gaps are extracted from this alignment
using the XML positions. Formal evaluation never aligns source audio. Missing or
incompatible annotations are dataset errors.

Latin alphanumeric runs form
units and Chinese characters form individual units; punctuation separates units.
The complete ordered sequence must match the transcript. Queries resolve positions
against these units, and boundaries must lie between units. Zero-duration words
retain their positions and timestamps.

Prosody cases include `annotations.span_granularity`, one of `single_word`,
`short_phrase`, `clause`, or `sentence`. Emotion intense cases include
`annotations.source_emotion`, one of `afraid`, `angry`, `happy`, `melancholic`,
`sad`, or `surprised`. These authored labels support subgroup analysis and do
not change sample weights. Candidate requests do not contain annotations.

## Instruction grammar

```xml
<ins>inserted text</ins>
<del>deleted text</del>
<sub targ="replacement">original</sub>
<emo type="happy" level="2">affected text</emo>
<pitch semitones="3">affected text</pitch>
<rate factor="1.2">affected text</rate>
<pause act="ins"/>
```

Fragments may combine several operations in source order. Literal XML characters
use entity escaping. Operations cannot nest or cross. Pause supports `ins` and
`red`, specifying a direction at a text boundary. Emotion levels are 1–3. Acoustic numeric parameters must
be finite and rate factors positive.

`parse_instruction()` returns operations with half-open Unicode character spans
on source and target text. Insertions have zero-width source spans, deletions have
zero-width target spans, and pause operations mark text boundaries. Text edits
normalize word-boundary spacing and punctuation in the target projection;
attribute-only edits preserve transcription whitespace.

Candidates receive XML and may call the standard parser/renderer. Model-specific
prompts, strength choices and auxiliary inference belong to the candidate system.
Frozen source alignments are evaluation references and must not be used by candidates.

`audio_assets` is consumed only by the materialization layer. It records whether
each stable asset is imported from the release bundle, acquired from an upstream
dataset, or synthesized by a named provider. Once materialization writes the
final paths, checksums, and any updated source alignments, candidate generation
and evaluation do not inspect these recipes. See [data materialization](materialization.md).
