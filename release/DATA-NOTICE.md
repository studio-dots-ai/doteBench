# doteBench data notice

The doteBench dataset contains 2,081 XML editing cases. Each XML fragment combines an
upstream source transcript with doteBench-authored edit markup, edit content,
and evaluation annotations. The repository's Apache-2.0 license covers only the
doteBench-authored portions.

`text-provenance.jsonl` records the origin and SHA-256 identity of every source
and target transcript without duplicating the transcript text. It also records
the license of edit markup, edit content, and annotations. The ledger is built
deterministically with:

```bash
PYTHONPATH=src python scripts/build_text_provenance.py --check
```

## Transcript sources

| Source | Cases | Terms | Evidence |
|---|---:|---|---|
| LibriTTS-R | 819 | CC BY 4.0 | https://www.openslr.org/141/ |
| Common Voice 20.0 | 354 | CC0 1.0 | https://commonvoice.mozilla.org/en/datasets |
| doteBench-authored/generated source text | 303 | Apache 2.0 | `../LICENSE` |
| WenetSpeech | 234 | CC BY 4.0; official site limits downloads to non-commercial purposes | https://wenet-e2e.github.io/WenetSpeech/ |
| AISHELL-3 | 222 | Apache 2.0 | https://www.openslr.org/93/ |
| Emo-Emilia | 57 | CC BY-NC 4.0 | https://huggingface.co/datasets/ASLP-lab/Emo-Emilia |
| Multilingual LibriSpeech | 49 | CC BY 4.0 | https://www.openslr.org/94/ |
| FLEURS | 42 | CC BY 4.0 | https://huggingface.co/datasets/google/fleurs |
| AISHELL-1 | 1 | Apache 2.0 | https://www.openslr.org/33/ |

Counts refer to benchmark cases and sum to 2,081. Several cases may share one
source transcript. The WenetSpeech and Emo-Emilia portions remain subject to
non-commercial conditions; cloning this repository does not remove those
conditions. CC BY sources require attribution. The corresponding license texts
are included under `release/licenses/`.

Source alignments are doteBench evaluation annotations over the attributed
transcripts. XML tags, operation parameters, grouping labels, and text inserted
or substituted by doteBench are Apache-2.0 contributions. Unchanged text inside
source and target projections retains its upstream transcript terms.

Audio rights and acquisition boundaries are separate. See `AUDIO-NOTICE.md` and
`audio-provenance.jsonl`; the complete benchmark audio set is materialized
locally rather than distributed as one repository bundle.
