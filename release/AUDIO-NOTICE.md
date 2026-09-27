# doteBench audio bundle notice

This bundle contains 1,026 content-addressed WAV files used to materialize the
doteBench data layer. Per-file source identifiers, license evidence, case
membership, and transformation provenance are recorded in
`audio-provenance.jsonl`.

The recorded material comes from LibriTTS-R, FLEURS, Multilingual LibriSpeech,
AISHELL-3, and AISHELL-1. LibriTTS-R, FLEURS, and Multilingual LibriSpeech are
provided under CC BY 4.0. AISHELL-3 and AISHELL-1 are provided under Apache 2.0.
The bundle also contains reference-free speech generated with
Qwen3-TTS-12Hz-1.7B-VoiceDesign, whose model is published under Apache 2.0.

Source and license evidence:

- LibriTTS-R: https://openslr.org/141/
- FLEURS: https://huggingface.co/datasets/google/fleurs
- Multilingual LibriSpeech: https://openslr.org/94/
- AISHELL-3: https://openslr.org/93/
- AISHELL-1: https://openslr.org/33/
- Qwen3-TTS-12Hz-1.7B-VoiceDesign: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign

The bundle intentionally does not contain Common Voice 20.0 clips, outputs made
from real-person voice-cloning prompts, or Qwen3-TTS preset-speaker outputs.
Those assets are acquired or generated during materialization from recipes in
the benchmark manifests.

The included license texts apply to their corresponding third-party material.
They do not replace the license for doteBench source code or benchmark metadata.
Transcript, XML, and annotation terms are recorded separately in
`DATA-NOTICE.md` and `text-provenance.jsonl`.
