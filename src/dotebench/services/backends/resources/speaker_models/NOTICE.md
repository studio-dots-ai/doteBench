# Speaker model source attribution

`WavLM.py` and `modules.py` are adapted from Microsoft's WavLM source in
[UniLM](https://github.com/microsoft/unilm/tree/master/wavlm). They retain the
Microsoft copyright headers and the [MIT license](LICENSE-WavLM). The WavLM
implementation incorporates fairseq code; its [MIT notice](LICENSE-fairseq)
is included here.

`ecapa_tdnn.py` is adapted from the speaker verification implementation in
[Microsoft UniSpeech](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification),
whose [Attribution-ShareAlike 3.0 Unported license](LICENSE-UniSpeech) is retained.
Its upstream attribution to [lawlict/ECAPA-TDNN](https://github.com/lawlict/ECAPA-TDNN)
is preserved in the file header.

Changes to the speaker implementation include a local WavLM feature wrapper,
checkpoint-based feature initialization, and package-relative imports. Source
comments describe these adaptations. Each notice applies to the associated
third-party source and adaptations. Model checkpoint terms are distributed with
the checkpoints by their respective providers.
