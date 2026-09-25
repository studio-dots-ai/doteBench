"""Built-in candidate entrypoints and capabilities, including bootstrap tools."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateSpec:
    adapter: str
    compilers: tuple[str, ...] = ("one_take", "sequential")
    native_runtime: bool = True


CANDIDATES = {
    "identity": CandidateSpec("Identity", ("one_take",), False),
    "auk_base": CandidateSpec("AuKBase"),
    "dots_tts_edit": CandidateSpec("DotsTtsEdit", native_runtime=False),
    "ming_uniaudio": CandidateSpec("MingUniAudio"),
    "step_audio_editx": CandidateSpec("StepAudioEditX"),
    "mimo_audio_instruct": CandidateSpec("MiMoAudioInstruct"),
    "fireredtts3_instruct": CandidateSpec("FireRedTTS3Instruct"),
}


def native_candidates():
    return tuple(
        name.replace("_", "-")
        for name, spec in CANDIDATES.items()
        if spec.native_runtime
    )
