"""AuK Base compiler requirements bind to the current audio."""

from dotebench.candidates.auk_base.compilers.one_take import OneTakeCompiler
from tests.candidates.helpers import audio_request, wav


def test_alignment_requirement_uses_current_audio():
    compiler = OneTakeCompiler()
    original = audio_request('left <pitch semitones="2">middle</pitch> right')
    (step,) = compiler.compile(original).steps
    (requirement,) = step.requirements
    current = wav(0.5)
    seen = []

    class Candidate:
        def resolve_alignment(self, audio, text, language):
            seen.append((audio, text, language))
            return "alignment", {"cache_key": "current"}

    assert requirement.resolve(Candidate(), current) == (
        "alignment",
        {"cache_key": "current"},
    )
    assert seen == [(current, "left middle right", "en")]


def test_cli_wiring(monkeypatch, tmp_path):
    from tests.candidates.helpers import assert_cli_wiring

    assert_cli_wiring(monkeypatch, tmp_path, "auk_base", has_aligner=True)
