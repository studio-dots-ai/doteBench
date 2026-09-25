"""Default model bindings for evaluator capabilities."""

from pathlib import Path

from dotebench.dataset import read_json, relative_path, resources
from dotebench.services.utils import model_assets

MODELS = {
    "qwen3_asr": "qwen3_asr_1_7b",
    "qwen3_aligner": "qwen3_forced_aligner_0_6b",
    "emotion": "qwen3_omni_30b_a3b_instruct",
    "utmos": "utmos",
    "speaker_similarity": "wavlm_sv",
}


def validate_service_model(ability, identity):
    """Validate neural identities; CPU capabilities do not load model assets."""
    if ability in MODELS:
        model_assets.validate_identity(
            MODELS[ability], identity, lock=model_locks()[MODELS[ability]]
        )
    elif identity is not None:
        raise ValueError(f"Unexpected model identity for CPU ability: {ability}")


def model_locks():
    """Pinned model policy for formal evaluation."""
    return read_json(resources() / "release/evaluation-models.json")["models"]


def model_directory(root, model_id):
    return relative_path(Path(root), model_locks()[model_id]["directory"])


def prepare_model(model_id, root):
    return model_assets.prepare_model(model_id, root, lock=model_locks()[model_id])


def prepared_identity(model_id, model_path, *, full_check=False):
    return model_assets.prepared_identity(
        model_id, model_path, lock=model_locks()[model_id], full_check=full_check
    )
