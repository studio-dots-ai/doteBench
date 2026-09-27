"""Verify the real service splice with a controlled model and VAE."""

from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from dotebench.candidates.auk_base import backend as service


class VAE:
    def encoding_and_normalization(self, source):
        n = source.shape[-1] // 480
        return torch.zeros(1, n, 2), torch.tensor([n])

    def denormalize(self, latents):
        return latents

    def inference_from_latents(self, latents):
        return torch.full((1, 1, latents.shape[-1] * 480), 0.25)


class Sampler:
    text_processor = None
    calls = 0

    def build_cond_inputs(self, messages, processor):
        assert len(messages[0][0]["content"]) == 2
        return messages

    def sample(self, *, cond, duration, known_target=None, edit_mask=None, **kwargs):
        self.calls += 1
        if known_target is None:
            known_target = torch.zeros(1, duration - cond.shape[1], cond.shape[-1])
            edit_mask = torch.ones(known_target.shape[:2], dtype=torch.bool)
        assert duration == cond.shape[1] + known_target.shape[1]
        result = known_target.clone()
        result[edit_mask] = 0.25
        return torch.cat((cond, result), 1), None


@pytest.mark.parametrize("rate", [16000, 22050, 24000, 32000, 44100, 48000])
@pytest.mark.parametrize("mode", ["clean", "bridge"])
def test_original_rate_context_and_one_call(rate, mode):
    source = torch.linspace(-0.4, 0.4, round(rate * 1.013)).reshape(1, -1)
    sampler = Sampler()
    engine = SimpleNamespace(
        target_sample_rate=24000,
        downsample_rate=480,
        is_flash=False,
        device="cpu",
        dtype=torch.float32,
        latent_dim=2,
        vae_model=VAE(),
        model=sampler,
    )
    req = service.LocalRequest(
        audio="unused",
        instruction="Change the emotion to happy.",
        start_frame=10,
        end_frame=20,
        target_frames=7,
        context_mode=mode,
        metadata={},
    )
    result, meta = service.generate_local(engine, source, rate, req)
    a, b, n = round(0.2 * rate), round(0.4 * rate), round(0.14 * rate)
    assert sampler.calls == 1
    assert torch.equal(result[:, :a], source[:, :a])
    assert torch.equal(result[:, a + n :], source[:, b:])
    assert result.shape[-1] == source.shape[-1] - (b - a) + n
    assert meta["context_latents_equal"]
    assert meta["sampler_calls"] == 1


def test_service_health_without_model(monkeypatch):
    monkeypatch.setattr(service, "engine", None)
    with TestClient(service.app) as client:
        health = client.get("/health").json()
        assert health["model_loaded"] is False
        assert health["model"] == "AuK Base"


@pytest.mark.parametrize("rate", [16000, 22050, 24000, 32000, 44100, 48000])
def test_whole_target_mask_and_length(rate):
    import math

    class WholeSampler(Sampler):
        def sample(self, **kwargs):
            assert "edit_mask" not in kwargs and "known_target" not in kwargs
            return super().sample(**kwargs)

    source = torch.linspace(-0.4, 0.4, round(rate * 1.013)).reshape(1, -1)
    results = []
    for mode in ["clean", "bridge"]:
        sampler = WholeSampler()
        engine = SimpleNamespace(
            target_sample_rate=24000,
            downsample_rate=480,
            is_flash=False,
            device="cpu",
            dtype=torch.float32,
            latent_dim=2,
            vae_model=VAE(),
            model=sampler,
        )
        req = service.LocalRequest(
            audio="unused",
            instruction="Change the spoken content to: hello",
            start_frame=0,
            end_frame=math.ceil(source.shape[-1] * 50 / rate),
            target_frames=1,
            context_mode=mode,
            metadata={"generation_scope": "whole"},
        )
        result, _meta = service.generate_local(engine, source, rate, req)
        assert result.shape[-1] == round(rate / 50)
        assert torch.isfinite(result).all() and sampler.calls == 1
        results.append(result)
    assert torch.equal(*results)
