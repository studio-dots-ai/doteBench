"""Test target context enforcement with an analytic constant velocity field."""

from types import SimpleNamespace

import pytest
import torch
from auk.model.cfm_edit import CFMEdit


class Field(torch.nn.Module):
    dim = 2

    def __init__(self):
        super().__init__()
        self.calls = []
        self.cleared = False

    def forward(self, x, time, **kwargs):
        self.calls.append((float(time), x.clone()))
        return torch.ones_like(x)

    def clear_cache(self):
        self.cleared = True


def model():
    encoder = torch.nn.Linear(2, 2)
    encoder.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=1))
    m = CFMEdit(Field(), encoder, None, 2)
    m.encode_text = lambda text, device: (
        torch.zeros(1, 1, 2),
        torch.ones(1, 1, dtype=torch.bool),
    )
    return m


@pytest.mark.parametrize("mode", ["clean", "bridge"])
def test_context_and_reference_preserved(mode):
    m = model()
    ref = torch.full((1, 2, 2), 7.0)
    known = torch.full((1, 5, 2), 3.0)
    mask = torch.tensor([[False, True, True, False, False]])
    out, _ = m.sample(
        ref,
        None,
        7,
        steps=4,
        cfg_strength=0,
        seed=123,
        known_target=known,
        edit_mask=mask,
        context_mode=mode,
    )
    assert torch.equal(out[:, :2], ref)
    assert torch.equal(out[:, 2:][~mask], known[~mask])
    torch.manual_seed(123)
    noise = torch.randn(5, 2).unsqueeze(0)
    torch.testing.assert_close(out[:, 2:][mask], (noise + 1)[mask])
    for t, x in m.transformer.calls:
        expected = known if mode == "clean" else (1 - t) * noise + t * known
        torch.testing.assert_close(x[~mask], expected[~mask])
    assert m.transformer.cleared


def test_original_sampling_and_mask_validation():
    m = model()
    ref = torch.zeros(1, 2, 2)
    a, _ = m.sample(ref, None, 7, cfg_strength=0, seed=3)
    b, _ = m.sample(ref, None, 7, cfg_strength=0, seed=3)
    assert torch.equal(a, b)
    with pytest.raises(ValueError, match="together"):
        m.sample(ref, None, 7, known_target=torch.zeros(1, 5, 2))
    with pytest.raises(ValueError, match="shape"):
        m.sample(
            ref,
            None,
            7,
            known_target=torch.zeros(1, 4, 2),
            edit_mask=torch.zeros(1, 4, dtype=torch.bool),
        )


def test_whole_mask_modes_match_original_sampler():
    m = model()
    ref = torch.zeros(1, 2, 2)
    kwargs = {"steps": 4, "cfg_strength": 0, "seed": 123}
    original, _ = m.sample(ref, None, 7, **kwargs)
    for mode in ["clean", "bridge"]:
        actual, _ = m.sample(
            ref,
            None,
            7,
            known_target=torch.zeros(1, 5, 2),
            edit_mask=torch.ones(1, 5, dtype=torch.bool),
            context_mode=mode,
            **kwargs,
        )
        assert torch.equal(actual, original)
