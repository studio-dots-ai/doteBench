"""Exercise service cache initialization independently of checkpoint loading."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
RotaryPositionalEmbeddings = pytest.importorskip(
    "torchtune.modules"
).RotaryPositionalEmbeddings


def load_initializer():
    path = (
        Path(__file__).resolve().parents[3] / "src/dotebench/candidates/ming_uniaudio/backend.py"
    )
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "extend_audio_rope_cache"
    )
    namespace = {}
    # Load the repository initializer without activating a checkpoint runtime.
    exec(  # noqa: S102
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace["extend_audio_rope_cache"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_audio_rope_extension_preserves_prefix_and_rotation(dtype):
    rope = RotaryPositionalEmbeddings(dim=64).to(dtype=dtype)
    old = rope.cache.clone()
    x = torch.randn(1, 5, 2, 64).to(dtype)
    positions = torch.tensor([[0, 33, 255, 1024, 4095]])
    before = rope(x, input_pos=positions)
    model = SimpleNamespace(
        audio=SimpleNamespace(
            decoder=SimpleNamespace(semantic_model=torch.nn.Sequential(rope))
        )
    )
    extend = load_initializer()
    extend(model)
    assert rope.cache.shape[0] == 8192
    assert torch.equal(rope.cache[:4096], old)
    assert torch.equal(rope(x, input_pos=positions), before)
    fresh = RotaryPositionalEmbeddings(dim=64, max_seq_len=8192).to(dtype=dtype)
    assert torch.equal(rope.cache[4096:], fresh.cache[4096:])
    result = rope(x, input_pos=torch.arange(4095, 4100).reshape(1, -1))
    assert torch.isfinite(result).all()
    extended = rope.cache
    extend(model)
    assert rope.cache is extended
    assert "cache" not in rope.state_dict()
