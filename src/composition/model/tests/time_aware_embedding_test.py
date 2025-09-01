import math
import sys
import torch
import pytest

from src.composition.model.time_aware_embedding import (
    apply_rotary,
    RotaryEmbedding,
    ExponentialRotaryEmbedding,
    LinearRotaryEmbedding,
    DualRotartyEmbedding,
)


def test_apply_rotary_small_example():
    # x shape: [B,S,D] with D even
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, -2.5]]])  # [1,2,4]

    # sin/cos over dim//2 = 2
    sin = torch.tensor([[[0.0, 1.0], [0.5, -0.5]]])  # [1,2,2]
    cos = torch.tensor([[[1.0, 0.0], [math.sqrt(3) / 2, math.sqrt(3) / 2]]])

    y = apply_rotary(x, sin, cos)
    assert y.shape == x.shape

    # Manual compute for first token (1,2) and (3,4)
    even_0, odd_0 = 1.0, 2.0
    sin_0, cos_0 = 0.0, 1.0
    even_0_p = even_0 * cos_0 - odd_0 * sin_0
    odd_0_p = odd_0 * cos_0 + even_0 * sin_0

    even_1, odd_1 = 3.0, 4.0
    sin_1, cos_1 = 1.0, 0.0
    even_1_p = even_1 * cos_1 - odd_1 * sin_1
    odd_1_p = odd_1 * cos_1 + even_1 * sin_1

    exp_0 = torch.tensor([even_0_p, odd_0_p, even_1_p, odd_1_p])
    assert torch.allclose(y[0, 0], exp_0, atol=1e-6)

    # Second token
    even_0, odd_0 = 0.5, -1.0
    sin_0, cos_0 = 0.5, math.sqrt(3) / 2
    even_0_p = even_0 * cos_0 - odd_0 * sin_0
    odd_0_p = odd_0 * cos_0 + even_0 * sin_0

    even_1, odd_1 = 2.0, -2.5
    sin_1, cos_1 = -0.5, math.sqrt(3) / 2
    even_1_p = even_1 * cos_1 - odd_1 * sin_1
    odd_1_p = odd_1 * cos_1 + even_1 * sin_1

    exp_1 = torch.tensor([even_0_p, odd_0_p, even_1_p, odd_1_p])
    assert torch.allclose(y[0, 1], exp_1, atol=1e-6)


def test_rotary_embedding_forward_shapes_and_values():
    freqs = torch.tensor([1.0, 2.0])  # dim//2 = 2
    emb = RotaryEmbedding(freqs)
    pos = torch.tensor([[0.0, 0.25, 0.5]])  # [1,3]
    sin, cos = emb(pos)
    assert sin.shape == (1, 3, 2)
    assert cos.shape == (1, 3, 2)

    # angles = pos * freqs
    angles = torch.einsum("bs,d->bsd", pos, freqs)
    assert torch.allclose(sin, torch.sin(angles))
    assert torch.allclose(cos, torch.cos(angles))


def test_exponential_rotary_frequencies():
    dim = 8
    base = 10.0
    emb = ExponentialRotaryEmbedding(dim=dim, base=base)
    out = emb.get_frequencies()

    expected = 1.0 / (base ** (torch.arange(0, dim // 2).float() / (dim // 2)))
    assert torch.allclose(out, expected)


def test_linear_rotary_frequencies():
    dim = 8
    emb = LinearRotaryEmbedding(dim=dim)
    out = emb.get_frequencies()
    expected = torch.arange(1, dim // 2 + 1).float() * 2 * math.pi
    assert torch.allclose(out, expected)


def test_dual_rotarty_embedding_concat_and_getters():
    dim = 12  # divisible by 4
    base = 1000.0
    dual = DualRotartyEmbedding(dim=dim, base=base)

    B, S = 2, 5
    pos_lin = torch.rand(B, S)
    pos_exp = torch.rand(B, S)
    sin, cos = dual(pos_lin, pos_exp)

    assert sin.shape == (B, S, dim // 2)
    assert cos.shape == (B, S, dim // 2)

    # Check halves correspond to components
    lin_sin, lin_cos = dual.linear_emb(pos_lin)
    exp_sin, exp_cos = dual.exponential_emb(pos_exp)
    assert torch.allclose(sin[..., : dim // 4], lin_sin)
    assert torch.allclose(cos[..., : dim // 4], lin_cos)
    assert torch.allclose(sin[..., dim // 4 :], exp_sin)
    assert torch.allclose(cos[..., dim // 4 :], exp_cos)

    # Getter passthrough
    assert torch.allclose(dual.get_linear_frequencies(), dual.linear_emb.get_frequencies())
    assert torch.allclose(dual.get_exponential_frequencies(), dual.exponential_emb.get_frequencies())


if __name__ == "__main__":
    sys.exit(pytest.main())
