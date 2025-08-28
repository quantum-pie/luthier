import sys
import torch
import torch.nn as nn
import pytest

from src.composition.model.control_encoder import ControlEncoder


def set_linear_identity(linear: nn.Linear):
    assert linear.in_features == linear.out_features
    with torch.no_grad():
        linear.weight.copy_(torch.eye(linear.in_features))
        if linear.bias is not None:
            linear.bias.zero_()


def test_forward_masked_mean_and_fuser_identity():
    # Choose dims so first Linear can be identity: latent_dim == 2 * control_embedding_dim
    control_embedding_dim = 4
    latent_dim = 8  # == 2*C
    genre_vocab_size = 10
    mood_vocab_size = 12

    enc = ControlEncoder(
        genre_vocab_size=genre_vocab_size,
        mood_vocab_size=mood_vocab_size,
        control_embedding_dim=control_embedding_dim,
        latent_dim=latent_dim,
    )

    # Make fuser an identity mapping with ReLU between them (inputs kept non-negative)
    assert isinstance(enc.fuser[0], nn.Linear) and isinstance(enc.fuser[2], nn.Linear)
    set_linear_identity(enc.fuser[0])
    set_linear_identity(enc.fuser[2])

    # Set embeddings to known non-negative constants per id
    with torch.no_grad():
        for i in range(genre_vocab_size):
            enc.genre_embedding.weight[i].fill_(float(i + 1))  # c_g(id) = id+1
        for j in range(mood_vocab_size):
            enc.mood_embedding.weight[j].fill_(float((j + 1) * 10))  # c_m(id) = 10*(id+1)

    # Build a small batch with masks
    # B=2, Lg=3, Lm=2
    genre_ids = torch.tensor([[1, 3, 5], [0, 0, 9]], dtype=torch.long)
    genre_mask = torch.tensor([[True, False, True], [True, False, True]])

    mood_ids = torch.tensor([[2, 4], [1, 7]], dtype=torch.long)
    mood_mask = torch.tensor([[True, True], [False, True]])

    out = enc(genre_ids, genre_mask, mood_ids, mood_mask)
    assert out.shape == (2, latent_dim)

    # Expected means (embeddings are constant vectors per id)
    # Sample 0: genre mean = mean([2, 6]) = 4.0, mood mean = mean([30, 50]) = 40.0
    # Sample 1: genre mean = mean([1, 10]) = 5.5, mood mean = mean([80]) = 80.0
    expected = torch.tensor([
        [4.0] * control_embedding_dim + [40.0] * control_embedding_dim,
        [5.5] * control_embedding_dim + [80.0] * control_embedding_dim,
    ])

    assert torch.allclose(out, expected, atol=1e-6)


def test_encode_default_uses_fuser_on_default_control():
    control_embedding_dim = 3
    latent_dim = 6  # == 2*C
    enc = ControlEncoder(
        genre_vocab_size=2,
        mood_vocab_size=2,
        control_embedding_dim=control_embedding_dim,
        latent_dim=latent_dim,
    )

    # Make fuser identity
    set_linear_identity(enc.fuser[0])
    set_linear_identity(enc.fuser[2])

    # Set default_control to a known non-negative vector so ReLU is neutral
    with torch.no_grad():
        enc.default_control.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    out = enc.encode_default()
    assert out.shape == (1, latent_dim)
    assert torch.allclose(out.squeeze(0), enc.default_control, atol=1e-6)

if __name__ == "__main__":
    sys.exit(pytest.main())
