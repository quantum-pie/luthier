import sys
import torch
import pytest

from src.composition.model.conductor_model import Conductor


def make_dummy_batch(B=2, P=3, T=4, L=6, hidden_dim=32):
    # Random input token embeddings [B,P,T,H]
    input_embeddings = torch.randn(B, P, T, hidden_dim)

    # Attention over tokens [B,P,T]
    attention_mask = torch.ones(B, P, T, dtype=torch.bool)

    # Bar positions/tempos [B,L]
    bar_tempos = torch.rand(B, L)
    bar_boundaries = torch.arange(L, dtype=torch.float32).unsqueeze(0).repeat(B, 1)

    model_inputs = {
        "attention_mask": attention_mask,
        "bar_tempos": bar_tempos,
        "bar_boundaries": bar_boundaries,
    }

    return model_inputs, input_embeddings


def test_conductor_forward_shapes_and_keys():
    latent_dim = 16
    control_embed_dim = 8
    hidden_dim = 32
    num_instruments = 11
    max_instances = 4

    model = Conductor(
        latent_dim=latent_dim,
        control_embed_dim=control_embed_dim,
        hidden_dim=hidden_dim,
        genre_vocab_size=7,
        mood_vocab_size=5,
        num_instruments=num_instruments,
        max_instrument_instances=max_instances,
    )

    B, P, T, L = 2, 3, 4, 6
    model_inputs, input_embeddings = make_dummy_batch(B, P, T, L, hidden_dim)

    out = model(model_inputs, input_embeddings)

    assert set(out.keys()) == {
        "tempos",
        "instrument_counts_rates",
        "instrument_density_logits",
        "mu_posterior",
        "logvar_posterior",
        "mu_prior",
        "logvar_prior",
    }

    assert out["tempos"].shape == (B, L)
    assert out["instrument_counts_rates"].shape == (B, num_instruments)
    assert out["instrument_density_logits"].shape == (
        B,
        L,
        num_instruments,
    )
    for k in ["mu_posterior", "logvar_posterior", "mu_prior", "logvar_prior"]:
        assert out[k].shape == (B, latent_dim)
        assert not torch.isnan(out[k]).any()


def test_conductor_forward_step_shapes_and_state():
    latent_dim = 16
    control_embed_dim = 8
    hidden_dim = 32
    num_instruments = 9
    max_instances = 3

    model = Conductor(
        latent_dim=latent_dim,
        control_embed_dim=control_embed_dim,
        hidden_dim=hidden_dim,
        genre_vocab_size=4,
        mood_vocab_size=6,
        num_instruments=num_instruments,
        max_instrument_instances=max_instances,
    )

    # First step initializes internal interpolator
    out0 = model.forward_step(torch.tensor(0.0))
    assert set(out0.keys()) == {
        "z",
        "tempo",
        "instrument_counts_rates",
        "instrument_density_logits",
    }
    assert out0["z"].shape == (latent_dim,)
    assert out0["tempo"].shape == (1,)
    assert out0["instrument_counts_rates"].shape == (num_instruments,)
    assert out0["instrument_density_logits"].shape == (num_instruments,)

    # Subsequent step should reuse state and produce same shapes
    out1 = model.forward_step(torch.tensor(1.0))
    assert out1["z"].shape == (latent_dim,)
    assert out1["instrument_density_logits"].shape == (num_instruments,)


def test_control_hash_order_invariance_and_none():
    model = Conductor(
        latent_dim=8,
        control_embed_dim=4,
        hidden_dim=16,
        genre_vocab_size=10,
        mood_vocab_size=10,
        num_instruments=5,
        max_instrument_instances=2,
    )

    # None input -> None hash
    assert model.control_hash(None) is None

    # Order invariance (sorting inside control_hash)
    ct1 = {
        "genre_ids": torch.tensor([5, 1, 3], dtype=torch.long),
        "mood_ids": torch.tensor([2, 9], dtype=torch.long),
    }
    ct2 = {
        "genre_ids": torch.tensor([3, 5, 1], dtype=torch.long),
        "mood_ids": torch.tensor([9, 2], dtype=torch.long),
    }
    h1 = model.control_hash(ct1)
    h2 = model.control_hash(ct2)
    assert isinstance(h1, str) and isinstance(h2, str)
    assert h1 == h2

    # Duplicates change payload length => different hash
    ct3 = {
        "genre_ids": torch.tensor([5, 1, 3, 3], dtype=torch.long),  # extra duplicate
        "mood_ids": torch.tensor([2, 9], dtype=torch.long),
    }
    h3 = model.control_hash(ct3)
    assert h3 != h1


def test_forward_step_z_constant_without_control_change_and_transitions_when_changed():
    torch.manual_seed(0)

    latent_dim = 12
    model = Conductor(
        latent_dim=latent_dim,
        control_embed_dim=6,
        hidden_dim=24,
        genre_vocab_size=8,
        mood_vocab_size=8,
        num_instruments=6,
        max_instrument_instances=2,
    )

    # Initial control A
    ctrl_a = {
        "genre_ids": torch.tensor([1, 2], dtype=torch.long),
        "mood_ids": torch.tensor([3], dtype=torch.long),
    }

    # Initialize at bar 0
    out0 = model.forward_step(torch.tensor(0.0), control_tokens=ctrl_a)
    z0 = out0["z"].clone()

    # Advance bars with same control: z should remain constant
    out1 = model.forward_step(torch.tensor(1.0), control_tokens=ctrl_a)
    out2 = model.forward_step(torch.tensor(2.0), control_tokens=ctrl_a)
    assert torch.allclose(out1["z"], z0)
    assert torch.allclose(out2["z"], z0)

    # Change control at current bar; seed to make sampling deterministic
    torch.manual_seed(1234)
    ctrl_b = {
        "genre_ids": torch.tensor([4], dtype=torch.long),
        "mood_ids": torch.tensor([7], dtype=torch.long),
    }
    out_change_same_bar = model.forward_step(torch.tensor(2.0), control_tokens=ctrl_b)
    # At the bar where change happens, interpolation alpha=0 -> still old z
    assert torch.allclose(out_change_same_bar["z"], z0)

    # Next bar, z should start moving
    out3 = model.forward_step(torch.tensor(3.0), control_tokens=ctrl_b)
    assert not torch.allclose(out3["z"], z0)

    # After several bars (> transition_duration=4), z stabilizes
    out7 = model.forward_step(torch.tensor(7.0), control_tokens=ctrl_b)
    out8 = model.forward_step(torch.tensor(8.0), control_tokens=ctrl_b)
    assert torch.allclose(out7["z"], out8["z"], atol=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main())
