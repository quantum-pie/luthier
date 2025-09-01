import sys
import torch
import torch.nn as nn
import pytest

from src.composition.model.input_embeddings import InputEmbeddings
from src.composition.midi.tokenizer import DRUMS_PROGRAM_ID


def zero_linear(m: nn.Linear):
    nn.init.zeros_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def test_get_instrument_embedding_shapes_and_values():
    hidden_dim = 8
    num_instruments = 5

    emb = InputEmbeddings(
        hidden_dim=hidden_dim,
        pitch_vocab_size=16,
        velocity_vocab_size=8,
        num_instruments=num_instruments,
    )

    # Set instrument embedding weights to a known pattern
    with torch.no_grad():
        for i in range(num_instruments):
            emb.instrument_embedding.weight[i].fill_(float(i))

    program_ids = torch.tensor([[0, 1, 2], [3, 4, 0]], dtype=torch.long)
    inst_embeds = emb.get_instrument_embedding(program_ids)

    assert inst_embeds.shape == (*program_ids.shape, hidden_dim)

    # All channels should equal the instrument id (since we filled with scalar)
    for i in range(program_ids.size(0)):
        for j in range(program_ids.size(1)):
            expected = float(program_ids[i, j].item())
            assert torch.allclose(inst_embeds[i, j], torch.full((hidden_dim,), expected))


def test_forward_masks_and_gm_vs_drums_switch():
    hidden_dim = 4
    pitch_vocab_size = 12
    velocity_vocab_size = 6
    num_instruments = int(DRUMS_PROGRAM_ID) + 1

    model = InputEmbeddings(
        hidden_dim=hidden_dim,
        pitch_vocab_size=pitch_vocab_size,
        velocity_vocab_size=velocity_vocab_size,
        num_instruments=num_instruments,
    )

    # Make embeddings isolate GM vs Drums selection and masking effects:
    with torch.no_grad():
        # GM contributes +1 per channel
        model.gm_embedding.weight.fill_(1.0)

        # Drums contributes +2 per channel
        model.drums_embedding.weight.fill_(2.0)

        # Remove other contributions
        model.velocity_embedding.weight.zero_()
        model.instrument_embedding.weight.zero_()

        # Zero duration projection
        for layer in model.duration_log1p_proj:
            if isinstance(layer, nn.Linear):
                zero_linear(layer)

    # Build a tiny batch: B=1, P=2 (gm and drums), T=3
    B, P, T = 1, 2, 3

    # Program 0 -> GM, Program 1 -> DRUMS
    program_ids = torch.tensor([[0, DRUMS_PROGRAM_ID]])
    pitch_tokens = torch.zeros(B, P, T, dtype=torch.long)
    velocity_tokens = torch.zeros(B, P, T, dtype=torch.long)
    note_durations_beats = torch.zeros(B, P, T)
    attention_mask = torch.tensor([[[1, 0, 1], [1, 1, 0]]], dtype=torch.bool)

    inputs = {
        "program_ids": program_ids,
        "pitch_tokens": pitch_tokens,
        "velocity_tokens": velocity_tokens,
        "note_durations_beats": note_durations_beats,
        "attention_mask": attention_mask,
    }

    out = model(inputs)
    assert out.shape == (B, P, T, hidden_dim)

    # Masking: positions with attention 0 must be exactly zero
    assert torch.all(out[0, 0, 1] == 0)  # GM track, middle token masked
    assert torch.all(out[0, 1, 2] == 0)  # Drums track, last token masked

    # Active positions: GM should be +1 per channel, Drums +2 per channel
    assert torch.allclose(out[0, 0, 0], torch.ones(hidden_dim))
    assert torch.allclose(out[0, 0, 2], torch.ones(hidden_dim))
    assert torch.allclose(out[0, 1, 0], torch.full((hidden_dim,), 2.0))
    assert torch.allclose(out[0, 1, 1], torch.full((hidden_dim,), 2.0))


if __name__ == "__main__":
    sys.exit(pytest.main())
