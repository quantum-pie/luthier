import sys
import pytest
from src.composition.training.losses.instrument_counts_loss import (
    generate_instrument_counts_targets,
    instrument_counts_loss,
)
import torch


def test_generate_instrument_counts_targets():
    # Test case 1: Basic case with one track per program
    track_mask = torch.tensor([[True, False, True], [False, True, True]])
    program_ids = torch.tensor([[0, 0, 1], [1, 2, 2]])
    num_programs = 3

    expected_counts = torch.tensor([[1, 1, 0], [0, 0, 2]])
    counts = generate_instrument_counts_targets(track_mask, program_ids, num_programs)

    assert torch.equal(counts, expected_counts)

    # Test case 2: No tracks
    track_mask = torch.tensor([[False, False], [False, False]])
    program_ids = torch.tensor([[0, 1], [1, 2]])

    expected_counts = torch.tensor([[0, 0, 0], [0, 0, 0]])
    counts = generate_instrument_counts_targets(track_mask, program_ids, num_programs)

    assert torch.equal(counts, expected_counts)

    # Test case 3: All tracks for one program
    track_mask = torch.tensor([[True], [True]])
    program_ids = torch.tensor([[1], [1]])

    expected_counts = torch.tensor([[0, 1, 0], [0, 1, 0]])
    counts = generate_instrument_counts_targets(track_mask, program_ids, num_programs)

    assert torch.equal(counts, expected_counts)


def test_instrument_count_loss():
    # Test case 1: Perfect prediction
    rates = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    target_counts = torch.tensor([[2, 3], [4, 5]])

    min_loss = instrument_counts_loss(rates, target_counts)
    assert min_loss.shape == (2, 2)

    # Test case 2: Smaller rates
    rates = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    loss = instrument_counts_loss(rates, target_counts)
    assert loss.shape == (2, 2)
    assert torch.all(loss > min_loss)  # should be higher than perfect prediction

    # Test case 3: Higher rates
    rates = torch.tensor([[3.0, 4.0], [5.0, 6.0]])

    loss = instrument_counts_loss(rates, target_counts)
    assert loss.shape == (2, 2)
    assert torch.all(loss > min_loss)  # should be higher than perfect prediction


if __name__ == "__main__":
    sys.exit(pytest.main())
