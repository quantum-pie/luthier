import sys
import pytest
from src.composition.training.losses.instrument_counts_loss import (
    generate_instance_mask_from_ground_truth,
    generate_instance_mask_from_logits,
    generate_instrument_counts_targets,
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


def test_generate_instance_mask_from_ground_truth():
    # Test case 1: Basic case with one track per program
    track_mask = torch.tensor([[True, False, True], [False, True, True]])
    program_ids = torch.tensor([[0, 0, 1], [1, 2, 2]])
    num_programs = 3
    max_instances = 2

    expected_mask = torch.tensor(
        [
            [[True, False], [True, False], [False, False]],
            [[False, False], [False, False], [True, True]],
        ]
    )

    mask = generate_instance_mask_from_ground_truth(
        track_mask, program_ids, num_programs, max_instances
    )

    assert torch.equal(mask, expected_mask)

    # Test case 2: No tracks
    track_mask = torch.tensor([[False, False], [False, False]])
    program_ids = torch.tensor([[0, 1], [1, 2]])
    num_programs = 3
    max_instances = 2
    expected_mask = torch.tensor(
        [
            [[False, False], [False, False], [False, False]],
            [[False, False], [False, False], [False, False]],
        ]
    )

    mask = generate_instance_mask_from_ground_truth(
        track_mask, program_ids, num_programs, max_instances
    )

    assert torch.equal(mask, expected_mask)

    # Test case 3: All tracks for one program
    track_mask = torch.tensor([[True], [True]])
    program_ids = torch.tensor([[1], [1]])
    num_programs = 3
    max_instances = 2

    expected_mask = torch.tensor(
        [
            [[False, False], [True, False], [False, False]],
            [[False, False], [True, False], [False, False]],
        ]
    )

    mask = generate_instance_mask_from_ground_truth(
        track_mask, program_ids, num_programs, max_instances
    )

    assert torch.equal(mask, expected_mask)


def test_generate_instance_mask_from_logits():
    # Test case 1: Basic case with one track per program
    logits = torch.tensor(
        [
            [[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.0, 4.0, 0.0]],
            [[0.0, 0.9, 0.1], [0.7, 0.3, 0.0], [0.0, 0.0, 0.9]],
        ]
    )  # Logits for 2 batches, 3 programs, 2 instances each
    num_programs = 3
    max_instances = 2

    expected_mask = torch.tensor(
        [
            [[False, False], [False, False], [True, False]],
            [[True, False], [False, False], [True, True]],
        ]
    )  # Expected mask for instances
    mask = generate_instance_mask_from_logits(logits)

    assert torch.equal(mask, expected_mask)

    # Test case 2: No tracks
    logits = torch.zeros((2, num_programs, max_instances + 1))
    logits[:, :, 0] = 1.0  # All logits are the same, no instances
    expected_mask = torch.zeros((2, num_programs, max_instances), dtype=torch.bool)

    mask = generate_instance_mask_from_logits(logits)

    assert torch.equal(mask, expected_mask)


if __name__ == "__main__":
    sys.exit(pytest.main())
