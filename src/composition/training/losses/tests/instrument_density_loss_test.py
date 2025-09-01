import pytest
import sys

from src.composition.training.losses.instrument_density_loss import (
    generate_instrument_density_targets,
    instrument_density_loss,
)
import torch


def test_generate_instrument_density_targets_case_1():
    num_programs = 3
    max_global_len = 3

    track_mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
    program_ids = torch.tensor([[1, 1, 2, 2], [0, 0, 1, -1]])

    instruments_counts = torch.tensor([[0, 2, 2], [2, 1, 0]])

    bar_activations = torch.tensor(
        [
            [[0, 1, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]],
            [[0, 0, 1], [0, 0, 1], [1, 0, 1], [0, 0, 0]],
        ]
    )

    expected_targets = torch.zeros((2, max_global_len, num_programs), dtype=torch.long)
    expected_targets[0] = torch.tensor(
        [
            [0, 0, 1],  # Bar 0
            [0, 2, 0],  # Bar 1
            [0, 1, 1],  # Bar 2
        ]
    )

    expected_targets[1] = torch.tensor(
        [
            [0, 1, 0],  # Bar 0
            [0, 0, 0],  # Bar 1
            [2, 1, 0],  # Bar 2
        ]
    )

    targets = generate_instrument_density_targets(
        bar_activations, track_mask, program_ids, instruments_counts, num_programs
    )

    assert targets.shape == (2, max_global_len, num_programs)
    assert torch.equal(targets, expected_targets), "Generated targets do not match expected values"


def test_generate_instrument_density_targets_case_2():
    num_programs = 4
    max_global_len = 4

    track_mask = torch.tensor([[True, True, True, False], [True, True, True, True]])  # Batch 0  # Batch 1

    program_ids = torch.tensor(
        [
            [0, 1, 0, -1],  # Batch 0: Track 0 & 2 -> program 0, Track 1 -> program 1
            [2, 2, 3, 3],  # Batch 1: Track 0 & 1 -> program 2, Track 2 & 3 -> program 3
        ]
    )

    instrument_counts = torch.tensor([[2, 1, 0, 0], [0, 0, 2, 2]])

    bar_activations = torch.tensor(
        [
            [[1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0]],  # Batch 0
            [[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 1], [0, 0, 0, 1]],  # Batch 1
        ],
        dtype=torch.bool,
    )

    expected_targets = torch.zeros((2, max_global_len, num_programs), dtype=torch.long)

    # --- Batch 0 ---
    expected_targets[0] = torch.tensor(
        [
            [2, 0, 0, 0],  # Bar 0
            [0, 1, 0, 0],  # Bar 1
            [1, 0, 0, 0],  # Bar 2
            [1, 0, 0, 0],  # Bar 3
        ]
    )

    # --- Batch 1 ---
    expected_targets[1] = torch.tensor(
        [
            [0, 0, 1, 1],  # Bar 0
            [0, 0, 2, 0],  # Bar 1
            [0, 0, 1, 1],  # Bar 2
            [0, 0, 0, 2],  # Bar 3
        ]
    )

    # Run
    targets = generate_instrument_density_targets(
        bar_activations,
        track_mask,
        program_ids,
        instrument_counts=instrument_counts,
        num_programs=num_programs,
    )

    # Check
    assert targets.shape == expected_targets.shape
    assert torch.equal(targets, expected_targets), "Test case 2 failed: Targets don't match expected output."


def test_instrument_density_loss():
    num_programs = 2
    max_global_len = 3

    expected_targets = torch.zeros((1, max_global_len, num_programs), dtype=torch.long)

    expected_targets[0, 0, 0] = 3
    expected_targets[0, 1, 0] = 3
    expected_targets[0, 2, 0] = 3

    expected_targets[0, 0, 1] = 1
    expected_targets[0, 1, 1] = 1

    instrument_counts = torch.zeros((1, num_programs), dtype=torch.long)
    instrument_counts[0, 0] = 3
    instrument_counts[0, 1] = 1

    # simple mapping because targets are strictly boolean
    logits = (expected_targets.float() / instrument_counts.unsqueeze(1).float() - 0.5) * 30.0

    loss = instrument_density_loss(logits, expected_targets.float(), instrument_counts).mean()

    assert torch.abs(loss) < 1e-5, "Loss should be close to zero for perfect predictions"

    # less expected density while keeping logits -> higher loss
    expected_targets_lower_density = expected_targets
    expected_targets_lower_density[0, 0, 0] = 1

    loss_lower_density = instrument_density_loss(
        logits, expected_targets_lower_density.float(), instrument_counts
    ).mean()

    assert torch.abs(loss_lower_density) > torch.abs(
        loss
    ), "Loss must be higher when logits correspond to higher density"

    expected_targets_higher_density = expected_targets

    # make program 1 more active in the bar 2 thatn expected
    expected_targets_higher_density[0, 2, 1] = 1

    loss_higher_density = instrument_density_loss(
        logits, expected_targets_higher_density.float(), instrument_counts
    ).mean()

    assert torch.abs(loss_higher_density) > torch.abs(
        loss
    ), "Loss must be higher when logits correspond to lower density"


if __name__ == "__main__":
    sys.exit(pytest.main())
