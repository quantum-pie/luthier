import pytest
import sys

from src.composition.training.losses.instrument_activation_loss import (
    generate_instrument_activation_targets,
    instrument_activation_loss,
)
import torch


def test_generate_instrument_activation_targets_case_1():
    num_programs = 3
    max_instances = 2
    max_global_len = 3

    track_mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
    program_ids = torch.tensor([[1, 1, 2, 2], [0, 0, 1, -1]])

    bar_activations = torch.tensor(
        [
            [[0, 1, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]],
            [[0, 0, 1], [0, 0, 1], [1, 0, 1], [0, 0, 0]],
        ]
    )

    expected_targets = torch.zeros(
        (2, max_global_len, num_programs, max_instances), dtype=torch.bool
    )
    expected_targets[0] = torch.tensor(
        [
            [[0, 0], [0, 0], [1, 0]],  # Bar 0
            [[0, 0], [1, 1], [0, 0]],  # Bar 1
            [[0, 0], [1, 0], [0, 1]],  # Bar 2
        ]
    )

    expected_targets[1] = torch.tensor(
        [
            [[0, 0], [1, 0], [0, 0]],  # Bar 0
            [[0, 0], [0, 0], [0, 0]],  # Bar 1
            [[1, 1], [1, 0], [0, 0]],  # Bar 2
        ]
    )

    targets = generate_instrument_activation_targets(
        bar_activations, track_mask, program_ids, num_programs, max_instances
    )

    assert targets.shape == (2, max_global_len, num_programs, max_instances)
    assert torch.equal(
        targets, expected_targets
    ), "Generated targets do not match expected values"


def test_generate_instrument_activation_targets_case_2():
    num_programs = 4
    max_instances = 2
    max_global_len = 4

    track_mask = torch.tensor(
        [[True, True, True, False], [True, True, True, True]]  # Batch 0  # Batch 1
    )

    program_ids = torch.tensor(
        [
            [0, 1, 0, -1],  # Batch 0: Track 0 & 2 -> program 0, Track 1 -> program 1
            [2, 2, 3, 3],  # Batch 1: Track 0 & 1 -> program 2, Track 2 & 3 -> program 3
        ]
    )

    bar_activations = torch.tensor(
        [
            [[1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0]],  # Batch 0
            [[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 1], [0, 0, 0, 1]],  # Batch 1
        ],
        dtype=torch.bool,
    )

    expected_targets = torch.zeros(
        (2, max_global_len, num_programs, max_instances), dtype=torch.bool
    )

    # --- Batch 0 ---
    # Program 0 gets tracks 0 and 2 → assigned instances 0 and 1
    # Track 0 active at bars 0 and 2
    expected_targets[0, 0, 0, 0] = True
    expected_targets[0, 2, 0, 0] = True
    # Track 2 active at bars 0 and 3
    expected_targets[0, 0, 0, 1] = True
    expected_targets[0, 3, 0, 1] = True

    # Program 1 gets track 1 → assigned instance 0
    # Track 1 active at bar 1
    expected_targets[0, 1, 1, 0] = True

    # --- Batch 1 ---
    # Program 2 gets tracks 0 and 1 → assigned instances 0 and 1
    # Track 0 active at bars 0 and 1
    expected_targets[1, 0, 2, 0] = True
    expected_targets[1, 1, 2, 0] = True
    # Track 1 active at bars 1 and 2
    expected_targets[1, 1, 2, 1] = True
    expected_targets[1, 2, 2, 1] = True

    # Program 3 gets tracks 2 and 3 → assigned instances 0 and 1
    # Track 2 active at bars 0, 2, 3
    expected_targets[1, 0, 3, 0] = True
    expected_targets[1, 2, 3, 0] = True
    expected_targets[1, 3, 3, 0] = True
    # Track 3 active at bar 3
    expected_targets[1, 3, 3, 1] = True

    # Run
    targets = generate_instrument_activation_targets(
        bar_activations,
        track_mask,
        program_ids,
        num_programs=num_programs,
        max_instances=max_instances,
    )

    # Check
    assert targets.shape == expected_targets.shape
    assert torch.equal(
        targets, expected_targets
    ), "Test case 2 failed: Targets don't match expected output."


def test_instrument_activation_loss():
    num_programs = 2
    max_instances = 3
    max_global_len = 3

    expected_targets = torch.zeros(
        (1, max_global_len, num_programs, max_instances), dtype=torch.bool
    )

    # Program 0 tracks: assigned to instance 0, 1, 2
    expected_targets[0, 0, 0, 0] = True  # track 0
    expected_targets[0, 1, 0, 1] = True  # track 1
    expected_targets[0, 2, 0, 2] = True  # track 2

    # Program 1 track (track 3): assigned to instance 0
    expected_targets[0, 0, 1, 0] = True
    expected_targets[0, 1, 1, 0] = True

    # simple mapping because targets are strictly boolean
    logits = (expected_targets.float() - 0.5) * 10000.0

    loss = instrument_activation_loss(
        logits,
        expected_targets.float(),
    )

    assert (
        torch.abs(loss) < 1e-5
    ), "Loss should be close to zero for perfect predictions"

    # check that loss is permutation invariant with respect to program instances
    # swap instances 0 and 1 in the first batch
    logits_swapped = logits.clone()
    logits_swapped[0, :, 0, 0] = logits[0, :, 0, 1]
    logits_swapped[0, :, 0, 1] = logits[0, :, 0, 0]

    loss_swapped = instrument_activation_loss(
        logits_swapped.float(),
        expected_targets.float(),
    )

    assert torch.allclose(
        loss, loss_swapped, atol=1e-5
    ), "Loss should be invariant to instance permutation"

    # Check masked loss
    # First, we're going to perturb logits in the instace 2 of program 0
    logits_perturbed = logits.clone()
    logits_perturbed[0, 2, 0, 2] = False
    logits_perturbed[0, 1, 0, 2] = True

    # Now, we expect the loss to be non-zero
    loss_perturbed = instrument_activation_loss(
        logits_perturbed.float(),
        expected_targets.float(),
    )

    assert (
        torch.abs(loss_perturbed) > 1e-5
    ), "Loss should be non-zero for perturbed predictions"

    # Now let's create a mask that ignores 3rd instance of program 0
    mask = torch.ones(
        (1, max_global_len, num_programs, max_instances), dtype=torch.bool
    )
    mask[:, :, 0, 2] = False  # Ignore 3rd instance of program 0

    # swap instances 0 and 1 in the first batch
    logits_swapped_perturbed = logits_perturbed.clone()
    logits_swapped_perturbed[0, :, 0, 0] = logits_perturbed[0, :, 0, 1]
    logits_swapped_perturbed[0, :, 0, 1] = logits_perturbed[0, :, 0, 0]

    loss_masked = instrument_activation_loss(
        logits_swapped_perturbed.float(),
        expected_targets.float(),
        dense_mask=mask,
    )

    # expect that masked loss is zero, since we ignore the perturbed instance
    assert (
        torch.abs(loss_masked) < 1e-5
    ), "Masked loss should be close to zero when ignoring perturbed instance"

    # now let's also perturb bar-wise
    logits_swapped_perturbed[0, 0, 0, 0] = -logits_swapped_perturbed[0, 0, 0, 0]
    loss_masked_bar = instrument_activation_loss(
        logits_swapped_perturbed.float(),
        expected_targets.float(),
        dense_mask=mask,
    )

    # expect that masked loss is non-zero, since we perturb the first bar
    assert (
        torch.abs(loss_masked_bar) > 1e-5
    ), "Masked loss should be non-zero when perturbing bar-wise"

    # now let's also add bar-wise mask
    mask[:, 0, :, :] = False  # Ignore first bar

    loss_masked_bar_ignored = instrument_activation_loss(
        logits_swapped_perturbed.float(),
        expected_targets.float(),
        dense_mask=mask,
    )

    # expect that masked loss is zero, since we ignore the first bar
    assert (
        torch.abs(loss_masked_bar_ignored) < 1e-5
    ), "Masked loss should be close to zero when ignoring the first bar"


if __name__ == "__main__":
    sys.exit(pytest.main())
