import torch.nn.functional as F
import torch


def instrument_counts_loss(logits, target_counts):
    """
    logits: [B, num_programs, max_instances + 1] (unnormalized logits)
    target_counts: [B, num_programs] (int counts from 0 to max_instances)
    """
    B, num_programs, num_classes = logits.shape
    return F.cross_entropy(
        logits.view(
            B * num_programs, num_classes
        ),  # [B * num_programs, max_instances + 1]
        target_counts.view(B * num_programs),  # [B * num_programs]
        reduction="none",
    )


def generate_instrument_counts_targets(track_mask, program_ids, num_programs):
    """
    Vectorized version of per-batch instrument count aggregation.

    Args:
        track_mask: BoolTensor of shape [B, T] — True if track exists
        program_ids: LongTensor of shape [B, T] — program ID for each track
        num_programs: int — total number of program IDs

    Returns:
        LongTensor of shape [B, num_programs] with per-program track counts
    """
    B, T = track_mask.shape

    track_mask_flat = track_mask.view(-1)  # [B*T]
    program_ids_flat = program_ids.view(-1)  # [B*T]

    # Mask out-of-range program IDs
    indices = torch.arange(B, device=program_ids.device).repeat_interleave(T)  # [B*T]
    indices = indices[track_mask_flat]
    prog_ids = program_ids_flat[track_mask_flat]

    # One-hot per valid (batch, program_id) pair
    counts = torch.zeros((B, num_programs), dtype=torch.long, device=program_ids.device)
    counts.index_add_(
        0, indices, F.one_hot(prog_ids, num_classes=num_programs).to(torch.long)
    )

    return counts


def generate_instance_mask_from_ground_truth(
    track_mask, program_ids, num_programs, max_instances
):
    """
    Vectorized version: Generate a mask for instances based on ground truth track mask and program IDs.

    Args:
        track_mask: BoolTensor of shape [B, T] — True if track exists
        program_ids: LongTensor of shape [B, T] — program ID for each track
        num_programs: int — total number of program IDs
        max_instances: int — maximum number of instances per program

    Returns:
        instance_mask: BoolTensor of shape [B, num_programs, max_instances] — True if instance exists
    """
    B, T = track_mask.shape

    # Get valid track indices
    valid_indices = track_mask.nonzero(
        as_tuple=False
    )  # [N, 2], where N is number of True entries
    b_idx, t_idx = valid_indices[:, 0], valid_indices[:, 1]
    prog_idx = program_ids[b_idx, t_idx]  # [N]

    # Count how many times each program_id has occurred per batch
    linear_idx = b_idx * num_programs + prog_idx  # Unique ID for each (batch, program)
    counts = torch.zeros(B * num_programs, dtype=torch.long, device=track_mask.device)
    instance_idx = torch.zeros_like(linear_idx)

    # Use scatter_add to compute instance indices (i.e., count per program within batch)
    for i in range(linear_idx.size(0)):
        instance_idx[i] = counts[linear_idx[i]]
        if counts[linear_idx[i]] < max_instances:
            counts[linear_idx[i]] += 1

    # Filter out indices where instance index exceeds max_instances
    keep = instance_idx < max_instances
    b_final = b_idx[keep]
    p_final = prog_idx[keep]
    i_final = instance_idx[keep]

    # Create final mask
    instance_mask = torch.zeros(
        (B, num_programs, max_instances), dtype=torch.bool, device=track_mask.device
    )
    instance_mask[b_final, p_final, i_final] = True

    return instance_mask


def generate_instance_mask_from_logits(logits):
    """
    Generate an instance presence mask from instrument count logits.

    Args:
        logits: FloatTensor of shape [B, num_programs, max_instances + 1] — logits over possible instance counts
                for each program. The index of the max logit is the predicted number of instances.

    Returns:
        instance_mask: BoolTensor of shape [B, num_programs, max_instances] — True for active instances.
    """
    B, num_programs, max_plus_1 = logits.shape
    max_instances = max_plus_1 - 1

    # Predict instance count (0 to max_instances) per [B, num_programs]
    predicted_counts = torch.argmax(logits, dim=-1)  # shape: [B, num_programs]

    # Create a tensor of shape [1, 1, max_instances] like [0, 1, 2, ..., max_instances - 1]
    instance_range = torch.arange(max_instances, device=logits.device).view(1, 1, -1)

    # Expand predicted_counts to compare against instance_range
    instance_mask = instance_range < predicted_counts.unsqueeze(-1)

    return instance_mask
