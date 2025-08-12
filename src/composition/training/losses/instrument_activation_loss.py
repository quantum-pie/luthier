import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def instrument_activation_loss(pred, target, dense_mask=None):
    """
    Compute permutation-invariant BCE loss across instrument instances using the Hungarian algorithm.

    Args:
        pred:        [B, T, P, I] — predicted logits
        target:      [B, T, P, I] — binary targets
        dense_mask:  [B, T, P, I] — boolean mask for valid positions (optional)

        B is the batch size,
        T is the number of time steps (bars),
        P is the number of programs (instruments),
        I is the number of instances per program.
    Returns:
        scalar averaged loss
    """
    B, T, P, I = pred.shape

    device = pred.device
    total_loss = torch.tensor(0.0, device=device)
    total_pairs = 0

    for b in range(B):
        for p in range(P):
            pred_block = pred[b, :, p, :]  # [T, I]
            target_block = target[b, :, p, :]  # [T, I]

            if dense_mask is not None:
                mask_block = dense_mask[b, :, p, :]  # [T, I]
            else:
                mask_block = torch.ones_like(pred_block, dtype=torch.bool)

            # Compute cost matrix [I, I]
            with torch.no_grad():
                used_idx = mask_block.any(dim=0).nonzero(as_tuple=False).squeeze(-1)
                num_used_idx = used_idx.numel()
                if num_used_idx == 0:
                    continue

                cost = torch.zeros((num_used_idx, num_used_idx), device=device)
                for i in range(num_used_idx):
                    for j in range(num_used_idx):
                        idx_i = used_idx[i]
                        idx_j = used_idx[j]
                        pred_vec = pred_block[:, idx_i]
                        target_vec = target_block[:, idx_j]
                        valid_mask = mask_block[:, idx_i] & mask_block[:, idx_j]  # [T]

                        cost[i, j] = F.binary_cross_entropy_with_logits(
                            pred_vec[valid_mask],
                            target_vec[valid_mask].float(),
                            reduction="mean",
                        )

                row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
                row_ind = torch.as_tensor(row_ind, device=device)
                col_ind = torch.as_tensor(col_ind, device=device)

            # Accumulate matched BCE losses
            for i, j in zip(row_ind.tolist(), col_ind.tolist()):
                idx_i = used_idx[i]
                idx_j = used_idx[j]
                pred_vec = pred_block[:, idx_i]
                target_vec = target_block[:, idx_j]
                valid = mask_block[:, idx_i] & mask_block[:, idx_j]
                total_loss = total_loss + F.binary_cross_entropy_with_logits(
                    pred_vec[valid],
                    target_vec[valid].float(),
                    reduction="mean",
                )
                total_pairs += 1

    return (
        total_loss / total_pairs
        if total_pairs > 0
        else torch.tensor(0.0, device=pred.device)
    )


def generate_instrument_activation_targets(
    bar_activations, track_mask, program_ids, num_programs, max_instances
):
    """
    Generate binary targets for instrument activation based on bar activations and program IDs.
    Args:
        bar_activations: [B, T, L] — binary activation matrix for bars
        track_mask:      [B, T] — mask indicating valid tracks
        program_ids:     [B, T] — program IDs for each track
        num_programs:    int — total number of programs (instruments)
        max_instances:   int — maximum number of instances per program

        B is the batch size,
        T is the number of tracks,
        L is the number of time steps (bars).
    Returns:
        [B, L, num_programs, max_instances] — binary targets for instrument activation
    """
    B, T, L = bar_activations.shape

    valid_mask = track_mask & (program_ids >= 0)
    valid_indices = valid_mask.nonzero(as_tuple=False)  # [N, 2]

    if valid_indices.numel() == 0:
        return torch.zeros(
            (B, L, num_programs, max_instances),
            dtype=torch.bool,
            device=bar_activations.device,
        )

    b_idx = valid_indices[:, 0]
    t_idx = valid_indices[:, 1]
    prog_idx = program_ids[b_idx, t_idx]
    linear_prog = b_idx * num_programs + prog_idx

    instance_counts = torch.zeros(
        B * num_programs, dtype=torch.long, device=bar_activations.device
    )
    instance_ids = torch.full_like(prog_idx, -1)

    # Vectorized group instance assignment (fast and non-stable)
    torch.unique_consecutive(linear_prog, return_inverse=True)
    for i in range(len(linear_prog)):
        key = linear_prog[i]
        count = instance_counts[key]
        if count < max_instances:
            instance_ids[i] = count
            instance_counts[key] += 1

    # Keep valid instances
    keep = instance_ids >= 0
    b_idx = b_idx[keep]
    t_idx = t_idx[keep]
    prog_idx = prog_idx[keep]
    inst_idx = instance_ids[keep]

    flat_bar_activations = bar_activations[b_idx, t_idx]  # [N, L]
    N = flat_bar_activations.size(0)

    l_idx = torch.arange(L, device=bar_activations.device).view(1, -1).expand(N, L)
    b_idx = b_idx.view(-1, 1).expand(N, L)
    prog_idx = prog_idx.view(-1, 1).expand(N, L)
    inst_idx = inst_idx.view(-1, 1).expand(N, L)

    active_mask = flat_bar_activations.bool()
    b_flat = b_idx[active_mask]
    l_flat = l_idx[active_mask]
    p_flat = prog_idx[active_mask]
    i_flat = inst_idx[active_mask]

    targets = torch.zeros(
        (B, L, num_programs, max_instances),
        dtype=torch.bool,
        device=bar_activations.device,
    )
    targets[b_flat, l_flat, p_flat, i_flat] = True

    return targets
