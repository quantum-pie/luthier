import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def masked_mean_bce_logits(logits, targets, mask, dim):
    # Assumes mask.sum(dim=dim) > 0 everywhere
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = loss * mask
    return loss.sum(dim=dim) / mask.sum(dim=dim)


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
    blocks = 0

    for b in range(B):
        for p in range(P):
            pred_block = pred[b, :, p, :]  # [T, I]
            target_block = target[b, :, p, :]  # [T, I]

            if dense_mask is not None:
                mask_block = dense_mask[b, :, p, :]  # [T, I]
            else:
                mask_block = torch.ones_like(pred_block, dtype=torch.bool)

            used_idx = mask_block.any(dim=0).nonzero(as_tuple=False).squeeze(-1)
            K = used_idx.numel()
            if K == 0:
                continue

            pred_block_filtered = pred_block[:, used_idx].float()  # [T, K]
            target_block_filtered = target_block[:, used_idx].float()  # [T, K]
            mask_block_filtered = mask_block[:, used_idx]  # [T, K]

            # Build K×K costs (vectorized over T, K, K)
            mask_i = mask_block_filtered.unsqueeze(2)  # [T, K, 1]
            mask_j = mask_block_filtered.unsqueeze(1)  # [T, 1, K]
            mask_ij = mask_i & mask_j  # [T, K, K] — assumed non-empty per (i,j)

            X = pred_block_filtered.unsqueeze(2).expand(-1, -1, K)  # [T, K, K]
            Y = target_block_filtered.unsqueeze(1).expand(-1, K, -1)  # [T, K, K]

            with torch.no_grad():
                cost = masked_mean_bce_logits(X, Y, mask_ij, dim=0)  # [K, K]
                row_idx, col_idx = linear_sum_assignment(cost.detach().cpu().numpy())
                row_idx = torch.as_tensor(row_idx, device=device)
                col_idx = torch.as_tensor(col_idx, device=device)

            # Recompute matched BCE with grads, vectorized over T and K
            pred_block_matched = pred_block_filtered[:, row_idx]  # [T, K]
            target_block_matched = target_block_filtered[:, col_idx]  # [T, K]
            mask_matched = (
                mask_block_filtered[:, row_idx] & mask_block_filtered[:, col_idx]
            )  # [T, K] — assumed non-empty per column

            per_col = masked_mean_bce_logits(
                pred_block_matched, target_block_matched, mask_matched, dim=0
            )  # [K]
            total_loss = total_loss + per_col.mean()
            blocks += 1

    return total_loss / blocks if blocks > 0 else torch.tensor(0.0, device=device)


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
