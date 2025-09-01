import torch
import torch.nn.functional as F
from torch.special import gammaln


def masked_mean_bce_logits(logits, targets, mask, dim):
    # Assumes mask.sum(dim=dim) > 0 everywhere
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = loss * mask
    return loss.sum(dim=dim) / mask.sum(dim=dim)


def instrument_density_loss(pred_logits, taget_instrument_density, instrument_counts):
    """
    Compute the binomial negative log-likelihood loss for instrument density predictions.
    Args:
        pred_logits:               [B, T, P] — predicted logits for instrument density
        taget_instrument_density:  [B, T, P] — target instrument density (0 to C)
        instrument_counts:         [B, P]    — piece-level instrument counts (C)
    Returns:
        Loss tensor of shape [B, T, P] (not reduced)
    """
    B, T, P = pred_logits.shape
    n = instrument_counts.unsqueeze(1).expand(B, T, P)  # [B, T, P]
    k = taget_instrument_density  # [B, T, P]
    q = torch.sigmoid(pred_logits)  # [B, T, P]

    log_binom = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    return -(log_binom + k * torch.log(q) + (n - k) * torch.log1p(-q))


def generate_instrument_density_targets(
    bar_activations,  # [B, T, L]  (0/1 or bool): per-track activation per bar
    track_mask,  # [B, T]     (bool): valid tracks
    program_ids,  # [B, T]     (long in [0..num_programs-1] for valid tracks)
    instrument_counts,  #  [B, num_programs] piece-level counts
    num_programs: int,
):
    """
    Returns:
        K: LongTensor [B, L, num_programs]  — active instance counts per bar & program
            - Each entry is the number of valid tracks of that program active in that bar.
            - Capping: K[b,l,p] <= C[b,p]
    """
    B, T, L = bar_activations.shape
    device = bar_activations.device

    # Edge case: nothing valid
    if not track_mask.any():
        K = torch.zeros((B, L, num_programs), dtype=torch.long, device=device)
        return K

    # Indices of valid tracks
    valid = track_mask.nonzero(as_tuple=False)  # [N,2] with (b,t)
    b_idx = valid[:, 0]
    t_idx = valid[:, 1]
    p_idx = program_ids[b_idx, t_idx]  # [N]

    # Gather per-bar activations for valid tracks: [N, L]
    acts = bar_activations[b_idx, t_idx].to(torch.long)  # ensure integer

    # Build scatter indices repeated across bars
    N = acts.size(0)
    l_range = torch.arange(L, device=device)
    l_idx = l_range.unsqueeze(0).expand(N, L).reshape(-1)  # [N*L]
    b_rep = b_idx.unsqueeze(1).expand(N, L).reshape(-1)  # [N*L]
    p_rep = p_idx.unsqueeze(1).expand(N, L).reshape(-1)  # [N*L]
    val = acts.reshape(-1)  # [N*L]

    # Accumulate into K: [B, L, P]
    K = torch.zeros((B, L, num_programs), dtype=torch.long, device=device)
    K.index_put_((b_rep, l_idx, p_rep), val, accumulate=True)

    # Cap by provided C (piece-level counts)
    K = torch.minimum(K, instrument_counts.unsqueeze(1).expand_as(K))

    return K
