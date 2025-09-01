import torch.nn.functional as F
import torch


def instrument_counts_loss(rates, target_counts):
    """
    Standard Poisson NLL.
    target_counts: [B, P] integers >= 0
    """
    # torch Poisson NLL expects log λ if log_input=True; we pass log(rates)
    return F.poisson_nll_loss(
        input=rates.clamp_min(1e-12).log(),
        target=target_counts,
        log_input=True,
        full=True,  # adds Stirling term for stability
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
    counts.index_add_(0, indices, F.one_hot(prog_ids, num_classes=num_programs).to(torch.long))

    return counts
