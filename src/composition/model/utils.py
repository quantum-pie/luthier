import torch


def sample_z_sequence_from_tempo_changes(mu, logvar, tempos, threshold=1.0):
    """
    Resample z from same (mu, logvar) every time tempo changes (abruptly).

    Args:
        mu: (batch, latent_dim)
        logvar: (batch, latent_dim)
        tempos: (batch, seq_len) – tempo values per token (e.g., in beats/sec or BPM)
        threshold: minimum delta in tempo to consider it a change

    Returns:
        z_sequence: (batch, seq_len, latent_dim)
    """
    batch_size, seq_len = tempos.shape
    latent_dim = mu.shape[-1]
    device = mu.device

    std = torch.exp(0.5 * logvar)  # (batch, latent_dim)
    z_sequence = torch.zeros(batch_size, seq_len, latent_dim, device=device)

    for b in range(batch_size):
        eps = torch.randn_like(std[b])  # First z sample
        z_current = mu[b] + eps * std[b]

        z_sequence[b, 0] = z_current
        for t in range(1, seq_len):
            tempo_delta = torch.abs(tempos[b, t] - tempos[b, t - 1])
            if tempo_delta > threshold:
                eps = torch.randn_like(std[b])
                z_current = mu[b] + eps * std[b]
            z_sequence[b, t] = z_current

    return z_sequence


def compute_bar_summaries(x, bar_idx, max_bars):
    """
    Args:
        x:         (B, T, H)       token embeddings
        bar_idx:   (B, T)          integer bar indices per token
    Returns:
        bar_means: (B, max_bars, H) mean embedding per bar
        bar_mask:  (B, max_bars)    True where a real bar exists
    """
    B, T, H = x.shape

    # Prepare index for scatter
    bar_idx_expanded = bar_idx.unsqueeze(-1).expand(-1, -1, H)  # (B, T, H)

    # Accumulate sums and counts per bar
    bar_sums = torch.zeros(B, max_bars, H, device=x.device)
    bar_sums = bar_sums.scatter_add(1, bar_idx_expanded, x)

    bar_counts = torch.zeros(B, max_bars, H, device=x.device)
    bar_counts = bar_counts.scatter_add(1, bar_idx_expanded, torch.ones_like(x))

    # Avoid division by zero
    bar_means = bar_sums / bar_counts.clamp(min=1.0)

    # Optional: bar mask (True where any tokens were present)
    bar_mask = bar_counts[:, :, 0] > 0  # (B, max_bars)

    return bar_means, bar_mask


def compute_bar_summary_for_given_bar(x, bar_idx, target_bar_id):
    """
    Compute mean embedding for a specific bar (per batch).

    Args:
        x:             (B, T, H) token embeddings
        bar_idx:       (B, T)    integer bar indices
        target_bar_id: int       bar index to summarize

    Returns:
        bar_summary: (B, H)
        mask:        (B,) boolean, True where bar was found
    """
    B, T, H = x.shape

    # Create mask for target bar
    mask = bar_idx == target_bar_id  # (B, T)

    # Prevent zero division
    count = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)

    # Apply mask and average
    masked_x = x * mask.unsqueeze(-1)  # (B, T, H)
    bar_summary = masked_x.sum(dim=1) / count  # (B, H)

    exists = mask.sum(dim=1) > 0  # (B,) boolean

    return bar_summary, exists


def build_cross_attention_context(
    full_bar_cache,  # list (B, num_bars, H)
    program_ids,  # list
    context_depth,  # int: how many past bars to attend
    self_index,  # int: current instrument ID
):
    """
    Returns:
        context: Tensor[B, num_bars, total_ctx, H]
        inst_ids: Tensor[B, num_bars, total_ctx]  (instrument IDs per context vector)
    """
    B, num_bars, H = next(iter(full_bar_cache.values())).shape
    device = next(iter(full_bar_cache.values())).device

    # Step 1: relative context bar indices
    indices = torch.arange(num_bars, device=device).unsqueeze(1)
    offsets = torch.arange(context_depth, device=device)
    bar_indices = indices - context_depth + 1 + offsets  # shape (num_bars, context_depth)
    bar_indices = bar_indices.clamp(min=0)

    # Step 2: Gather per-instrument context and assign inst_ids
    context_tensors = []
    id_tensors = []

    for inst_id, bar_summary in zip(program_ids, full_bar_cache):
        if inst_id == self_index:
            continue

        gather_idx = bar_indices.unsqueeze(0).expand(B, -1, -1)  # (B, num_bars, context_depth)
        gather_idx_exp = gather_idx.unsqueeze(-1).expand(-1, -1, -1, H)

        ctx = torch.gather(bar_summary, dim=1, index=gather_idx_exp)  # (B, num_bars, context_depth, H)
        context_tensors.append(ctx)

        inst_id_tensor = torch.full(
            (B, num_bars, context_depth),
            fill_value=inst_id,
            dtype=torch.long,
            device=device,
        )
        id_tensors.append(inst_id_tensor)

    if not context_tensors:
        ctx = torch.zeros(B, num_bars, 1, H, device=device)
        ids = torch.zeros(B, num_bars, 1, dtype=torch.long, device=device)
        return ctx, ids

    full_context = torch.cat(context_tensors, dim=2)  # (B, num_bars, total_ctx, H)
    full_ids = torch.cat(id_tensors, dim=2)  # (B, num_bars, total_ctx)

    return full_context, full_ids


def prepare_bar_offsets(context_tensor, context_depth):
    """
    Args:
        context_tensor: Tensor[B, num_bars, ctx_per_bar, H]
        context_depth:  int, how many bars back we're looking
    Returns:
        Tensor[B, num_bars, ctx_per_bar] with relative bar offsets (0 = most recent)
    """
    B, num_bars, ctx_per_bar, _ = context_tensor.shape
    offsets = torch.arange(context_depth - 1, -1, -1, device=context_tensor.device)  # e.g. [3, 2, 1, 0]

    # If multiple instruments, repeat offsets per instrument
    n_inst = ctx_per_bar // context_depth
    offsets = offsets.repeat(n_inst)  # shape: (ctx_per_bar,)

    return offsets.view(1, 1, -1).expand(B, num_bars, -1)


# Gather context per token based on bar_positions
def gather_token_context(context, token_bar_ids):
    B, T = token_bar_ids.shape
    _, num_bars, ctx, H = context.shape
    token_bar_ids = token_bar_ids.clamp(0, num_bars - 1)

    gather_idx = token_bar_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ctx, H)
    return torch.gather(context.unsqueeze(1).expand(-1, T, -1, -1, -1), dim=2, index=gather_idx)  # (B, T, ctx, H)


def gather_token_meta(meta_tensor, token_bar_ids):
    B, T = token_bar_ids.shape
    _, num_bars, ctx = meta_tensor.shape
    token_bar_ids = token_bar_ids.clamp(0, num_bars - 1)

    gather_idx = token_bar_ids.unsqueeze(-1).expand(-1, -1, ctx)
    return torch.gather(meta_tensor.unsqueeze(1).expand(-1, T, -1, -1), dim=2, index=gather_idx)  # (B, T, ctx)
