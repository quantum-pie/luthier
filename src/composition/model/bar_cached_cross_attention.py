import math
import torch.nn as nn
import torch
import torch.nn.functional as F


class BarCachedCrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, num_instruments=16, max_bars=8):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.instrument_embed = nn.Embedding(num_instruments, hidden_dim)
        self.bar_pos_embed = nn.Embedding(max_bars, hidden_dim)

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

    def forward(
        self, query, context, instrument_ids, current_instrument_id, bar_offsets
    ):
        """
        Args:
            query:          (B, T, H)
            context:        (B, N_ctx, H)
            instrument_ids: (B, N_ctx) integer IDs
            bar_offsets:    (B, N_ctx) integer relative bar positions (optional)
        Returns:
            Tensor of shape (B, T, H)
        """
        B, T, H = query.shape
        _, N_ctx, _ = context.shape

        # === Instrument-aware query ===
        if isinstance(current_instrument_id, int):
            current_instrument_id = torch.full(
                (B,), current_instrument_id, dtype=torch.long, device=query.device
            )
        query_inst_embed = self.instrument_embed(current_instrument_id)  # (B, H)
        query = query + query_inst_embed.unsqueeze(1)  # (B, T, H)

        # Embed instrument IDs
        inst_embed = self.instrument_embed(instrument_ids)  # (B, N_ctx, H)

        # Embed bar offsets if provided
        bar_embed = self.bar_pos_embed(
            bar_offsets.clamp(0, self.bar_pos_embed.num_embeddings - 1)
        )
        context = context + inst_embed + bar_embed

        # Project to Q, K, V
        Q = (
            self.query_proj(query)
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)  # (B, num_heads, T, head_dim)
        )
        K = (
            self.key_proj(context)
            .view(B, N_ctx, self.num_heads, self.head_dim)
            .transpose(1, 2)  # (B, num_heads, N_ctx, head_dim)
        )
        V = (
            self.value_proj(context)
            .view(B, N_ctx, self.num_heads, self.head_dim)
            .transpose(1, 2)  # (B, num_heads, N_ctx, head_dim)
        )

        # Compute attention weights
        attn_weights = (Q @ K.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # (B, num_heads, T, N_ctx)

        # Mask out attention to same instrument
        mask = instrument_ids != current_instrument_id  # (B, N_ctx)
        mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N_ctx)
        attn_weights = attn_weights.masked_fill(~mask, -1e9)

        # Softmax and attention
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = (attn_probs @ V).transpose(1, 2).reshape(B, T, H)

        return self.out_proj(attn_output)
