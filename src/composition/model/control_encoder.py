import torch
import torch.nn as nn


class ControlEncoder(nn.Module):
    """
    Encodes genre and mood controls into a single conditioning vector.

    Inputs:
        genre_ids:  [B, L] LongTensor of genre token ids
        genre_mask: [B, L] BoolTensor (True = valid position)
        mood_ids:   [B, L] LongTensor of mood token ids
        mood_mask:  [B, L] BoolTensor (True = valid position)

    Assumptions:
        - Every True in masks corresponds to a valid id.
        - Ideally each row has at least one True; an assertion checks this.
    """

    def __init__(
        self,
        genre_vocab_size: int,
        mood_vocab_size: int,
        control_embedding_dim: int,
        latent_dim: int,
    ):
        super().__init__()
        self.genre_embedding = nn.Embedding(genre_vocab_size, control_embedding_dim)
        self.mood_embedding = nn.Embedding(mood_vocab_size, control_embedding_dim)

        fused_in = 2 * control_embedding_dim

        self.default_control = nn.Parameter(torch.randn(fused_in))

        self.fuser = nn.Sequential(
            nn.Linear(fused_in, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    @staticmethod
    def _masked_mean(
        token_ids: torch.LongTensor,  # [B, L]
        token_mask: torch.BoolTensor,  # [B, L] (True = keep)
        embedding: nn.Embedding,
    ) -> torch.FloatTensor:
        """Return [B, C] masked mean of embeddings along L."""
        # sanity: ensure at least one True per row
        assert token_mask.any(dim=1).all(), "Each sample must have at least one valid token."

        emb = embedding(token_ids)  # [B, L, C]
        mask_f = token_mask.unsqueeze(-1).to(emb.dtype)  # [B, L, 1]
        summed = (emb * mask_f).sum(dim=1)  # [B, C]
        counts = mask_f.sum(dim=1)  # [B, 1]
        return summed / counts  # [B, C]

    def forward(
        self,
        genre_ids: torch.LongTensor,
        genre_mask: torch.BoolTensor,
        mood_ids: torch.LongTensor,
        mood_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """Returns fused conditioning vector of shape [B, output_dim]."""
        genre_vec = self._masked_mean(genre_ids, genre_mask, self.genre_embedding)  # [B, C]
        mood_vec = self._masked_mean(mood_ids, mood_mask, self.mood_embedding)  # [B, C]
        fused = torch.cat([genre_vec, mood_vec], dim=-1)  # [B, 2C]
        return self.fuser(fused)  # [B, D]

    def encode_default(self) -> torch.FloatTensor:
        """Returns a default fused control embedding of shape [1, D]."""
        return self.fuser(self.default_control.unsqueeze(0))  # [1, D]
