import torch
import torch.nn as nn


class ControlEncoder(nn.Module):
    def __init__(self, control_dims, control_token_dim, latent_dim):
        """
        control_dims: dict of {control_type: vocab_size}, e.g. {'genre': 20, 'mood': 10}
        control_token_dim: dimensionality of each control embedding
        latent_dim: output dimension
        """
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {
                key: nn.Embedding(vocab_size, control_token_dim)
                for key, vocab_size in control_dims.items()
            }
        )

        self.fuser = nn.Sequential(
            nn.Linear(control_token_dim * len(control_dims), latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, control_token_ids):
        """
        control_token_ids: dict of {control_type: token_id}, e.g. {'genre': 4, 'mood': 2}
        Returns a latent vector: (latent_dim,)
        """
        embeds = []
        for key, token_id in control_token_ids.items():
            emb = self.embeddings[key](token_id.unsqueeze(0))  # shape (1, dim)
            embeds.append(emb)

        x = torch.cat(embeds, dim=-1).squeeze(0)  # (combined_dim,)
        return self.fuser(x)  # (latent_dim,)
