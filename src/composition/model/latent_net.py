import torch.nn as nn
import torch


def masked_global_mean(x, mask, eps=1e-6):
    # x: [B,P,T,D], mask: [B,P,T] (bool),
    m = mask.float()
    num = (x * m.unsqueeze(-1)).sum(dim=(1, 2))  # [B,D]
    den = m.sum(dim=(1, 2)).clamp_min(eps).unsqueeze(-1)  # [B,1]
    return num / den  # [B,D]


class LatentPriorNet(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim * 2),  # Outputs mean and logvar
        )

    def forward(self, z_control):
        """
        z_control: [B, latent_dim]
        Returns:
            mu, logvar: both [B, latent_dim]
        """
        params = self.net(z_control)  # (batch, latent_dim * 2)
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar


class LatentPosteriorNet(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim * 2),  # Outputs mean and logvar
        )

    def forward(self, z_control, x, attention):
        """
        z_control: [B, latent_dim] - control embedding
        x: [B, P, seq_len, latent_dim] - input embeddings to condition on
        attention: [B, P, seq_len] - attention mask for x
        Returns:
            mu, logvar: both [B, latent_dim]
        """
        x_emb = torch.cat([z_control, masked_global_mean(x, attention)], dim=-1)
        params = self.net(x_emb)  # (B, latent_dim * 2)
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar
