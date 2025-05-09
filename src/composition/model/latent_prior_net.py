import torch.nn as nn
import torch


class LatentPriorNet(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.default_control = nn.Parameter(torch.randn(latent_dim))
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim * 2),  # Outputs mean and logvar
        )

    def forward(self, z_control=None):
        if z_control is None:
            z_control = self.default_control.unsqueeze(0)

        params = self.net(z_control)  # (batch, seq_len, latent_dim * 2)
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar
