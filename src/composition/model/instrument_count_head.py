import torch.nn as nn
import torch
import torch.nn.functional as F


class InstrumentCountHead(nn.Module):
    def __init__(self, latent_dim, num_programs, max_instances):
        super().__init__()
        self.num_programs = num_programs
        self.max_instances = max_instances
        self.fc = nn.Linear(latent_dim, num_programs)

    def forward(self, z):
        # Predict λ > 0
        rates = F.softplus(self.fc(z)) + 1e-6  # [B, P]
        return rates
