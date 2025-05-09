import torch.nn as nn
import torch
import torch.nn.functional as F


class InstrumentCountHead(nn.Module):
    def __init__(self, latent_dim, num_programs, max_instances):
        super().__init__()
        self.num_programs = num_programs
        self.max_instances = max_instances

        # Per-program classifier: (max_instances + 1) classes per program (e.g. 0, 1, 2, 3)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(
                latent_dim, num_programs * (max_instances + 1)
            ),  # one classifier per program
        )

    def forward(self, cond_vector):
        """
        cond_vector: [B, cond_dim]
        Returns:
            counts: [B, num_programs] (int values from 0 to max_instances)
            probs: [B, num_programs, max_instances + 1] (softmax over classes)
        """
        B = cond_vector.size(0)
        logits = self.fc(cond_vector)  # [B, num_programs * (max_instances + 1)]
        logits = logits.view(B, self.num_programs, self.max_instances + 1)
        counts = torch.argmax(logits, dim=-1)  # [B, num_programs]

        return counts, logits
