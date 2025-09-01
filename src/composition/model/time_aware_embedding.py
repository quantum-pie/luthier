import torch
import torch.nn as nn
import math


def apply_rotary(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
    """
    Applies rotary position embedding.

    Args:
        x:   (batch, seq_len, dim)
        sin: (batch, seq_len, dim // 2)
        cos: (batch, seq_len, dim // 2)

    Returns:
        Tensor of shape (batch, seq_len, dim)
    """
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_odd * cos + x_even * sin

    return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, frequencies: torch.Tensor):
        super().__init__()
        freqs = frequencies
        self.register_buffer("freqs", freqs)

    def forward(self, positions: torch.Tensor):
        """
        Args:
            position_in_bar: (batch, seq_len), float in [0, 1)

        Returns:
            sin, cos: both (batch, seq_len, hidden_dim // 2)
        """
        angles = torch.einsum("bs,d -> bsd", positions, self.freqs)  # (B, S, hidden_dim//2)
        return torch.sin(angles), torch.cos(angles)

    def get_frequencies(self):
        return self.freqs


class ExponentialRotaryEmbedding(RotaryEmbedding):
    def __init__(self, dim=64, base=512.0):
        assert dim % 2 == 0

        # Exponential frequencies (transformer-style)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim // 2).float() / (dim // 2)))

        super(ExponentialRotaryEmbedding, self).__init__(inv_freq)


class LinearRotaryEmbedding(RotaryEmbedding):
    def __init__(self, dim=64):
        assert dim % 2 == 0

        # Linear frequencies (in radians)
        freqs = torch.arange(1, dim // 2 + 1).float() * 2 * math.pi

        super(LinearRotaryEmbedding, self).__init__(freqs)


class DualRotartyEmbedding(nn.Module):
    def __init__(
        self,
        dim=64,
        base=10000.0,
    ):
        super().__init__()

        assert dim % 4 == 0

        self.linear_dim = dim // 2
        self.exponential_dim = dim // 2
        self.total_dim = dim

        self.linear_emb = LinearRotaryEmbedding(dim=self.linear_dim)

        self.exponential_emb = ExponentialRotaryEmbedding(dim=self.exponential_dim, base=base)

    def forward(self, linear_pos: torch.Tensor, exponential_pos: torch.Tensor):
        """
        Args:
            linear_pos: (batch, seq_len), float in [0, 1)
            exponential_pos: (batch, seq_len), float in [0, 1)

        Returns:
            sin, cos: both (batch, seq_len, total_dim // 2)
        """
        linear_sin, linear_cos = self.linear_emb.forward(linear_pos)
        exponential_sin, exponential_cos = self.exponential_emb.forward(exponential_pos)
        # (batch, seq_len, total_dim // 2)
        return (
            torch.cat([linear_sin, exponential_sin], dim=-1),
            torch.cat([linear_cos, exponential_cos], dim=-1),
        )

    def get_linear_frequencies(self):
        return self.linear_emb.get_frequencies()

    def get_exponential_frequencies(self):
        return self.exponential_emb.get_frequencies()
