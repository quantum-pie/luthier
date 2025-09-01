import torch
import math


class LatentInterpolator:
    def __init__(self, transition_duration=4, smooth=True):
        """
        Args:
            transition_duration (int): How many bars to interpolate over
            smooth (bool): Whether to use cosine interpolation
        """
        self.transition_duration = transition_duration
        self.smooth = smooth
        self.z_old = None
        self.z_new = None
        self.transition_start_bar = None

    def update(self, new_z, current_bar_pos):
        """
        Updates the latent state when control tokens change.

        Args:
            new_z (Tensor): New z_global from new control tokens (shape: latent_dim)
            current_bar_pos (float): Current bar index
        """
        if self.z_old is None:
            # First-time init — no interpolation needed
            self.z_old = new_z
            self.z_new = None
            self.transition_start_bar = None
        else:
            # Start interpolating from z_old to z_new
            self.z_old = self.get_current_z(current_bar_pos)
            self.z_new = new_z
            self.transition_start_bar = current_bar_pos

    def is_initialized(self):
        return self.z_old is not None

    def get_current_z(self, bar_pos):
        """
        Get the current interpolated z_global at a given bar position.

        Args:
            bar_pos (float): Current bar index

        Returns:
            Tensor: Interpolated latent vector
        """
        if self.z_old is None:
            raise ValueError("LatentInterpolator must be initialized with update() before use.")

        if self.z_new is None or self.transition_start_bar is None:
            return self.z_old  # No transition happening

        # Compute interpolation factor α ∈ [0, 1]
        bar_offset = bar_pos - self.transition_start_bar
        alpha = torch.clamp(bar_offset / self.transition_duration, 0.0, 1.0)

        if self.smooth:
            alpha = 0.5 * (1 - torch.cos(math.pi * alpha))

        # Interpolate between z_old and z_new
        z_interp = (1 - alpha) * self.z_old + alpha * self.z_new

        # If interpolation is done, finalize transition
        if alpha >= 1.0:
            self.z_old = self.z_new
            self.z_new = None
            self.transition_start_bar = None

        return z_interp
