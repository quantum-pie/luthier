from src.composition.model.control_encoder import ControlEncoder
from src.composition.model.instrument_count_head import InstrumentCountHead
from src.composition.model.latent_interpolator import LatentInterpolator
from src.composition.model.latent_prior_net import LatentPriorNet

from src.composition.model.time_aware_embedding import (
    ExponentialRotaryEmbedding,
    apply_rotary,
)

from src.composition.model.utils import (
    sample_z_sequence_from_tempo_changes,
)

import torch
import torch.nn as nn


class Conductor(nn.Module):
    def __init__(
        self,
        latent_dim,
        control_embed_dim,
        genre_vocab_size,
        mood_vocab_size,
        num_instruments=129,  # 128 GM + 1 for drums
        max_instrument_instances=10,
    ):
        super().__init__()
        self.num_instruments = num_instruments
        self.max_instrument_instances = max_instrument_instances

        self.control_encoder = ControlEncoder(
            genre_vocab_size, mood_vocab_size, control_embed_dim, latent_dim
        )
        self.latent_prior_net = LatentPriorNet(latent_dim)

        self.bar_embedding = ExponentialRotaryEmbedding(latent_dim, base=512.0)

        self.tempo_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 1)
        )

        self.used_in_piece_head = InstrumentCountHead(
            latent_dim=latent_dim,
            num_programs=num_instruments,
            max_instances=max_instrument_instances,
        )

        self.activation_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_instruments * max_instrument_instances),
        )

        self.last_control_hash = None
        self.latent_interpolator = LatentInterpolator(
            transition_duration=4, smooth=True
        )

    def control_hash(self, control_tokens):
        """
        Compute a hash of the control tokens for caching purposes.
        """
        return hash(frozenset(control_tokens))

    def forward(
        self,
        model_inputs,
        control_tokens=None,
    ):
        batch_size = len(model_inputs["bar_tempos"])
        if control_tokens is not None:
            mu, logvar = self.latent_prior_net(
                self.control_encoder(
                    control_tokens["genre_ids"],
                    control_tokens["genre_mask"],
                    control_tokens["mood_ids"],
                    control_tokens["mood_mask"],
                )
            )
        else:
            # If no control tokens, use default prior
            mu, logvar = self.latent_prior_net()
            mu = mu.repeat(batch_size, 1)
            logvar = logvar.repeat(batch_size, 1)

        z_sequence = sample_z_sequence_from_tempo_changes(
            mu, logvar, model_inputs["bar_tempos"]
        )

        tempos_pred = self.tempo_head(z_sequence).squeeze(-1)

        # Infer used instruments
        _, instruments_counts_logits = self.used_in_piece_head(z_sequence.mean(dim=1))

        # Apply bar embedding to latent vector. The total bar is sequence of unique bars
        bar_sin, bar_cos = self.bar_embedding(
            model_inputs["bar_boundaries"]
        )  # (batch, total_bars, latent_dim)

        z_bar = apply_rotary(z_sequence, bar_sin, bar_cos)

        # Instruments conditioned on bar-modulated latent vector
        instrument_activation_logits = self.activation_head(
            z_bar
        )  # (batch, total_bars, num_instruments * max_instrument_instances)

        return {
            "tempos": tempos_pred,
            "instrument_counts_logits": instruments_counts_logits,
            "instrument_activation_logits": instrument_activation_logits,
            "latent_mu": mu,
            "latent_logvar": logvar,
        }
