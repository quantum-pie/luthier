import hashlib
import struct
from src.composition.model.control_encoder import ControlEncoder
from src.composition.model.instrument_count_head import InstrumentCountHead
from src.composition.model.latent_interpolator import LatentInterpolator
from src.composition.model.latent_net import LatentPriorNet, LatentPosteriorNet

from src.composition.model.time_aware_embedding import (
    ExponentialRotaryEmbedding,
    apply_rotary,
)

from src.composition.model.utils import (
    sample_z_sequence_from_tempo_changes,
)

import torch
import torch.nn as nn
import numpy as np


class Conductor(nn.Module):
    def __init__(
        self,
        latent_dim,
        control_embed_dim,
        hidden_dim,
        genre_vocab_size,
        mood_vocab_size,
        num_instruments,
        max_instrument_instances,
    ):
        super().__init__()
        self.num_instruments = num_instruments
        self.max_instrument_instances = max_instrument_instances

        self.hidden_to_latent_proj = nn.Linear(hidden_dim, latent_dim)

        self.control_encoder = ControlEncoder(genre_vocab_size, mood_vocab_size, control_embed_dim, latent_dim)
        self.latent_prior_net = LatentPriorNet(latent_dim)
        self.latent_posterior_net = LatentPosteriorNet(latent_dim)

        self.bar_embedding = ExponentialRotaryEmbedding(latent_dim, base=512.0)

        self.tempo_head = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 1))

        self.used_in_piece_head = InstrumentCountHead(
            latent_dim=latent_dim,
            num_programs=num_instruments,
            max_instances=max_instrument_instances,
        )

        self.density_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_instruments),
        )

        self.last_control_hash = None
        self.latent_interpolator = LatentInterpolator(transition_duration=4, smooth=True)

    def forward(
        self,
        model_inputs,
        input_embeddings,
        control_tokens=None,
    ):
        batch_size = len(model_inputs["bar_tempos"])
        if control_tokens is not None:
            control_embeddings = self.control_encoder(
                control_tokens["genre_ids"],
                control_tokens["genre_mask"],
                control_tokens["mood_ids"],
                control_tokens["mood_mask"],
            )
        else:
            # If no control tokens, use default control
            control_embeddings = self.control_encoder.encode_default()  # [1, latent_dim]
            control_embeddings = control_embeddings.repeat(batch_size, 1)  # [B, latent_dim]

        input_attention_mask = model_inputs["attention_mask"]

        latent_inputs = self.hidden_to_latent_proj(input_embeddings)  # (batch, seq_len, latent_dim)

        mu_prior, logvar_prior = self.latent_prior_net(control_embeddings)
        mu_posterior, logvar_posterior = self.latent_posterior_net(
            control_embeddings, latent_inputs, input_attention_mask
        )

        z_sequence = sample_z_sequence_from_tempo_changes(mu_posterior, logvar_posterior, model_inputs["bar_tempos"])

        tempos_pred = self.tempo_head(z_sequence).squeeze(-1)

        # Infer used instruments
        instruments_counts_rates = self.used_in_piece_head(z_sequence.mean(dim=1))

        # Apply bar embedding to latent vector. The total bar is sequence of unique bars
        bar_sin, bar_cos = self.bar_embedding(model_inputs["bar_boundaries"])  # (batch, total_bars, latent_dim)

        z_bar = apply_rotary(z_sequence, bar_sin, bar_cos)

        # Instrumentation density conditioned on bar-modulated latent vector
        instrument_density_logits = self.density_head(z_bar)  # (batch, total_bars, num_instruments)

        return {
            "tempos": tempos_pred,
            "instrument_counts_rates": instruments_counts_rates,
            "instrument_density_logits": instrument_density_logits,
            "mu_posterior": mu_posterior,
            "logvar_posterior": logvar_posterior,
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
        }

    def control_hash(self, control_tokens):
        """
        Compute a hash of the control tokens for caching purposes.
        """
        if control_tokens is None:
            return None

        # Extract and normalize dtypes (consistent endianness & size)
        g = np.asarray(control_tokens["genre_ids"].cpu().numpy())
        m = np.asarray(control_tokens["mood_ids"].cpu().numpy())

        g = np.sort(g)
        m = np.sort(m)

        # Build a tagged, length-prefixed payload: [ver][G][len][data][M][len][data]
        # Lengths are counts (not bytes) to avoid ambiguity if dtype changes.
        payload = bytearray()
        payload += b"V1"  # version tag for future-proofing

        payload += struct.pack("<cI", b"G", g.size)
        payload += g.newbyteorder("<").tobytes()  # little-endian

        payload += struct.pack("<cI", b"M", m.size)
        payload += m.newbyteorder("<").tobytes()

        return hashlib.sha256(payload).hexdigest()

    def forward_step(self, bar_position, control_tokens=None):
        control_hash = self.control_hash(control_tokens)

        # A single step forward for generation
        if not self.latent_interpolator.is_initialized() or self.last_control_hash != control_hash:
            self.last_control_hash = control_hash

            if control_tokens is not None:
                genre_ids = control_tokens["genre_ids"].unsqueeze(0)
                genre_mask = torch.ones_like(genre_ids, dtype=torch.bool)
                mood_ids = control_tokens["mood_ids"].unsqueeze(0)
                mood_mask = torch.ones_like(mood_ids, dtype=torch.bool)

                control_embeddings = self.control_encoder(genre_ids, genre_mask, mood_ids, mood_mask)
            else:
                # If no control tokens, use default prior
                control_embeddings = self.control_encoder.encode_default()  # [1, latent_dim]

            mu, logvar = self.latent_prior_net(control_embeddings)

            # Update latent interpolator with new control token
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std  # (batch, 1, latent_dim)
            self.latent_interpolator.update(z, bar_position)

        z = self.latent_interpolator.get_current_z(bar_position)

        tempo = self.tempo_head(z)

        # Infer used instruments
        instrument_counts_rates = self.used_in_piece_head(z)

        # Apply bar embedding to latent vector.
        bar_sin, bar_cos = self.bar_embedding(bar_position.unsqueeze(0).unsqueeze(0))

        z_bar = apply_rotary(z, bar_sin, bar_cos)

        instrument_density_logits = self.density_head(z_bar)

        return {
            "z": z.squeeze(0),
            "tempo": tempo.squeeze(0),
            "instrument_counts_rates": instrument_counts_rates.squeeze(0),
            "instrument_density_logits": instrument_density_logits.squeeze(0).squeeze(0),
        }
