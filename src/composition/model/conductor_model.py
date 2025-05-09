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
        control_vocab_size,
        num_instruments=129,  # 128 GM + 1 for drums
        max_instrument_instances=10,
    ):
        super().__init__()
        self.num_instruments = num_instruments
        self.max_instrument_instances = max_instrument_instances

        self.control_encoder = ControlEncoder(
            control_vocab_size, control_embed_dim, latent_dim
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
        model_inputs,  # Model inputs (batch, seq_len, input_dim)
        control_tokens=None,  # Control token tensor (batch,)
    ):
        # bar_positions = beat_positions.floor()  # (batch, seq_len)
        # SAMPLE TEMPOS, BAR POS AND INST PROBS ON GLOBAL GRID FOR LOSS, make sure that global tempo grid as same accross the batch dim
        # THIS GRID IS SAMPLED AT EVERY BAR - GENIUS, MAKE SURE BAR POSITIONS START OT 0???
        # tempos sampled on instruments grids go to instrument heads and z sequencing happens again for each instrument (why? not needed there)

        # generate latent and derived latent parameters (tempo, unstruments used, instrument activation).
        # think about adding more for auxiliary loss and latent training

        batch_size = len(model_inputs["bar_tempos"])
        if control_tokens is not None:
            mu, logvar = self.latent_prior_net(self.control_encoder(control_tokens))
        else:
            # If no control tokens, use default prior
            mu, logvar = self.latent_prior_net()
            mu = mu.repeat(batch_size, 1)
            logvar = logvar.repeat(batch_size, 1)

        print("shape of bar tempos:", model_inputs["bar_tempos"].shape)

        z_sequence = sample_z_sequence_from_tempo_changes(
            mu, logvar, model_inputs["bar_tempos"]
        )
        # TRAIN ONLY LATENT USING AUX OUTPUT AS A FIRST STEP

        # THEN START 100% TEACHER FORCING AND GRADUALLY SHIFT TO GENERATIVE LOSS WITH PERCEPTUAL LOSS

        # Infer tempro from global latent vector

        print("shape of z_sequence:", z_sequence.shape)
        tempos_pred = self.tempo_head(z_sequence).squeeze(-1)  # (batch, total_bars)

        # Infer used instruments
        instruments_counts, instruments_counts_logits = self.used_in_piece_head(
            z_sequence.mean(dim=1)
        )

        print("inst counts logits shape: ", instruments_counts_logits.shape)
        print("inst counts shape: ", instruments_counts.shape)

        print("track mask shape: ", model_inputs["track_mask"].shape)

        # during inference build instance mask to maks activation

        # Apply bar embedding to latent vector. The total bar is sequence of unique bars
        bar_sin, bar_cos = self.bar_embedding(
            model_inputs["bar_boundaries"]
        )  # (batch, total_bars, latent_dim)

        z_bar = apply_rotary(z_sequence, bar_sin, bar_cos)

        # Instruments conditioned on bar-modulated latent vector
        instrument_activation_logits = self.activation_head(
            z_bar
        )  # (batch, total_bars, num_instruments * max_instrument_instances)

        # USE HARD MASKING Of INSTRUMENTS BY UNSTRUMENTS_USED DURING INFERENCE
        # USE CONSISTENCY LOSS TO JOITNLY TRAIN USED INST AND ACTIVATION
        # ADD WEIGHT TO INSTRUMENT ACTIVATION LOSS FROM INSTRUMENTS_USED

        return {
            "tempos": tempos_pred,
            "instrument_counts_logits": instruments_counts_logits,
            "instrument_activation_logits": instrument_activation_logits,
            "latent_mu": mu,
            "latent_logvar": logvar,
        }
