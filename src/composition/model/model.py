from composition.midi.tokenizer import DRUMS_PROGRAM_ID
from composition.model.control_encoder import ControlEncoder
from composition.model.instrument_count_head import InstrumentCountHead
from composition.model.latent_interpolator import LatentInterpolator
from composition.model.latent_prior_net import LatentPriorNet

from composition.model.time_aware_embedding import (
    DualRotartyEmbedding,
    ExponentialRotaryEmbedding,
    apply_rotary,
)

from composition.model.utils import (
    build_cross_attention_context,
    compute_bar_summaries,
    gather_token_context,
    gather_token_meta,
    prepare_bar_offsets,
    sample_z_sequence_from_tempo_changes,
)

import torch
import torch.nn as nn
from mamba_ssm import Mamba2


class LocalMIDIGenerator(nn.Module):
    def __init__(
        self,
        midi_vocab_size,
        velocity_vocab_size,
        hidden_dim,
        latent_dim,
        n_layers,
        control_embed_dim,
        control_vocab_size,
        num_instruments=129,  # 128 GM + 1 for drums
        max_instrument_instances=10,
    ):
        super().__init__()
        self.num_instruments = num_instruments
        self.max_instrument_instances = max_instrument_instances
        self.velocity_vocab_size = velocity_vocab_size

        self.control_encoder = ControlEncoder(
            control_vocab_size, control_embed_dim, latent_dim
        )
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.latent_prior_net = LatentPriorNet(latent_dim)

        self.gm_midi_embedding = nn.Embedding(midi_vocab_size, hidden_dim)
        self.drums_midi_embedding = nn.Embedding(midi_vocab_size, hidden_dim)
        self.velocity_embedding = nn.Embedding(velocity_vocab_size, hidden_dim)
        self.pos_embedding = DualRotartyEmbedding(hidden_dim, base=10000.0)
        self.bar_embedding = ExponentialRotaryEmbedding(latent_dim, base=512.0)
        self.instrument_embedding = nn.Embedding(num_instruments, hidden_dim)

        self.duration_log1p_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),  # or GELU / ReLU
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.hidden_dim = hidden_dim

        self.mamba_block = nn.ModuleList(
            [Mamba2(d_model=hidden_dim) for _ in range(n_layers)]
        )
        self.output_head = nn.Linear(hidden_dim, midi_vocab_size)

        self.rhythm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # output log(delta_beats)
        )

        self.duration_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # output log(duration_beats)
        )

        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, velocity_vocab_size),
        )

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
        track_grouped_input,  # list of dict of batches
        bar_positions,  # bar position tensor (batch, total_bars)
        control_token=None,  # Control token tensor (batch,)
        tempos=None,  # tempo values tensor (batch, total_bars) for each instrument
    ):
        # bar_positions = beat_positions.floor()  # (batch, seq_len)
        # SAMPLE TEMPOS, BAR POS AND INST PROBS ON GLOBAL GRID FOR LOSS, make sure that global tempo grid as same accross the batch dim
        # THIS GRID IS SAMPLED AT EVERY BAR - GENIUS, MAKE SURE BAR POSITIONS START OT 0???
        # tempos sampled on instruments grids go to instrument heads and z sequencing happens again for each instrument (why? not needed there)

        # generate latent and derived latent parameters (tempo, unstruments used, instrument activation).
        # think about adding more for auxiliary loss and latent training

        # check batching logic here, probably we need to expand it to batch dim if there's no control input
        mu, logvar = self.latent_prior_net(
            (
                self.control_encoder(control_token)
                if control_token is not None
                else None
            ),
        )

        if tempos is not None:
            z_sequence = sample_z_sequence_from_tempo_changes(mu, logvar, tempos)
        else:
            # Sample z_sequence from prior
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z_sequence = mu + eps * std
            z_sequence = z_sequence.unsqueeze(1).expand(-1, bar_positions.shape[1], -1)

        # TRAIN ONLY LATENT USING AUX OUTPUT AS A FIRST STEP

        # THEN START 100% TEACHER FORCING AND GRADUALLY SHIFT TO GENERATIVE LOSS WITH PERCEPTUAL LOSS

        # Infer tempro from global latent vector
        tempos_pred = self.tempo_head(z_sequence).squeeze(-1)  # (batch, total_bars)

        # Infer used instruments
        instruments_counts_logits = self.used_in_piece_head(z_sequence)

        # during inference build instance mask to maks activation

        # Apply bar embedding to latent vector. The total bar is sequence of unique bars
        bar_sin, bar_cos = self.bar_embedding(
            bar_positions
        )  # (batch, total_bars, latent_dim)

        z_bar = apply_rotary(z_sequence, bar_sin, bar_cos)

        # Instruments conditioned on bar-modulated latent vector
        instrument_activation_logits = self.activation_head(
            z_bar
        )  # (batch, total_bars, num_instruments * max_instrument_instances)

        # USE HARD MASKING Of INSTRUMENTS BY UNSTRUMENTS_USED DURING INFERENCE
        # USE CONSISTENCY LOSS TO JOITNLY TRAIN USED INST AND ACTIVATION
        # ADD WEIGHT TO INSTRUMENT ACTIVATION LOSS FROM INSTRUMENTS_USED

        # project latent to hidden
        z_proj = self.latent_proj(z_sequence)  # (batch, total_bars, hidden_dim)

        bar_idx_map = {bar: i for i, bar in enumerate(bar_positions)}

        global_max_bar = max(
            batches["bar_positions"].max().item() for batches in track_grouped_input
        )

        bar_cache = []
        program_ids = []
        instrument_inputs = []
        for batches in track_grouped_input:
            program_id = batches["program_id"]
            pitch_tokens = batches["pitch_tokens"]
            velocity_tokens = batches["velocity_tokens"]
            beat_positions = batches["beat_positions"]
            bar_positions_inst = batches["bar_positions"]
            within_bar_positions = batches["within_bar_positions"]
            note_durations_beats = batches["note_durations_beats"]

            note_durations_beats_log1p = torch.log1p(note_durations_beats)

            midi_embeds = (
                self.gm_midi_embedding(pitch_tokens)
                if program_id != DRUMS_PROGRAM_ID
                else self.drums_midi_embedding(pitch_tokens)
            )

            velocity_embeds = self.velocity_embedding(velocity_tokens)
            duration_embeds = self.duration_log1p_proj(note_durations_beats_log1p)
            instrument_embeds = self.instrument_embedding(program_id)

            indices = torch.tensor(
                [bar_idx_map[b] for b in bar_positions_inst], device=z_proj.device
            )

            # Step 2: Expand indices to match batch dim
            # indices: (inst_len,) → (batch, inst_len)
            indices = indices.unsqueeze(0).expand(
                z_proj.size(0), -1
            )  # shape: (batch, inst_len)

            # Step 3: Gather along bars axis (axis 1)
            z_inst = torch.gather(
                z_proj,
                dim=1,
                index=indices.unsqueeze(-1).expand(-1, -1, z_proj.size(2)),
            )
            # z_inst shape: (batch, inst_len, hidden_dim)

            x = (
                z_inst
                + midi_embeds
                + velocity_embeds
                + duration_embeds
                + instrument_embeds
            )

            instrument_inputs.append(x)

            summaries, _ = compute_bar_summaries(x, bar_positions, global_max_bar)
            bar_cache.append(summaries)  # (B, num_bars, H)
            program_ids.append(program_id)

        ######## FROM THIS POINT WE PASS ON TO INSTRUMENTAL HEADS
        outputs = []
        for program_id, inputs, batches in zip(
            program_ids, instrument_inputs, track_grouped_input
        ):
            pitch_tokens = batches["pitch_tokens"]
            velocity_tokens = batches["velocity_tokens"]
            beat_positions = batches["beat_positions"]
            bar_positions_inst = batches["bar_positions"]
            within_bar_positions = batches["within_bar_positions"]
            note_durations_beats = batches["note_durations_beats"]

            context_depth = 4
            context, inst_ids = build_cross_attention_context(
                full_bar_cache=bar_cache,
                program_ids=program_ids,
                context_depth=context_depth,
                self_index=program_id,
            )

            # Optional: add relative bar offsets if you want
            bar_offsets = prepare_bar_offsets(
                context_tensor=context, context_depth=context_depth
            )

            # Gather per-token aligned slices
            context_per_token = gather_token_context(context, bar_positions)
            inst_ids_per_token = gather_token_meta(inst_ids, bar_positions)
            bar_offsets_per_token = gather_token_meta(bar_offsets, bar_positions)

            # Flatten and apply attention
            B, T, ctx, H = context_per_token.shape
            x = x + self.cross_attn(
                query=x.view(B * T, 1, H),
                context=context_per_token.view(B * T, ctx, H),
                instrument_ids=inst_ids_per_token.view(B * T, ctx),
                bar_offsets=bar_offsets_per_token.view(B * T, ctx),
            ).view(B, T, H)

            rotary_sin, rotary_cos = self.pos_embedding.forward(
                within_bar_positions, beat_positions
            )
            x = apply_rotary(inputs, rotary_sin, rotary_cos)

            for block in self.mamba_block:
                x, _ = block(x, state=None)

            midi_logits = self.output_head(x)
            log1p_delta_beats_pred = self.rhythm_head(x)
            log1p_duration_beats_pred = self.duration_head(x)
            velocity_logits = self.velocity_head(x)

            outputs.append(
                {
                    "program_id": program_id,
                    "midi_logits": midi_logits,
                    "velocity_logits": velocity_logits,
                    "delta_beats_log1p": log1p_delta_beats_pred,
                    "duration_beats_log1p": log1p_duration_beats_pred,
                }
            )

        return (
            outputs,
            tempos_pred,
            instruments_counts_logits,
            instrument_activation_logits,
        )

    # def forward_step(
    #     self,
    #     midi_token,
    #     beat_position,
    #     bar_position,
    #     within_bar_position,
    #     control_token=None,
    #     past_states=None,
    # ):
    #     assert midi_token.ndim == 2 and midi_token.shape[1] == 1

    #     mu, logvar = self.latent_prior_net(
    #         self.control_encoder(control_token) if control_token is not None else None,
    #     )  # (batch, 1, latent_dim)

    #     control_changed = False
    #     if self.last_control_hash is None:
    #         control_changed = True
    #     elif control_token is not None:
    #         control_hash = self.control_hash(control_token)
    #         if self.last_control_hash != control_hash:
    #             control_changed = True
    #             self.last_control_hash = control_hash

    #     if control_changed:
    #         # Update latent interpolator with new control token
    #         std = torch.exp(0.5 * logvar)
    #         eps = torch.randn_like(std)
    #         z = mu + eps * std  # (batch, 1, latent_dim)
    #         self.latent_interpolator.update(z, bar_position)

    #     z = self.latent_interpolator.get_current_z(bar_position)

    #     tempo_pred = self.tempo_head(z).squeeze(-1)  # (batch, 1)

    #     # Apply bar embedding to latent vector
    #     bar_sin, bar_cos = self.bar_embedding(bar_position)  # (batch, 1, latent_dim)
    #     z_bar = apply_rotary(z, bar_sin, bar_cos)
    #     z_proj = self.latent_proj(z_bar)  # (batch, 1, hidden_dim)

    #     # Embeddings
    #     midi_embed = self.midi_embedding(midi_token)

    #     x = midi_embed + z_proj  # (batch, 1, hidden_dim)
    #     rotary_sin, rotary_cos = self.pos_embedding.forward(
    #         within_bar_position, beat_position
    #     )
    #     x = apply_rotary(x, rotary_sin, rotary_cos)

    #     # Forward through blocks
    #     new_states = []
    #     for i, block in enumerate(self.mamba_blocks):
    #         state = None if past_states is None else past_states[i]
    #         x, new_state = block(x, state=state)
    #         new_states.append(new_state)

    #     logits = self.output_head(x)
    #     log_delta_beats_pred = self.rhythm_head(x)
    #     log_duration_beats_pred = self.duration_head(x)
    #     delta_beats = torch.exp(log_delta_beats_pred)
    #     duration_beats = torch.exp(log_duration_beats_pred)

    #     return logits, new_states, delta_beats, duration_beats, tempo_pred


# class FullMusicGenerator(nn.Module):
#     def __init__(
#         self,
#         control_vocab_size,
#         control_embed_dim,
#         midi_vocab_size,
#         hidden_dim,
#         generator_layers,
#         ticks_per_beat,
#         time_signature=Fraction(4, 4),
#         max_seq_length=4096,
#     ):
#         super().__init__()
#         self.control_encoder = ControlEncoder(control_vocab_size, control_embed_dim)
#         self.generator = LocalMIDIGenerator(
#             midi_vocab_size,
#             hidden_dim,
#             generator_layers,
#             control_embed_dim,
#             ticks_per_beat,
#             max_seq_length,
#         )

#         # --- Internal state for streaming ---
#         self.streaming_past_states = None
#         self.streaming_beats_position = 0.0
#         self.streaming_bar_position = 0.0
#         self.streaming_duration = 0.0
#         self.streaming_ticks_position = 0.0
#         self.streaming_segment_limit = 30.0  # seconds
#         self.streaming_builder = None  # To be injected externally
#         self.ticks_per_beat = ticks_per_beat

#         # e.g. 4/4 = 4 beats per bar, 1 bar = 4 beats
#         # e.g. 7/8 = 7 half beats per bar, 1 bar = 7 half beats = 3.5 beats
#         self.bar_length = 4 * time_signature.numerator / time_signature.denominator

#     def forward(self, control_tokens, interleaved_midi_tokens):
#         if control_tokens is not None:
#             control_embeds = self.control_encoder(
#                 control_tokens
#             )  # (batch, seq_len, dim)
#         else:
#             control_embeds = None
#         return self.generator(interleaved_midi_tokens, control_embeds)

#     def forward_step(
#         self,
#         midi_token,
#         beat_position,
#         bar_position,
#         control_token=None,
#         past_states=None,
#     ):
#         if control_token is not None:
#             control_embed = self.control_encoder(control_token)  # (batch, 1, dim)
#         else:
#             control_embed = None
#         return self.generator.forward_step(
#             midi_token, beat_position, bar_position, control_embed, past_states
#         )

#     def step_streaming(
#         self,
#         midi_token,
#         delta_ticks,
#         control_token=None,
#         tempo_bpm=None,
#     ):
#         """One step of real-time generation with streaming state and auto reset."""
#         if self.streaming_builder is None:
#             raise RuntimeError(
#                 "You must assign .streaming_builder before calling step_streaming()"
#             )

#         # Reset if needed
#         emitted_notes = []
#         if self.streaming_duration >= self.streaming_segment_limit:
#             self.streaming_builder.fade_out()
#             self.streaming_builder.reset()
#             self.streaming_duration = 0.0
#             self.streaming_beats_position = 0.0
#             self.streaming_bar_position = 0.0
#             self.streaming_ticks_position = 0
#             self.streaming_past_states = None

#         # Perform generation step
#         logits, new_states, ticks_logits, tempo_pred = self.forward_step(
#             midi_token,
#             self.streaming_beats_position,
#             self.streaming_bar_position,
#             control_token,
#             self.streaming_past_states,
#         )

#         self.streaming_past_states = new_states
#         self.streaming_ticks_position += delta_ticks
#         self.streaming_beats_position = (
#             self.streaming_ticks_position / self.ticks_per_beat
#         )
#         self.streaming_bar_position = (
#             self.streaming_beats_position % self.bar_length
#         ) / self.bar_length

#         emitted_notes = []
#         if tempo_bpm is not None:
#             seconds_per_beat = 60.0 / tempo_bpm
#             self.streaming_duration += (
#                 delta_ticks * seconds_per_beat / self.ticks_per_beat
#             )
#             self.streaming_builder.step(
#                 token_id=midi_token.item(),  # assuming shape (1, 1)
#                 delta_beats=self.streaming_ticks_position / self.ticks_per_beat,
#                 tempo=tempo_bpm,
#             )
#             emitted_notes = self.streaming_builder.emit_if_ready()

#         return logits, ticks_logits, delta_pred, tempo_pred, emitted_notes


# def sample_top_p(logits, top_p=0.9, temperature=1.0):
#     """
#     Sample from logits using top-p (nucleus) sampling with temperature scaling.

#     Args:
#         logits (torch.Tensor): shape (1, 1, vocab_size)
#         top_p (float): cumulative probability threshold (e.g., 0.9)
#         temperature (float): scaling factor for logits (e.g., 1.0 = neutral, <1 = more confident)

#     Returns:
#         int: sampled token ID
#     """
#     logits = logits[0, 0] / temperature  # shape (vocab,)
#     probs = F.softmax(logits, dim=-1)

#     # Sort tokens by probability
#     sorted_probs, sorted_indices = torch.sort(probs, descending=True)
#     cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

#     # Keep only tokens with cumulative probability <= top_p
#     cutoff = cumulative_probs > top_p
#     cutoff_idx = torch.argmax(cutoff).item() + 1 if cutoff.any() else len(probs)
#     filtered_probs = sorted_probs[:cutoff_idx]
#     filtered_indices = sorted_indices[:cutoff_idx]

#     # Re-normalize and sample
#     filtered_probs /= filtered_probs.sum()
#     next_token_id = filtered_indices[torch.multinomial(filtered_probs, 1)].item()
#     return next_token_id


# # Assume these are already defined
# tickes_per_beat = 480
# rhythm_quantizer = RhythmQuantizer(ticks_per_beat=tickes_per_beat)
# tokenizer = MultiTrackMidiTokenizer(rhythm_quantizer=rhythm_quantizer)
# builder = StreamingMidiBuilder(tokenizer)
# generator = FullMusicGenerator(
#     control_vocab_size=10,
#     control_embed_dim=32,
#     midi_vocab_size=tokenizer.vocab_size(),
#     hidden_dim=128,
#     generator_layers=4,
#     ticks_per_beat=tickes_per_beat,
# )
# generator.streaming_builder = builder
# generator.streaming_segment_limit = 5.0  # shorter for test

# # Starting token (no idea what this should be)
# start_token = tokenizer.event2id["Track_0"]
# current_token = torch.tensor([[start_token]])  # (1, 1)

# # Optional control token (keep fixed or change over time)
# control_token = torch.tensor([[1]])

# # Init loop state
# delta_ticks = 0  # First step is usually zero delay
# tempo = None
# steps = 100  # How many tokens to generate

# for i in range(steps):
#     logits, ticks_logits, delta_pred, tempo_pred, emitted_notes = (
#         generator.step_streaming(
#             midi_token=current_token,
#             delta_ticks=delta_ticks,
#             control_token=control_token,
#             tempo=TempoNormalizer().unnormalize_bpm(tempo),
#         )
#     )

#     # --- Sample next token from logits ---
#     next_token_id = sample_top_p(logits, top_p=0.95, temperature=1.0)
#     current_token = torch.tensor([[next_token_id]])

#     # --- Use predicted values for next step ---
#     delta_ticks = ticks_logits.argmax(dim=-1).item()  # (1, 1) → (1,)
#     tempo = tempo_pred.squeeze(0).item()  # (1, 1) → (1,)

#     # --- Emit notes (already faded + segmented) ---
#     for note_info in emitted_notes:
#         note = note_info["note"]
#         program = note_info["program"]
#         segment_id = note_info["segment_id"]
#         print(
#             f"[Segment {segment_id}] Program {program} → Note {note.pitch} "
#             f"({note.start:.2f}s - {note.end:.2f}s, vel={note.velocity})"
#         )

#     sleep(0.1)  # Simulate real-time token step
