from collections import defaultdict
import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba  # Assuming you have Mamba installed
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
)  # HuggingFace helper
import pretty_midi


class MusicalRotaryEmbedding:
    def __init__(self, dim, beat_lengths):
        self.beat_lengths = beat_lengths
        self.frequencies = torch.tensor(
            [2 * math.pi / b for b in self.beat_lengths]
        )  # radians per beat
        self.num_freqs = len(self.frequencies)
        self.dim = dim
        self.freqs_per_group = dim // self.num_freqs

        assert (
            self.freqs_per_group % 2 == 0
        ), "Each frequency group must have an even number of dimensions"

    def build_rotary_pos_emb(self, seq_len, device):
        """
        Builds rotary position embedding for a full sequence.

        Returns:
            Tensor of shape (1, seq_len, 2 * dim)
        """
        positions = torch.arange(seq_len, dtype=torch.float32, device=device).view(
            -1, 1
        )  # (seq_len, 1)
        angles = positions * self.frequencies.to(device)  # (seq_len, num_freqs)

        sin = angles.sin().repeat_interleave(
            self.freqs_per_group, dim=1
        )  # (seq_len, dim)
        cos = angles.cos().repeat_interleave(
            self.freqs_per_group, dim=1
        )  # (seq_len, dim)

        return sin.unsqueeze(0), cos.unsqueeze(0)  # (1, seq_len, dim)

    def rotary_pos_emb_single(self, angle):
        """
        Computes rotary embedding for a single token (streaming).

        Args:
            angle (Tensor): (dim,) radians per dimension

        Returns:
            Tensor of shape (1, 1, 2 * dim)
        """
        return angle.sin().unsqueeze(0).unsqueeze(0), angle.cos().unsqueeze(
            0
        ).unsqueeze(0)

    def get_frequencies(self):
        """
        Returns:
            Tensor of rotary frequencies in radians per beat.
        """
        return self.frequencies

    @staticmethod
    def apply_rotary(x, sin, cos):
        """
        Applies rotary embedding directly to input tensor.

        Args:
            x:   (batch, seq_len, dim)
            sin: (batch, seq_len, dim)
            cos: (batch, seq_len, dim)

        Returns:
            Rotated tensor of same shape as x.
        """
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        sin = sin[..., ::2]
        cos = cos[..., 1::2]

        rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return rotated


class TempoQuantizer:
    def __init__(self, min_bpm=30, max_bpm=240, step=5):
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.step = step
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        bpm_values = list(range(self.min_bpm, self.max_bpm + 1, self.step))
        return {bpm: i for i, bpm in enumerate(bpm_values)}

    def vocab_size(self):
        return len(self.vocab)

    def normalize(self, bpm_tensor):
        """Convert BPM → normalized float in [0, 1]."""
        return (bpm_tensor - self.min_bpm) / (self.max_bpm - self.min_bpm)

    def denormalize(self, norm_tensor):
        """Convert normalized float in [0, 1] → BPM."""
        return norm_tensor * (self.max_bpm - self.min_bpm) + self.min_bpm

    def bpm_to_token_id(self, bpm_tensor):
        bpm = torch.round(bpm / self.step) * self.step
        bpm = bpm.clamp(min=self.min_bpm, max=self.max_bpm).int()
        token_ids = [self.vocab[b.item()] for b in bpm]
        return torch.tensor(token_ids, device=bpm_tensor.device)

    def normalized_bpm_to_token_id(self, norm_tensor):
        """Convert normalized float in [0, 1] → token IDs."""
        bpm_tensor = self.denormalize(norm_tensor)
        return self.bpm_to_token_id(bpm_tensor)

    def token_id_to_bpm(self, token_ids):
        """Return BPM values corresponding to token IDs."""
        reverse_map = {v: k for k, v in self.vocab.items()}
        return [reverse_map[i] for i in token_ids]


class ControlEncoder(nn.Module):
    def __init__(self, control_vocab_size, control_embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(control_vocab_size, control_embed_dim)

    def forward(self, control_tokens):
        return self.embedding(control_tokens)  # (batch, seq_len, dim)


class LocalMIDIGenerator(nn.Module):
    def __init__(
        self,
        midi_vocab_size,
        hidden_dim,
        n_layers,
        control_embed_dim,
        beat_lengths,
        tempo_quantizer,
        max_seq_length=4096,
    ):
        super().__init__()
        self.midi_embedding = nn.Embedding(midi_vocab_size, hidden_dim)
        self.control_proj = nn.Linear(control_embed_dim, hidden_dim)
        self.tempo_quantizer = tempo_quantizer
        self.tempo_embedding = nn.Embedding(
            self.tempo_quantizer.vocab_size(), hidden_dim
        )
        self.pos_embedding = MusicalRotaryEmbedding(hidden_dim, len(beat_lengths))
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length

        self.mamba_blocks = nn.ModuleList([Mamba(hidden_dim) for _ in range(n_layers)])
        self.output_head = nn.Linear(hidden_dim, midi_vocab_size)
        self.tempo_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.rhythm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(beat_lengths)),
        )

    def forward(self, midi_tokens, control_tokens=None, tempo_token_ids=None):
        batch_size, seq_len = midi_tokens.shape
        device = midi_tokens.device

        midi_embeds = self.midi_embedding(midi_tokens)
        rotary_sin, rotary_cos = self.pos_embedding.build_rotary_pos_emb(
            seq_len, device
        )

        if control_tokens is not None:
            control_embed = self.control_proj(control_tokens)  # (batch, seq_len, dim)
            tempo_pred = self.tempo_head(control_tokens).squeeze(-1)  # (batch, seq_len)
        else:
            control_embed = None
            tempo_pred = torch.full((batch_size, seq_len), 0.5, device=device)

        if tempo_token_ids is not None:
            tempo_embeds = self.tempo_embedding(
                tempo_token_ids
            )  # (batch, seq_len, dim)
        else:
            tempo_token_ids = self.tempo_quantizer.normalized_bpm_to_token_id(
                tempo_pred
            )
            tempo_embeds = self.tempo_embedding(tempo_token_ids)

        if control_embed is not None:
            x = midi_embeds + control_embed + tempo_embeds
        else:
            x = midi_embeds + tempo_embeds

        x = self.pos_embedding.apply_rotary(x, rotary_sin, rotary_cos)

        for block in self.mamba_blocks:
            x, _ = block(x, state=None)

        logits = self.output_head(x)
        rhythm_logits = self.rhythm_head(x)

        return logits, rhythm_logits, tempo_pred

    def forward_step(
        self,
        midi_token,
        delta_time,
        control_token=None,
        past_states=None,
        previous_angle=None,
    ):
        assert midi_token.ndim == 2 and midi_token.shape[1] == 1
        device = midi_token.device

        # Infer tempo from control
        if control_token is not None:
            control_embed = self.control_proj(control_token)  # (batch, 1, dim)
            tempo_pred = self.tempo_head(control_token.squeeze(1))  # (batch,)
        else:
            control_embed = None
            tempo_pred = torch.full(
                (midi_token.shape[0],), 0.5, device=device
            )  # default tempo

        # Convert beat delta into rotary angle increment
        angle_delta = delta_time * self.pos_embedding.get_frequencies()
        angle = (previous_angle + angle_delta) % (2 * torch.pi)

        # Embeddings
        midi_embed = self.midi_embedding(midi_token)
        tempo_token_ids = self.tempo_quantizer.normalized_bpm_to_token_id(tempo_pred)
        tempo_embed = self.tempo_embedding(tempo_token_ids).unsqueeze(
            1
        )  # (batch, 1, dim)

        x = midi_embed + tempo_embed
        if control_embed is not None:
            x += control_embed

        rotary_sin, rotary_cos = self.pos_embedding.rotary_pos_emb_single(angle)
        x = self.pos_embedding.apply_rotary(x, rotary_sin, rotary_cos)

        # Forward through blocks
        new_states = []
        for i, block in enumerate(self.mamba_blocks):
            state = None if past_states is None else past_states[i]
            x, new_state = block(x, state=state)
            new_states.append(new_state)

        logits = self.output_head(x)
        rhythm_logits = self.rhythm_head(x)

        return logits, new_states, angle, rhythm_logits, tempo_pred


class FullMusicGenerator(nn.Module):
    def __init__(
        self,
        control_vocab_size,
        control_embed_dim,
        midi_vocab_size,
        hidden_dim,
        generator_layers,
        max_seq_length=4096,
    ):
        super().__init__()
        self.control_encoder = ControlEncoder(control_vocab_size, control_embed_dim)
        self.generator = LocalMIDIGenerator(
            midi_vocab_size,
            hidden_dim,
            generator_layers,
            control_embed_dim,
            max_seq_length,
        )

        # --- Internal state for streaming ---
        self.streaming_past_states = None
        self.streaming_angle = 0.0
        self.streaming_duration = 0.0
        self.streaming_segment_limit = 30.0  # seconds
        self.streaming_builder = None  # To be injected externally

    def forward(self, control_tokens, interleaved_midi_tokens):
        if control_tokens is not None:
            control_embeds = self.control_encoder(
                control_tokens
            )  # (batch, seq_len, dim)
        else:
            control_embeds = None
        return self.generator(interleaved_midi_tokens, control_embeds)

    def forward_step(
        self,
        midi_token,
        control_token=None,
        past_states=None,
        current_angle=None,
        delta_time=None,
    ):
        if control_token is not None:
            control_embed = self.control_encoder(control_token)  # (batch, 1, dim)
        else:
            control_embed = None
        return self.generator.forward_step(
            midi_token, control_embed, past_states, current_angle, delta_time
        )

    def step_streaming(
        self,
        midi_token,
        delta_time,
        control_token=None,
        tempo=None,
    ):
        """One step of real-time generation with streaming state and auto reset."""
        if self.streaming_builder is None:
            raise RuntimeError(
                "You must assign .streaming_builder before calling step_streaming()"
            )

        # Reset if needed
        emitted_notes = []
        if self.streaming_duration >= self.streaming_segment_limit:
            self.streaming_builder.fade_out()
            self.streaming_builder.reset()
            self.streaming_duration = 0.0
            self.streaming_past_states = None
            self.streaming_angle = 0.0

        # Perform generation step
        logits, new_states, new_angle, delta_pred, tempo_pred = self.forward_step(
            midi_token,
            control_token,
            self.streaming_past_states,
            self.streaming_angle,
            delta_time,
        )

        self.streaming_past_states = new_states
        self.streaming_angle = new_angle

        emitted_notes = []
        if tempo is not None:
            seconds_per_beat = 60.0 / tempo
            self.streaming_duration += delta_time * seconds_per_beat
            self.streaming_builder.step(
                token_id=midi_token.item(),  # assuming shape (1, 1)
                delta_time=delta_time,
                tempo=tempo,
            )
            emitted_notes = self.streaming_builder.emit_if_ready()

        return logits, delta_pred, tempo_pred, emitted_notes


class StreamingMidiBuilder:
    def __init__(self, tokenizer, velocity_bins=32, max_duration=300.0):
        self.tokenizer = tokenizer
        self.current_beat = 0.0
        self.current_program = 0
        self.current_velocity = {}
        self.instruments = {}
        self.active_notes = {}
        self.velocity_bins = velocity_bins
        self.generated_duration = 0.0
        self.max_duration = max_duration
        self.segment_id = 0  # NEW

    def step(self, token_id: int, delta_time: float, tempo: float):
        self.seconds_per_beat = 60.0 / tempo
        self.generated_duration += delta_time * self.seconds_per_beat
        self.current_beat += delta_time
        self.current_time = (
            self.current_beat * 60.0 / tempo
        )  # timestamps remain monotonic
        token = self.tokenizer.id2event.get(token_id, "UNK")

        if token.startswith("Track_"):
            self.current_program = int(token[len("Track_") :])
            if self.current_program not in self.instruments:
                inst = pretty_midi.Instrument(
                    program=self.current_program if self.current_program < 128 else 0,
                    is_drum=(self.current_program == 128),
                )
                self.instruments[self.current_program] = inst
                self.active_notes[self.current_program] = {}
                self.current_velocity[self.current_program] = 64

        elif token.startswith("Velocity_"):
            v_bin = int(token[len("Velocity_") :])
            velocity = int((v_bin + 0.5) * (128 / self.velocity_bins))
            self.current_velocity[self.current_program] = velocity

        elif token.startswith("Note-On_"):
            pitch = int(token[len("Note-On_") :])
            self.active_notes[self.current_program][pitch] = (
                self.current_time,
                self.current_velocity.get(self.current_program, 64),
            )

        elif token.startswith("Note-Off_"):
            pitch = int(token[len("Note-Off_") :])
            if pitch in self.active_notes[self.current_program]:
                start_time, velocity = self.active_notes[self.current_program].pop(
                    pitch
                )
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=start_time,
                    end=self.current_time,
                )
                self.instruments[self.current_program].notes.append(note)

    def fade_out(self, fade_time=2.0):
        """Apply a fade-out by shortening or zeroing final notes."""
        for inst in self.instruments.values():
            for note in inst.notes:
                note_end = min(note.end, note.start + fade_time)
                note.end = note_end
                note.velocity = int(note.velocity * 0.5)  # simple fade down

    def reset(self):
        self.current_beat = 0.0
        self.generated_duration = 0.0
        self.current_program = 0
        self.current_velocity = {}
        self.instruments = {}
        self.active_notes = {}
        self.segment_id += 1  # NEW
        self.pending_padding_beats = 0.25

    def emit_if_ready(self):
        emitted = []
        for program, instrument in self.instruments.items():
            for note in instrument.notes:
                emitted.append(
                    {
                        "note": note,
                        "program": program,
                        "segment_id": self.segment_id,  # NEW
                    }
                )
            instrument.notes.clear()
        return emitted


class RhythmQuantizer:
    def __init__(self, bins=None):
        # Default: list of (beat_value, human_label)
        default_bins = [
            # Binary values
            (4.0, "whole"),
            (2.0, "half"),
            (1.0, "quarter"),
            (0.5, "eighth"),
            (0.25, "sixteenth"),
            (0.125, "thirty_second"),
            # Dotted values
            (3.0, "dotted_half"),
            (1.5, "dotted_quarter"),
            (0.75, "dotted_eighth"),
            (0.375, "dotted_sixteenth"),
            (0.1875, "dotted_thirty_second"),
            # Triplets
            (2 / 3, "half_triplet"),
            (1 / 3, "quarter_triplet"),
            (1 / 6, "eighth_triplet"),
            # Quintuplets
            (0.8, "half_quintuplet"),
            (0.4, "quarter_quintuplet"),
            (0.2, "eighth_quintuplet"),
            # Septuplets
            (4 / 7, "half_septuplet"),  # ≈ 0.571
            (2 / 7, "quarter_septuplet"),  # ≈ 0.286
            (1 / 7, "eighth_septuplet"),  # ≈ 0.143
            # Rest / no-op
            (0.0, "null"),
        ]

        self.bins = bins or default_bins
        self.values = [b[0] for b in self.bins]
        self.labels = [b[1] for b in self.bins]
        self.value_to_idx = {round(v, 6): i for i, v in enumerate(self.values)}
        self.idx_to_value = {i: v for i, v in enumerate(self.values)}
        self.idx_to_label = {i: l for i, l in enumerate(self.labels)}

    def quantize(self, beat_duration):
        diffs = [abs(beat_duration - b) for b in self.values]
        best_idx = diffs.index(min(diffs))
        return self.values[best_idx], best_idx, self.labels[best_idx]

    def quantize_sequence(
        self, beat_durations: List[float]
    ) -> Tuple[List[float], List[int], List[str]]:
        """
        Quantize a sequence of beat durations using dynamic programming,
        preserving total duration and minimizing total snapping error.

        Returns:
            Tuple of:
                - snapped durations (List[float])
                - class indices (List[int])
                - human-readable labels (List[str])
        """
        n = len(beat_durations)
        total = round(sum(beat_durations), 6)

        # dp[pos][accumulated_sum] = (total_error, path_of_values)
        dp = [defaultdict(lambda: (float("inf"), [])) for _ in range(n + 1)]
        dp[0][0.0] = (0.0, [])

        for i in range(n):
            for acc_sum, (err, path) in dp[i].items():
                for val in self.values:
                    new_sum = round(acc_sum + val, 6)
                    new_err = err + abs(val - beat_durations[i])
                    if new_err < dp[i + 1][new_sum][0]:
                        dp[i + 1][new_sum] = (new_err, path + [val])

        if total not in dp[n]:
            raise ValueError("No valid quantization path found.")

        best_path = dp[n][total][1]
        indices = [self.value_to_idx[round(v, 6)] for v in best_path]
        labels = [self.idx_to_label[i] for i in indices]

        return best_path, indices, labels

    def decode_class(self, class_idx):
        return self.idx_to_value.get(class_idx)

    def class_label(self, class_idx):
        return self.idx_to_label.get(class_idx, "unknown")

    def num_classes(self):
        return len(self.values)


class MultiTrackMidiTokenizer:
    def __init__(
        self, rhythm_quantizer, tempo_quantizer, velocity_bins=32, include_drums=True
    ):
        self.velocity_bins = velocity_bins
        self.include_drums = include_drums
        self.event2id = {}
        self.id2event = {}

        self.rhythm_quantizer = rhythm_quantizer
        self.tempo_quantizer = tempo_quantizer

        self.build_vocab()

    def build_vocab(self):
        events = set()
        for pitch in range(128):
            events.add(f"Note-On_{pitch}")
            events.add(f"Note-Off_{pitch}")
        for v in range(self.velocity_bins):
            events.add(f"Velocity_{v}")
        for program in range(129):  # 0–127 = GM, 128 = drums
            events.add(f"Track_{program}")
        self.event2id = {e: idx for idx, e in enumerate(sorted(events))}
        self.id2event = {idx: e for e, idx in self.event2id.items()}

    def encode(self, midi_file_path):
        pm = pretty_midi.PrettyMIDI(midi_file_path)
        beat_times = pm.get_beats()
        ts_changes = pm.time_signature_changes
        beats_per_bar = ts_changes[0].numerator if ts_changes else 4
        bar_starts = beat_times[::beats_per_bar]
        last_bar_time = bar_starts[-1] if bar_starts else beat_times[-1]

        # Trim notes that extend past the final bar
        for instrument in pm.instruments:
            instrument.notes = [n for n in instrument.notes if n.start < last_bar_time]
            for note in instrument.notes:
                if note.end > last_bar_time:
                    note.end = last_bar_time

        # Get tempo changes
        tempo_times, tempi = pm.get_tempo_changes()

        def tempo_at_beat(beat):
            time = pm.get_beats()[0] + beat * 60.0 / tempi[0]  # rough start
            for i in range(len(tempo_times) - 1):
                if tempo_times[i] <= time < tempo_times[i + 1]:
                    return tempi[i]
            return tempi[-1]

        # Collect timed events
        event_list = []
        for instrument in pm.instruments:
            if not self.include_drums and instrument.is_drum:
                continue
            program = 128 if instrument.is_drum else instrument.program
            event_list.append((0.0, f"Track_{program}"))
            for note in instrument.notes:
                start = pm.time_to_beat(note.start)
                end = pm.time_to_beat(note.end)
                velocity_bin = int(note.velocity / (128 / self.velocity_bins))
                event_list.append((start, f"Note-On_{note.pitch}"))
                event_list.append((start, f"Velocity_{velocity_bin}"))
                event_list.append((end, f"Note-Off_{note.pitch}"))

        # Ensure the sequence ends exactly on the bar boundary
        last_beat_time = max([t for t, _ in event_list], default=0.0)
        final_beat_target = pm.time_to_beat(last_bar_time)
        if last_beat_time < final_beat_target:
            event_list.append((final_beat_target, "DUMMY_END"))

        # Sort events by beat time
        event_list.sort(key=lambda x: x[0])

        # Extract deltas, tokens, tempo
        token_ids = []
        tempo_token_ids = []
        beat_deltas = []
        last_time = 0.0

        for t, event in event_list:
            dt = t - last_time
            last_time = t
            beat_deltas.append(dt)

            if event in self.event2id:
                token_ids.append(self.event2id[event])
                bpm = tempo_at_beat(t)
                _, tempo_token = self.tempo_quantizer.quantize(bpm)
                tempo_token_ids.append(tempo_token)

            elif event == "DUMMY_END":
                pass  # no token or tempo

        # Quantize the beat deltas
        beat_deltas, rhythm_classes, _ = self.rhythm_quantizer.quantize_sequence(
            beat_deltas
        )

        return (
            torch.tensor(token_ids).unsqueeze(0),
            torch.tensor(rhythm_classes[: len(token_ids)]).unsqueeze(0),
            torch.tensor(beat_deltas[: len(token_ids)]).unsqueeze(0),
        )

    def vocab_size(self):
        return len(self.event2id)


def sample_top_p(logits, top_p=0.9, temperature=1.0):
    """
    Sample from logits using top-p (nucleus) sampling with temperature scaling.

    Args:
        logits (torch.Tensor): shape (1, 1, vocab_size)
        top_p (float): cumulative probability threshold (e.g., 0.9)
        temperature (float): scaling factor for logits (e.g., 1.0 = neutral, <1 = more confident)

    Returns:
        int: sampled token ID
    """
    logits = logits[0, 0] / temperature  # shape (vocab,)
    probs = F.softmax(logits, dim=-1)

    # Sort tokens by probability
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Keep only tokens with cumulative probability <= top_p
    cutoff = cumulative_probs > top_p
    cutoff_idx = torch.argmax(cutoff).item() + 1 if cutoff.any() else len(probs)
    filtered_probs = sorted_probs[:cutoff_idx]
    filtered_indices = sorted_indices[:cutoff_idx]

    # Re-normalize and sample
    filtered_probs /= filtered_probs.sum()
    next_token_id = filtered_indices[torch.multinomial(filtered_probs, 1)].item()
    return next_token_id


# Assume these are already defined
tokenizer = MultiTrackMidiTokenizer()
builder = StreamingMidiBuilder(tokenizer)
generator = FullMusicGenerator(
    control_vocab_size=10,
    control_embed_dim=32,
    midi_vocab_size=tokenizer.vocab_size(),
    hidden_dim=128,
    generator_layers=4,
)
generator.streaming_builder = builder
generator.streaming_segment_limit = 5.0  # shorter for test

# Starting token (no idea what this should be)
start_token = tokenizer.event2id["Track_0"]
current_token = torch.tensor([[start_token]])  # (1, 1)

# Optional control token (keep fixed or change over time)
control_token = torch.tensor([[1]])

# Init loop state
delta_time = 0.0  # First step is usually zero delay
tempo = None
steps = 100  # How many tokens to generate

for i in range(steps):
    logits, delta_pred, tempo_pred, emitted_notes = generator.step_streaming(
        midi_token=current_token,
        delta_time=delta_time,
        control_token=control_token,
        tempo=tempo,
    )

    # --- Sample next token from logits ---
    next_token_id = sample_top_p(logits, top_p=0.95, temperature=1.0)
    current_token = torch.tensor([[next_token_id]])

    # --- Use predicted values for next step ---
    delta_time = max(delta_pred.item(), 0.0)  # in beats
    tempo = max(tempo_pred.item(), 30.0)  # safety clamp

    # --- Emit notes (already faded + segmented) ---
    for note_info in emitted_notes:
        note = note_info["note"]
        program = note_info["program"]
        segment_id = note_info["segment_id"]
        print(
            f"[Segment {segment_id}] Program {program} → Note {note.pitch} "
            f"({note.start:.2f}s - {note.end:.2f}s, vel={note.velocity})"
        )

    sleep(0.1)  # Simulate real-time token step
