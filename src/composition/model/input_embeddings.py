from src.composition.midi.tokenizer import DRUMS_PROGRAM_ID
import torch.nn as nn
import torch


class InputEmbeddings(nn.Module):
    def __init__(self, hidden_dim, pitch_vocab_size, velocity_vocab_size, num_instruments):
        super().__init__()
        self.gm_embedding = nn.Embedding(pitch_vocab_size, hidden_dim)
        self.drums_embedding = nn.Embedding(pitch_vocab_size, hidden_dim)
        self.velocity_embedding = nn.Embedding(velocity_vocab_size, hidden_dim)
        self.instrument_embedding = nn.Embedding(num_instruments, hidden_dim)

        self.duration_log1p_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),  # or GELU / ReLU
            nn.Linear(hidden_dim, hidden_dim),
        )

    def get_instrument_embedding(self, program_ids):
        return self.instrument_embedding(program_ids)

    def forward(self, model_inputs):
        program_id = model_inputs["program_ids"]
        pitch_tokens = model_inputs["pitch_tokens"]
        velocity_tokens = model_inputs["velocity_tokens"]
        note_durations_beats = model_inputs["note_durations_beats"]
        attention_mask = model_inputs["attention_mask"]

        is_drum = program_id == DRUMS_PROGRAM_ID

        B, P, T = pitch_tokens.shape
        D = self.gm_embedding.embedding_dim
        midi_embeds = torch.empty(B, P, T, D, device=pitch_tokens.device, dtype=self.gm_embedding.weight.dtype)

        # Non-drums
        gm_mask = ~is_drum
        if gm_mask.any():
            gm_emb = self.gm_embedding(pitch_tokens[gm_mask])  # -> [N_gm,T,D]
            midi_embeds[gm_mask] = gm_emb  # broadcast back into [B,P,T,D]

        # Drums
        dr_mask = is_drum
        if dr_mask.any():
            dr_emb = self.drums_embedding(pitch_tokens[dr_mask])  # -> [N_dr,T,D]
            midi_embeds[dr_mask] = dr_emb

        out = midi_embeds + self.velocity_embedding(velocity_tokens)
        out = out + self.duration_log1p_proj(torch.log1p(note_durations_beats).unsqueeze(-1))
        out = out + self.get_instrument_embedding(program_id).unsqueeze(-2)
        return out * attention_mask.float().unsqueeze(-1)
