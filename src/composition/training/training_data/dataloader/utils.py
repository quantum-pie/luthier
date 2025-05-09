import bisect
import torch

PAD_VALUE = 0
PAD_TRACK_ID = -1
MAX_BARS = 512
MAX_SEQ_LEN = 4096


def pad_and_truncate(seq, max_len, pad_value=PAD_VALUE):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_value] * (max_len - len(seq))


def collate_fn(batch):
    bar_boundaries_batch = []
    bar_tempos_batch = []
    global_attention_mask_batch = []
    track_data_batch = []
    track_programs_batch = []

    # find max len of global latent grid throughout batch
    for item in batch:
        tracks = item["tracks"]
        bar_tempos = item["bar_tempos"]
        bar_bounds = item["bar_boundaries"]

        valid_bars_end = min(MAX_BARS, len(bar_bounds))

        bar_tempos_padded = pad_and_truncate(bar_tempos, MAX_BARS)
        bar_bounds_padded = pad_and_truncate(bar_bounds, MAX_BARS)

        bar_tempos_batch.append(torch.tensor(bar_tempos_padded, dtype=torch.float))
        bar_boundaries_batch.append(torch.tensor(bar_bounds_padded, dtype=torch.float))

        global_attention_mask = [
            1 if i < len(bar_bounds) else 0 for i in range(MAX_BARS)
        ]

        global_attention_mask_batch.append(
            torch.tensor(global_attention_mask, dtype=torch.long)
        )

        sample_tracks = []
        sample_programs = []

        for track in tracks:
            # Pad all per-note sequences

            # Make sure that notes beyond max valid bar (due to global grid size limitation)
            # are masked out. It is important because in the model we will later do mapping
            # from notes to global grid to mix latent into generation, therefore latent
            # information must be available for all notes.
            idx = bisect.bisect_left(track["bar_positions"], valid_bars_end)

            bar_positions = pad_and_truncate(track["bar_positions"][:idx], MAX_SEQ_LEN)
            pitch_tokens = pad_and_truncate(track["pitch_tokens"][:idx], MAX_SEQ_LEN)
            velocity_tokens = pad_and_truncate(
                track["velocity_tokens"][:idx], MAX_SEQ_LEN
            )
            beat_positions = pad_and_truncate(
                track["beat_positions"][:idx], MAX_SEQ_LEN
            )

            within_bar_positions = pad_and_truncate(
                track["within_bar_positions"][:idx], MAX_SEQ_LEN
            )
            note_durations_beats = pad_and_truncate(
                track["note_durations_beats"][:idx], MAX_SEQ_LEN
            )

            attention_mask = [
                1 if i < len(track["pitch_tokens"][:idx]) else 0
                for i in range(MAX_SEQ_LEN)
            ]

            bar_activations_int = [int(flag) for flag in track["bar_activations"]]
            bar_activations = pad_and_truncate(bar_activations_int, MAX_BARS)

            track_dict = {
                "program_id": track["program_id"],
                "pitch_tokens": torch.tensor(pitch_tokens, dtype=torch.long),
                "velocity_tokens": torch.tensor(velocity_tokens, dtype=torch.long),
                "beat_positions": torch.tensor(beat_positions, dtype=torch.float),
                "bar_positions": torch.tensor(bar_positions, dtype=torch.long),
                "within_bar_positions": torch.tensor(
                    within_bar_positions, dtype=torch.float
                ),
                "note_durations_beats": torch.tensor(
                    note_durations_beats, dtype=torch.float
                ),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "bar_activations": torch.tensor(bar_activations, dtype=torch.long),
            }
            sample_tracks.append(track_dict)
            sample_programs.append(track["program_id"])

        track_data_batch.append(sample_tracks)
        track_programs_batch.append(sample_programs)

    return {
        "tracks": track_data_batch,
        "track_programs": track_programs_batch,
        "bar_boundaries": bar_boundaries_batch,
        "bar_tempos": bar_tempos_batch,
        "global_attention_mask": global_attention_mask_batch,
    }


def prepare_batch_for_model(batch):
    batch_size = len(batch["tracks"])
    max_tracks = max(len(tracks) for tracks in batch["tracks"])

    # Initialize output tensors
    def init_track_tensor(dtype):
        return torch.full((batch_size, max_tracks, MAX_SEQ_LEN), PAD_VALUE, dtype=dtype)

    pitch_tokens = init_track_tensor(torch.long)
    velocity_tokens = init_track_tensor(torch.long)
    beat_positions = init_track_tensor(torch.float)
    bar_positions = init_track_tensor(torch.long)
    within_bar_positions = init_track_tensor(torch.float)
    note_durations_beats = init_track_tensor(torch.float)
    attention_masks = init_track_tensor(torch.long)
    bar_activations = torch.full(
        (batch_size, max_tracks, MAX_BARS), PAD_VALUE, dtype=torch.long
    )

    track_mask = torch.zeros((batch_size, max_tracks), dtype=torch.bool)
    program_ids = torch.full((batch_size, max_tracks), PAD_TRACK_ID, dtype=torch.long)

    bar_boundaries = torch.stack(batch["bar_boundaries"], dim=0)
    bar_tempos = torch.stack(batch["bar_tempos"], dim=0)
    global_attention_mask = torch.stack(batch["global_attention_mask"], dim=0)

    for i, sample_tracks in enumerate(batch["tracks"]):
        for j, track in enumerate(sample_tracks):
            pitch_tokens[i, j] = track["pitch_tokens"]
            velocity_tokens[i, j] = track["velocity_tokens"]
            beat_positions[i, j] = track["beat_positions"]
            bar_positions[i, j] = track["bar_positions"]
            within_bar_positions[i, j] = track["within_bar_positions"]
            note_durations_beats[i, j] = track["note_durations_beats"]
            attention_masks[i, j] = track["attention_mask"]
            track_mask[i, j] = True
            program_ids[i, j] = track["program_id"]
            bar_activations[i, j] = track["bar_activations"]

    return {
        "pitch_tokens": pitch_tokens,
        "velocity_tokens": velocity_tokens,
        "beat_positions": beat_positions,
        "bar_positions": bar_positions,
        "within_bar_positions": within_bar_positions,
        "note_durations_beats": note_durations_beats,
        "attention_mask": attention_masks,
        "track_mask": track_mask,
        "program_ids": program_ids,
        "bar_activations": bar_activations,
        "bar_boundaries": bar_boundaries,
        "bar_tempos": bar_tempos,
        "global_attention_mask": global_attention_mask,
    }
