import bisect
import os
import tempfile
import torch

PAD_VALUE = 0
MAX_BARS = 512
MAX_SEQ_LEN = 4096
MAX_TRACKS = 50


def atomic_write(path: str, write_fn):
    """
    Atomically write to `path` using a temporary file in the same directory.
    `write_fn` should accept a binary file object and write all data to it.
    No leftover temp files on error/KeyboardInterrupt.
    """
    dst_dir = os.path.dirname(path) or "."
    os.makedirs(dst_dir, exist_ok=True)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=dst_dir, delete=False) as tmp:
            write_fn(tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name
        os.replace(tmp_path, path)  # atomic on same filesystem
    finally:
        # If replace failed or we were interrupted, remove the temp file.
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def pad_and_truncate(seq, max_len, pad_value=PAD_VALUE):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_value] * (max_len - len(seq))


def collate_fn(input_and_control_batch):
    bar_boundaries_batch = []
    bar_tempos_batch = []
    global_attention_mask_batch = []
    track_data_batch = []
    track_programs_batch = []

    # find max len of global latent grid throughout batch
    for input_and_control_item in input_and_control_batch:
        item = input_and_control_item["input"]
        tracks = item["tracks"]

        if len(tracks) > MAX_TRACKS:
            tracks = tracks[:MAX_TRACKS]

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
            torch.tensor(global_attention_mask, dtype=torch.bool)
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
                "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
                "bar_activations": torch.tensor(bar_activations, dtype=torch.long),
            }
            sample_tracks.append(track_dict)
            sample_programs.append(track["program_id"])

        track_data_batch.append(sample_tracks)
        track_programs_batch.append(sample_programs)

    input_collated = {
        "tracks": track_data_batch,
        "track_programs": track_programs_batch,
        "bar_boundaries": bar_boundaries_batch,
        "bar_tempos": bar_tempos_batch,
        "global_attention_mask": global_attention_mask_batch,
    }

    # TODO: include control later from input_and_control_batch["control"]. If any is None there, then return None
    conrol_collated = None

    return {
        "input": input_collated,
        "control": conrol_collated,
    }


def prepare_batch_for_model(input_and_control_batch):
    input_batch = input_and_control_batch["input"]

    batch_size = len(input_batch["tracks"])
    max_tracks = max(len(tracks) for tracks in input_batch["tracks"])

    # Initialize output tensors
    def init_track_tensor(dtype):
        return torch.full((batch_size, max_tracks, MAX_SEQ_LEN), PAD_VALUE, dtype=dtype)

    pitch_tokens = init_track_tensor(torch.long)
    velocity_tokens = init_track_tensor(torch.long)
    beat_positions = init_track_tensor(torch.float)
    bar_positions = init_track_tensor(torch.long)
    within_bar_positions = init_track_tensor(torch.float)
    note_durations_beats = init_track_tensor(torch.float)
    attention_masks = init_track_tensor(torch.bool)
    bar_activations = torch.full(
        (batch_size, max_tracks, MAX_BARS), PAD_VALUE, dtype=torch.long
    )

    track_mask = torch.zeros((batch_size, max_tracks), dtype=torch.bool)
    program_ids = torch.full((batch_size, max_tracks), PAD_VALUE, dtype=torch.long)

    bar_boundaries = torch.stack(input_batch["bar_boundaries"], dim=0)
    bar_tempos = torch.stack(input_batch["bar_tempos"], dim=0)
    global_attention_mask = torch.stack(input_batch["global_attention_mask"], dim=0)

    for i, sample_tracks in enumerate(input_batch["tracks"]):
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

    input_batch_prepared = {
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

    # When control batch is not None, create control tensors:
    # genre_ids, genre_mask, mood_ids, mood_mask
    control_batch_prepared = None

    return {
        "input": input_batch_prepared,
        "control": control_batch_prepared,
    }
